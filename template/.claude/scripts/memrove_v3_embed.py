"""Pluggable embedders and the record-vector cache for message-time recall (stage 1, dense half). Stdlib only.

Embedders (all return L2-normalised array('f') vectors):
  ngram   built in: hashed word + char-trigram vectors; no download, ~0.1 ms per record; the always-available floor.
  static  Model2Vec-style static model (potion-multilingual-128M): token-embedding lookup + mean, in process, no daemon.
          Weights are installed once by `install_static()`; the hook only mmaps rows (resident cost is the touched pages).
  daemon  EmbeddingGemma 2 (MLX mxfp8) behind the model daemon (memrove_modeld.py); fail-open: unavailable -> caller falls back.
Record vectors live in <state>/vectors.sqlite3 keyed by content hash (+ a record-id link), computed by `refresh()` at sync time,
never in the hook. A hook that finds no vectors simply has no dense candidates."""
import array
import hashlib
import json
import math
import mmap
import os
import re
import socket
import sqlite3
import struct
import subprocess
import sys
import time
import unicodedata
import zlib
from pathlib import Path

EMBED_CHARS = 700
STATIC_NAME = "potion-multilingual-128M"
STATIC_FILES = ("model.safetensors", "tokenizer.json")
STATIC_URL = "https://huggingface.co/minishlab/potion-multilingual-128M/resolve/main/"


def models_dir():
    return Path(os.environ.get("MEMROVE_MODELS") or Path.home() / ".local/share/memrove/models")


def sock_path():
    return os.environ.get("MEMROVE_MODELD_SOCK") or str(Path.home() / ".local/share/memrove/modeld.sock")


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return array.array("f", (x / n for x in v) if n else v)


def cosine(a, b):
    """Dot product of two unit vectors."""
    return sum(map(float.__mul__, a, b))


def clip_query(text, n=800):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n // 2] + " " + text[-n // 2:]


# --- ngram ---------------------------------------------------------------------------------------------------------
class NgramEmbedder:
    name, dim = "ngram-v1", 256

    def embed(self, texts):
        from memrove_v3_recall import STOP, fold
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9_]{3,}", fold(text)):
                if w in STOP:
                    continue
                feats = ["w:" + w[:5]] + [("c:" + p[i:i + 3]) for p in ("^" + w + "$",) for i in range(len(p) - 2)]
                for f in feats:
                    h = zlib.crc32(f.encode())
                    v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            out.append(_unit(v))
        return out


# --- static (Model2Vec-style) -----------------------------------------------------------------------------------------
class StaticEmbedder:
    """Unigram (sentencepiece-style) tokenizer over a sqlite piece table + mean of f16 token embeddings (mmap)."""
    name = "static-" + STATIC_NAME
    MAXLEN = 16

    def __init__(self, folder=None):
        self.dir = Path(folder or models_dir() / STATIC_NAME)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.dim, self.unk = meta["dim"], meta.get("unk", 1)
        self._f = open(self.dir / "emb.f16", "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._db = sqlite3.connect(f"file:{self.dir / 'vocab.sqlite'}?mode=ro", uri=True)
        self._row = struct.Struct(f"<{self.dim}e")

    @staticmethod
    def installed(folder=None):
        d = Path(folder or models_dir() / STATIC_NAME)
        return all((d / n).is_file() for n in ("meta.json", "emb.f16", "vocab.sqlite"))

    def tokenize(self, text):
        s = "▁" + "▁".join(unicodedata.normalize("NFKC", text).split())
        n = len(s)
        best = [(0.0, 0, -1)] + [(-1e18, 0, -1)] * n  # (score, prev position, token id) for prefixes of s
        q = self._db.execute
        for i in range(n):
            base = best[i][0]
            if base <= -1e17:
                continue
            subs = [s[i:j] for j in range(i + 1, min(n, i + self.MAXLEN) + 1)]
            rows = q(f"SELECT piece, id, score FROM vocab WHERE piece IN ({','.join('?' * len(subs))})", subs).fetchall()
            if not any(len(p) == 1 for p, _, _ in rows):  # unknown character: skip it with a penalty
                if base - 10.0 > best[i + 1][0]:
                    best[i + 1] = (base - 10.0, i, self.unk)
            for piece, tid, sc in rows:
                j = i + len(piece)
                if base + sc > best[j][0]:
                    best[j] = (base + sc, i, tid)
        ids, j = [], n
        while j > 0:
            _, i, tid = best[j]
            ids.append(tid)
            j = i
        return [t for t in reversed(ids) if t > 1]  # drop pad/unk

    def embed(self, texts):
        out = []
        for text in texts:
            ids = self.tokenize(clip_query(text, 1200))
            acc = [0.0] * self.dim
            for t in ids:
                for k, x in enumerate(self._row.unpack_from(self._mm, t * self._row.size)):
                    acc[k] += x
            out.append(_unit(acc))
        return out


def install_static(src_dir=None, progress=print):
    """Download (or take from `src_dir`) the potion-multilingual-128M safetensors + tokenizer and convert them to the compact
    layout StaticEmbedder reads: emb.f16 (256 MB instead of 512 MB), vocab.sqlite (piece -> id, score), meta.json."""
    dest = models_dir() / STATIC_NAME
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(src_dir) if src_dir else dest / "_src"
    work.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FILES:
        if not (work / name).is_file():
            progress(f"download {name}")
            import urllib.request
            urllib.request.urlretrieve(STATIC_URL + name, work / name)
    st = work / "model.safetensors"
    with st.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        info = next(v for k, v in header.items() if k != "__metadata__" and len(v["shape"]) == 2)
        rows, dim = info["shape"]
        assert info["dtype"] in ("F32", "F16"), info["dtype"]
        f.seek(8 + hlen + info["data_offsets"][0])
        width = 4 if info["dtype"] == "F32" else 2
        fmt = "f" if width == 4 else "e"
        with (dest / "emb.f16.tmp").open("wb") as out:
            step = 8192
            for r in range(0, rows, step):
                n = min(step, rows - r)
                vals = struct.unpack(f"<{n * dim}{fmt}", f.read(n * dim * width))
                out.write(struct.pack(f"<{n * dim}e", *[max(-65000.0, min(65000.0, x)) for x in vals]))
    os.replace(dest / "emb.f16.tmp", dest / "emb.f16")
    tok = json.loads((work / "tokenizer.json").read_text(encoding="utf-8"))["model"]
    tmp = dest / "vocab.sqlite.tmp"
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.execute("CREATE TABLE vocab(piece TEXT PRIMARY KEY, id INTEGER, score REAL) WITHOUT ROWID")
    db.executemany("INSERT OR IGNORE INTO vocab VALUES(?,?,?)", ((p, i, s) for i, (p, s) in enumerate(tok["vocab"])))
    db.commit()
    db.close()
    os.replace(tmp, dest / "vocab.sqlite")
    (dest / "meta.json").write_text(json.dumps({"dim": dim, "rows": rows, "unk": tok.get("unk_id", 1), "source": STATIC_URL}), encoding="utf-8")
    if not src_dir:
        for name in STATIC_FILES:
            (work / name).unlink(missing_ok=True)
    progress(f"installed {dest}")
    return dest


# --- daemon ----------------------------------------------------------------------------------------------------------
def daemon_call(request, timeout=1.2):
    """One JSON line to the model daemon -> its reply dict, or None (not running, slow, or errored): callers fail open."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        s.connect(sock_path())
        s.sendall((json.dumps(request) + "\n").encode())
        buf = b""
        end = time.monotonic() + timeout
        while not buf.endswith(b"\n"):
            s.settimeout(max(0.01, end - time.monotonic()))
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
        reply = json.loads(buf)
        return reply if reply.get("ok") else None
    except (OSError, ValueError):
        return None
    finally:
        s.close()


def spawn_daemon(python=None, idle=600):
    """Start the daemon detached (it exits after `idle` seconds without a request). Returns immediately; never raises."""
    try:
        marker = Path(sock_path() + ".spawned")  # at most one spawn per minute: a broken daemon must not fork-storm the hook
        if marker.exists() and time.time() - marker.stat().st_mtime < 60:
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        script = Path(__file__).with_name("memrove_modeld.py")
        env = dict(os.environ, OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")  # fanless laptop: keep the CPU side of inference small
        subprocess.Popen([python or sys.executable, str(script), "--idle", str(idle)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True, env=env)
    except OSError:
        pass


class DaemonEmbedder:
    """EmbeddingGemma 2 text tower (MLX mxfp8, 768-d) behind the model daemon. Task prefixes are applied by the daemon."""
    FORMAT = "f2"  # bump with any change of prefix, title slot or text cap: it is part of the vector name and the content-hash key, so stale vectors are never reused
    name, dim, model = f"daemon-embeddinggemma-2-mxfp8-{FORMAT}", 768, "gemma"
    titled, fmt = True, FORMAT  # documents go in as `title: {title} | text: {body}` with no 700-char cap (the daemon truncates at 512 tokens)
    DOC_CHARS = 4000

    def __init__(self, timeout=0.4):
        self.timeout = timeout

    def embed(self, texts, kind="doc", titles=None):
        texts = [clip_query(t, 1500) for t in texts] if kind == "query" else [" ".join(str(t).split())[:self.DOC_CHARS] for t in texts]
        req = {"op": "embed", "model": self.model, "kind": kind, "texts": texts}
        if titles:
            req["titles"] = [" ".join(str(t or "").split())[:200] for t in titles]
        reply = daemon_call(req, self.timeout if len(texts) < 4 else 60)
        if not reply:
            raise OSError("embed daemon unavailable")
        return [_unit(v) for v in reply["vectors"]]

    def embed_query(self, texts):  # queries carry the retrieval task prefix, records the document one
        return self.embed(texts, "query")


def get_embedder(spec=None):
    """spec: ngram | static | daemon | auto (static when installed, else ngram). Raises only for an explicit spec that cannot load."""
    spec = spec or os.environ.get("MEMROVE_EMBEDDER") or "auto"
    if spec == "ngram":
        return NgramEmbedder()
    if spec in ("daemon", "daemon-gemma"):
        return DaemonEmbedder()
    if spec == "static" or (spec == "auto" and StaticEmbedder.installed()):
        return StaticEmbedder()
    return NgramEmbedder()


# --- vector cache ----------------------------------------------------------------------------------------------------
def embed_text(r, full=False):
    """The record as one string; capped at EMBED_CHARS for the static model (full=True: no cap, for embedders with their own length limit)."""
    from memrove_v3_recall import _body, _list
    parts = [str(r.get("title") or ""), " ".join(_list(r.get("topics"))), _body(r)]
    text = " ".join(p for p in parts if p)
    return text if full else text[:EMBED_CHARS]


def doc_input(r, embedder):
    """-> (title or None, text) as the embedder wants a document. A `titled` embedder gets the title in its own slot, not repeated in the text."""
    if not getattr(embedder, "titled", False):
        return None, embed_text(r)
    from memrove_v3_recall import _body, _list
    text = " ".join(p for p in (" ".join(_list(r.get("topics"))), _body(r)) if p)
    return str(r.get("title") or ""), text


def vec_key(embedder, title, text):
    """Cache key of a document vector: the embedder's format version plus exactly what it embeds (unchanged for static / ngram)."""
    return _chash(getattr(embedder, "fmt", "") + (text if title is None else title + "\0" + text))


def _chash(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


class VecStore:
    def __init__(self, state, readonly=False):
        self.path = Path(state) / "vectors.sqlite3"
        self.readonly = readonly

    def _db(self):
        if self.readonly:
            return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=1)
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS vec(h TEXT, model TEXT, v BLOB, PRIMARY KEY(h, model)) WITHOUT ROWID")
        db.execute("CREATE TABLE IF NOT EXISTS link(id TEXT, model TEXT, h TEXT, PRIMARY KEY(id, model)) WITHOUT ROWID")
        return db

    def load(self, model):
        """-> {record id: unit vector}. Empty when nothing has been computed yet."""
        if not self.path.is_file():
            return {}
        try:
            db = self._db()
            try:
                rows = db.execute("SELECT link.id, vec.v FROM link JOIN vec ON vec.h=link.h AND vec.model=link.model WHERE link.model=?", (model,)).fetchall()
            finally:
                db.close()
        except sqlite3.Error:
            return {}
        out = {}
        for rid, blob in rows:
            a = array.array("f")
            a.frombytes(blob)
            out[rid] = a
        return out

    def refresh(self, records, embedder, limit=None, prune=True, budget_s=None):
        """records: [(id, record dict)]. Embeds only content hashes the cache has not seen, relinks ids, drops vanished ones.
        -> {"new": n, "total": n}."""
        db = self._db()
        try:
            have = {h for (h,) in db.execute("SELECT h FROM vec WHERE model=?", (embedder.name,))}
            todo, links = {}, []
            for rid, r in records:
                inp = doc_input(r, embedder)
                h = vec_key(embedder, *inp)
                links.append((rid, embedder.name, h))
                if h not in have and h not in todo:
                    todo[h] = inp
            hs = list(todo)[:limit] if limit else list(todo)
            t0, done = time.monotonic(), []
            for i in range(0, len(hs), 16):
                if budget_s and time.monotonic() - t0 > budget_s:
                    break
                batch = hs[i:i + 16]
                texts = [todo[h][1] for h in batch]
                vs = embedder.embed(texts, titles=[todo[h][0] for h in batch]) if getattr(embedder, "titled", False) else embedder.embed(texts)
                for h, v in zip(batch, vs):
                    db.execute("INSERT OR REPLACE INTO vec VALUES(?,?,?)", (h, embedder.name, v.tobytes()))
                done += batch
            hs = done
            if prune:  # a full record list (sync time): vanished ids and unlinked vectors go
                db.execute("DELETE FROM link WHERE model=?", (embedder.name,))
            db.executemany("INSERT OR REPLACE INTO link VALUES(?,?,?)", [l for l in links if l[2] in have or l[2] in todo and l[2] in hs])
            if prune:
                db.execute("DELETE FROM vec WHERE model=? AND h NOT IN (SELECT h FROM link WHERE model=?)", (embedder.name, embedder.name))
            db.commit()
            return {"new": len(hs), "total": len(links)}
        finally:
            db.close()


def refresh(state, spec=None, limit=None):
    """Sync-time / background entry: embed new or changed records. Fail-open (returns {"error": ...}); never called from the hook.
    With the cascade on, the EG2 vectors (L2) are built too, a slice per call (`l2_limit`), so a fresh store converges over a few syncs
    and every later one embeds only new records; `python memrove_v3_embed.py <state>` backfills in one go."""
    try:
        import json as _json
        from memrove_v3_recall import StateStore, _load
        from memrove_v3_hybrid import config
        cfg = config(state)
        embedder = get_embedder(spec or cfg["embedder"])
        records = [(rid, _json.loads(payload)) for payload, _, rid in _load(StateStore(state))]
        out = VecStore(state).refresh(records, embedder, limit)
        if cfg.get("cascade") and not isinstance(embedder, DaemonEmbedder) and ensure_daemon(cfg.get("daemon_python"), cfg.get("daemon_idle", 600)):
            out["l2"] = VecStore(state).refresh(records, DaemonEmbedder(), limit or cfg.get("l2_limit", 128))
        return out
    except Exception as exc:
        return {"error": type(exc).__name__}


def ensure_daemon(python=None, idle=600, wait=20.0):
    """Background paths only (the hook never waits): start the daemon if nothing answers, then wait for its socket. -> True when it answers."""
    if daemon_call({"op": "ping"}, 0.5):
        return True
    spawn_daemon(python, idle)
    end = time.monotonic() + wait
    while time.monotonic() < end:
        time.sleep(0.5)
        if daemon_call({"op": "ping"}, 0.5):
            return True
    return False


if __name__ == "__main__":  # backfill: python memrove_v3_embed.py <state dir> [limit]   (run under nice; the first call loads EG2)
    print(json.dumps(refresh(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6)))
