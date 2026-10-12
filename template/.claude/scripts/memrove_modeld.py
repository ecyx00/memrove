#!/usr/bin/env python3
"""memrove model daemon: EmbeddingGemma 2 (text tower, MLX mxfp8) behind a unix socket. Optional; the default embedder is the in-process static one.
Needs mlx-vlm at the commit pinned on the model card (PyPI 0.7.6 does not know `embedding_gemma2`), so run it with the venv python
(~/.local/share/memrove/venv/bin/python; recall config `daemon_python`). Not a memrove_v3 module on purpose: the packaged runtime stays stdlib-only.
Protocol: one JSON line in, one JSON line out.  {"op":"ping"} | {"op":"embed","texts":[..],"kind":"query|doc","titles":[..]} | {"op":"rerank",...}
`rerank` is the stage-2 slot: no model is configured, so it answers ok=false and the hook falls back to stage 1.
The hook is the client and fails open. The model loads on first use; the process exits after --idle seconds without a request (--idle 0: stays resident).
Slim by default (text tower + compact tokenizer, ~330 MB RSS); MEMROVE_GEMMA_SLIM=0 loads the full multimodal model (~1.6 GB)."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

GEMMA_REPO = os.environ.get("MEMROVE_GEMMA_MODEL", "mlx-community/embeddinggemma-2-mxfp8")  # needs mlx-vlm; bf16/fp32 activations only (fp16 gives NaN)
SLIM = os.environ.get("MEMROVE_GEMMA_SLIM", "1") != "0"  # default: text tower only + the compact tokenizer below (~330 MB RSS); MEMROVE_GEMMA_SLIM=0 loads the whole multimodal model (~1.6 GB)
MODELS = {}


# --- compact BPE tokenizer: the HF `tokenizers` object for this 262k vocab / 515k merges costs ~400 MB resident ------------------
# Same result as tokenizer.json (Replace ' '->'▁', BPE with byte fallback, <bos>/<eos>, 512 cut), but the merge table is two sorted numpy
# arrays memory-mapped from a one-off .npz built from tokenizer.json in a throw-away process.
class CompactBPE:
    def __init__(self, npz):
        import numpy as np
        z = np.load(npz, allow_pickle=False)
        self.keys, self.res = z["keys"], z["res"]  # sorted pair keys a*2^18+b; res = rank*2^18 + merged id (rank orders merges)
        chars = z["chars"].tolist()
        self.char_id = dict(zip(chars, z["char_ids"].tolist()))
        self.byte_id = z["byte_ids"].tolist()  # id of <0xXX> for each byte value
        self.special = dict(zip(z["sp_tok"].tolist(), z["sp_ids"].tolist()))
        import re
        self.sp_re = re.compile("(" + "|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)) + ")")
        self.np = np

    @staticmethod
    def build(tokenizer_json, npz):
        import numpy as np
        t = json.load(open(tokenizer_json, encoding="utf-8"))
        vocab = t["model"]["vocab"]
        arr = np.empty(len(t["model"]["merges"]), dtype=np.int64)
        res = np.empty(len(arr), dtype=np.int64)
        for rank, m in enumerate(t["model"]["merges"]):
            a, b = m if isinstance(m, list) else m.split(" ", 1)
            arr[rank] = (vocab[a] << 18) | vocab[b]
            res[rank] = (rank << 18) | vocab[a + b]
        order = np.argsort(arr, kind="stable")
        chars = [c for c in vocab if len(c) == 1]
        byte_ids = [vocab.get("<0x%02X>" % i, -1) for i in range(256)]
        sp = {a["content"]: a["id"] for a in t["added_tokens"]}
        tmp = str(npz) + ".tmp.npz"
        np.savez(tmp, keys=arr[order], res=res[order], chars=np.array(chars), char_ids=np.array([vocab[c] for c in chars]),
                 byte_ids=np.array(byte_ids), sp_tok=np.array(list(sp)), sp_ids=np.array(list(sp.values())))
        os.replace(tmp, npz)

    def _merge_word(self, ids):
        """Lowest-rank pair first, leftmost on ties (what `tokenizers` does); a doubly linked list over `ids` with a heap of candidate pairs."""
        import heapq
        np, keys, res = self.np, self.keys, self.res
        n = len(ids)
        if n < 2:
            return ids
        prev, nxt, ids, alive = list(range(-1, n - 1)), list(range(1, n + 1)), list(ids), [True] * n

        def look(i):
            j = nxt[i]
            if j >= n:
                return None
            k = (ids[i] << 18) | ids[j]
            p = int(keys.searchsorted(k))
            if p < len(keys) and keys[p] == k:
                r = int(res[p])
                return (r >> 18, i, j, r & 0x3FFFF, ids[i], ids[j])
            return None
        heap = [h for h in map(look, range(n - 1)) if h]
        heapq.heapify(heap)
        while heap:
            _, i, j, new, ai, bj = heapq.heappop(heap)
            if not (alive[i] and alive[j] and nxt[i] == j and ids[i] == ai and ids[j] == bj):
                continue  # stale candidate
            ids[i] = new
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            for x in (prev[i] if prev[i] >= 0 else None, i):
                if x is not None:
                    h = look(x)
                    if h:
                        heapq.heappush(heap, h)
        out, i = [], 0
        while i < n:
            out.append(ids[i])
            i = nxt[i]
        return out

    def _piece(self, text):
        ids = []
        for ch in text.replace(" ", "\u2581"):
            cid = self.char_id.get(ch)
            if cid is not None:
                ids.append(cid)
            else:
                ids.extend(self.byte_id[b] for b in ch.encode("utf-8"))
        return self._merge_word(ids)

    def encode(self, text, max_length=512):
        """-> [<bos>, ..., <eos>], cut to max_length with the <eos> kept (HF truncation of a single sequence)."""
        out = []
        for k, seg in enumerate(self.sp_re.split(text)):
            if k % 2:
                out.append(self.special[seg])
            elif seg:
                out.extend(self._piece(seg))
        return [2] + out[:max_length - 2] + [1]


def _compact_tokenizer(path):
    npz = Path(os.environ.get("MEMROVE_MODELS") or Path.home() / ".local/share/memrove/models") / "eg2-bpe.npz"
    if not npz.exists():  # one-off, in its own process so the json tree never counts against the daemon
        npz.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--build-bpe", str(Path(path) / "tokenizer.json"), str(npz)], check=True,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return CompactBPE(npz)


def _load_gemma():
    import mlx.core as mx
    mx.set_cache_limit(32 << 20)  # MLX keeps freed buffers for reuse; short texts need little
    from mlx_vlm.utils import get_model_path
    path = get_model_path(GEMMA_REPO)
    if not SLIM:
        from mlx_vlm.embedding_loader import load_embedding_model
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        return load_embedding_model(path), lambda text: tok([text], truncation=True, max_length=512)["input_ids"]
    from mlx_vlm.encoder_loader import load_encoder_model
    from mlx_vlm.utils import load_config
    cfg = load_config(path)
    cfg["vision_config"] = cfg["audio_config"] = None  # no vision / audio towers: their weights are never read from the mmapped file
    bpe = _compact_tokenizer(path)
    model = load_encoder_model(path, model_remapping={}, config=cfg, lazy=True)
    from mlx.utils import tree_flatten
    for _, w in tree_flatten(model.parameters()):  # one tensor at a time: evaluating everything at once spikes phys_footprint to ~700 MB
        mx.eval(w)
        mx.clear_cache()
    class _Eager:  # evaluate after every layer: one lazy 24-layer graph over 512 tokens peaks ~250 MB above the weights
        def __init__(self, layer):
            self.layer = layer

        def __call__(self, *a, **k):
            h = self.layer(*a, **k)
            mx.eval(h)
            return h
    model.language_model.layers = [_Eager(layer) for layer in model.language_model.layers]
    return model, lambda text: [bpe.encode(text)]


def sock_path():
    return os.environ.get("MEMROVE_MODELD_SOCK") or str(Path.home() / ".local/share/memrove/modeld.sock")


def embed_gemma(texts, kind="doc", titles=None):
    """EmbeddingGemma 2 (text tower): task prefixes from its card, 768-d, already unit length."""
    import mlx.core as mx
    if GEMMA_REPO not in MODELS:
        MODELS[GEMMA_REPO] = _load_gemma()
    model, encode = MODELS[GEMMA_REPO]
    out = []
    for i, t in enumerate(texts):  # documents: the card's `title: {title} | text: {content}`; `none` only when the record has no title
        title = (titles[i] if titles and i < len(titles) else "") or "none"
        text = f"task: search result | query: {t}" if kind == "query" else f"title: {title} | text: {t}"
        ids = mx.array(encode(text))
        out.append([float(x) for x in model(input_ids=ids, attention_mask=mx.ones_like(ids)).text_embeds[0].astype(mx.float32).tolist()])
    return out


def handle(req):
    op = req.get("op")
    if op == "ping":
        r = {"ok": True, "pid": os.getpid(), "warm": bool(MODELS)}  # warm = the model is already loaded; a cascade client asks before it embeds
        if MODELS:
            import mlx.core as mx
            r["mlx_mb"] = {k: round(f() / 1e6, 1) for k, f in (("active", mx.get_active_memory), ("cache", mx.get_cache_memory), ("peak", mx.get_peak_memory))}
        return r
    if op == "embed":
        return {"ok": True, "vectors": embed_gemma(req["texts"], req.get("kind", "doc"), req.get("titles"))}
    if op == "rerank":
        return {"ok": False, "error": "no reranker configured"}
    return {"ok": False, "error": "unknown op"}


def serve(idle):
    path = sock_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(path):
        probe = socket.socket(socket.AF_UNIX)
        try:
            probe.connect(path)
            return  # another daemon is alive
        except OSError:
            os.unlink(path)
        finally:
            probe.close()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(4)
    srv.settimeout(1.0)
    if idle <= 0:  # resident: load EG2 now (requests queue on the socket meanwhile), so the first cascade ping already finds it warm
        try:
            embed_gemma(["warm"], "query")
        except Exception:
            pass
    last = time.monotonic()
    try:
        while idle <= 0 or time.monotonic() - last < idle:  # --idle 0: resident
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(30)
                try:
                    buf = b""
                    while not buf.endswith(b"\n"):
                        chunk = conn.recv(1 << 16)
                        if not chunk:
                            break
                        buf += chunk
                    reply = handle(json.loads(buf))
                except Exception as exc:  # a bad request must not kill the daemon
                    reply = {"ok": False, "error": type(exc).__name__}
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    pass
            last = time.monotonic()
    finally:
        srv.close()
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    import faulthandler
    import signal
    faulthandler.enable()  # a crash leaves a traceback in the daemon's stderr instead of vanishing
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=600, help="seconds without a request before exit; 0 = stay resident (cascade L2)")
    ap.add_argument("--build-bpe", nargs=2, metavar=("TOKENIZER_JSON", "NPZ"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.build_bpe:
        CompactBPE.build(*a.build_bpe)
    else:
        serve(a.idle)
