#!/usr/bin/env python3
"""Local reranker / embedder bench for memrove (C5 in docs/design/2026-11-07-memrove-engine.md). One model per process, so ru_maxrss is that model's peak.

  /tmp/lme/models-venv/bin/python bench/models.py run --model hs-rerank-1      # reranker: top 30 of memrove-chatlog, LME-S + golden
  /tmp/lme/models-venv/bin/python bench/models.py run --model harrier-270m     # embedder: RRF(bm25, dense) like cs._hybrid, LME-S subset + golden
  /tmp/lme/models-venv/bin/python bench/models.py table

Inputs: /tmp/lme/models-out/memrove-chatlog.jsonl (from `bench/lme.py --arms memrove-chatlog --out /tmp/lme/models-out`), the LME-S json, the golden copy
/tmp/lme/livecopy (never the live state). eg2 needs a private daemon: MEMROVE_MODELD_SOCK=/tmp/lme/models-sock python3.14 memrove_modeld.py --idle 3600.
Engine code is imported from template/.claude/scripts. Golden queries and the chat log never leave the machine: HF is reached only to download weights.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", "/tmp/lme/models-hf")   # anonymous: no token file is read from here
import lme   # noqa: E402

REAL_HOME = Path.home()   # lme.setup() repoints HOME at a temp dir

DATA = "/tmp/lme/data/longmemeval_s_cleaned.json"
RANKS = "/tmp/lme/models-out/memrove-chatlog.jsonl"
STATE = "/tmp/lme/livecopy"
GOLDEN = "/Users/ecy/Desktop/personal-intelligence/tools/audit/memrecall/golden.jsonl"
RES = Path("/tmp/lme/models-results")
POOL, DOC_CHARS = 30, 1200   # rerank / dense-rescore depth; chars of a document any model sees
RERANK_TASK = "Given a search query, retrieve the chat message or memory record that answers it"
MODELS = {   # name -> (kind, repo)
    "hs-rerank-1": ("rerank", "hindsight-ai/hs-rerank-1"),
    "qwen3-rerank-0.6b": ("rerank", "Qwen/Qwen3-Reranker-0.6B"),
    "hs-embed-1": ("embed", "hindsight-ai/hs-embed-1"),
    "harrier-270m": ("embed", "microsoft/harrier-oss-v1-270m"),
    "eg2": ("embed", "mlx-community/embeddinggemma-2-mxfp8"),
    "bm25": ("none", ""),
}
QTASK = {"lme": "Given a question about earlier conversations, retrieve the chat message that contains the answer",
         "golden": "Given a query, retrieve relevant memory records"}


def fetch(repo):
    """-> (local dir, bytes of the files the model needs). onnx and images are skipped."""
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(repo, ignore_patterns=["onnx/*", "assets/*", "*.png", "*.gguf", "*.msgpack", "*.h5"]))
    return d, sum(f.stat().st_size for f in d.rglob("*") if f.is_file())   # stat follows the snapshot symlinks


class HsRerank:
    def __init__(self, repo):
        import torch
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(repo, device="mps", max_length=512, model_kwargs={"torch_dtype": torch.float16})

    def scores(self, q, docs):
        return [float(x) for x in self.m.predict([(q, d[:DOC_CHARS]) for d in docs], batch_size=16, show_progress_bar=False)]


class QwenRerank:
    PRE = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, repo):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
        self.m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("mps").eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")

    def scores(self, q, docs, bs=4):
        out = []
        for i in range(0, len(docs), bs):
            texts = [f"{self.PRE}<Instruct>: {RERANK_TASK}\n<Query>: {q}\n<Document>: {d[:DOC_CHARS]}{self.SUF}" for d in docs[i:i + bs]]
            enc = self.tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("mps")
            with self.torch.no_grad():
                lg = self.m(**enc, logits_to_keep=1).logits[:, -1, :]   # full-sequence logits (bs x len x 152k vocab) were the swap hog
            out += self.torch.softmax(lg[:, [self.no, self.yes]].float(), dim=1)[:, 1].cpu().tolist()
        return out


class StEmbed:
    """sentence-transformers bi-encoder. qprompt: Harrier needs its instruction on the query side only."""
    def __init__(self, repo, qprompt=None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(repo, device="mps", model_kwargs={"torch_dtype": torch.bfloat16 if qprompt else torch.float16})
        self.m.max_seq_length = 320
        self.qprompt = qprompt

    def _enc(self, texts, **kw):
        import numpy as np
        return np.asarray(self.m.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, **kw), dtype="float32")

    def query(self, texts, task):
        return self._enc(texts, prompt=f"Instruct: {task}\nQuery: ") if self.qprompt else self._enc(texts)

    def docs(self, texts):
        return self._enc([t[:DOC_CHARS] for t in texts])


class Eg2:
    """The production embedder, through a private modeld instance (the live daemon is left alone)."""
    def __init__(self, repo):
        import numpy as np
        import memrove_v3_embed as emb
        self.np, self.d = np, emb.DaemonEmbedder(timeout=120)
        emb.DaemonEmbedder.DOC_CHARS = DOC_CHARS
        assert emb.sock_path().startswith("/tmp/lme/"), "set MEMROVE_MODELD_SOCK to the private daemon"
        self.d.embed(["warm up"], "query")

    def query(self, texts, task):
        return self.np.asarray(self.d.embed_query(texts), dtype="float32")

    def docs(self, texts):
        out = []
        for i in range(0, len(texts), 16):
            out += self.d.embed(texts[i:i + 16])
        return self.np.asarray(out, dtype="float32")


def load(name):
    kind, repo = MODELS[name]
    if name == "bm25":
        return kind, None, 0
    if name == "eg2":
        sub = REAL_HOME / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--"))
        size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
        return kind, Eg2(repo), size
    path, size = fetch(repo)
    cls = {"hs-rerank-1": HsRerank, "qwen3-rerank-0.6b": QwenRerank, "hs-embed-1": StEmbed, "harrier-270m": lambda p: StEmbed(p, qprompt=True)}[name]
    return kind, cls(str(path)), size


def timed_p50(fn, xs, n=40):
    fn(xs[0])   # warm
    ts = []
    for x in xs[:n]:
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def rrf(lists, w=(1, 2), k=60):
    sc = {}
    for li, ids in enumerate(lists):
        for r, i in enumerate(ids):
            sc[i] = sc.get(i, 0.0) + w[li] / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i])


# ---- LongMemEval-S ----
def cache_path(every):
    return Path(f"/tmp/lme/models-cache/lme-every{every}.json")


def cmd_prep(a):
    """Slim per-question file (the top-50 rows only) so a model process never holds the 265 MB dataset (it would drown the model's RSS)."""
    items = lme.scored(json.load(open(DATA, encoding="utf-8")))[::a.every]
    ranks = {json.loads(l)["qid"]: json.loads(l)["turns"][:lme.TOP] for l in open(RANKS)}
    out = []
    for it, gs, gt in items:
        text = {f"{sid}_{k}": t["content"] for sid, s in zip(it["haystack_session_ids"], it["haystack_sessions"]) for k, t in enumerate(s, 1)}
        turns = ranks[it["question_id"]]
        out.append({"q": it["question"], "type": it["question_type"], "gs": sorted(gs), "gt": sorted(gt), "turns": turns,
                    "text": {t: text[t] for t in turns}})
    p = cache_path(a.every)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(p, len(out))


def lme_rows(kind, model, every):
    import numpy as np
    items = json.loads(cache_path(every).read_text())
    rows, lat = [], []
    dcache = {}
    t0 = time.time()
    for n, it in enumerate(items):
        gs, gt = set(it["gs"]), set(it["gt"])
        turns, text, q = it["turns"], it["text"], it["q"]
        if kind == "rerank":
            docs = [text[t] for t in turns[:POOL]]
            if n == 0:
                model.scores(q, docs)   # warm
            t = time.perf_counter()
            sc = model.scores(q, docs)
            lat.append((time.perf_counter() - t) * 1000)
            order = [turns[i] for i in sorted(range(len(docs)), key=lambda i: -sc[i])] + turns[POOL:]
        elif kind == "embed":
            ids = turns[:POOL]   # shortcut: dense rescoring of the 30 bm25 rows, not a dense scan of the haystack (180k messages; EG2 embeds ~6 docs/s); upgrade if the embedder is to add recall beyond bm25
            need = [i for i in ids if text[i][:DOC_CHARS] not in dcache]
            if need:
                for i, v in zip(need, model.docs([text[i] for i in need])):
                    dcache[text[i][:DOC_CHARS]] = v
            D = np.stack([dcache[text[i][:DOC_CHARS]] for i in ids])
            t = time.perf_counter()
            qv = model.query([q], QTASK["lme"])[0]
            lat.append((time.perf_counter() - t) * 1000)
            dense = [ids[i] for i in np.argsort(-(D @ qv))]
            order = rrf([turns, dense])[:lme.TOP]   # same fusion as cs._hybrid (w2, k=60); bm25 list is the 50 memrove-chatlog rows
        else:
            order = turns
        rows.append((it["type"],{"turns": order, "sessions": lme.sessions_of(order)}, gs, gt))
        if n % 20 == 0:
            print(f"lme {n}/{len(items)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    m = lme.evaluate(rows)
    return {"n": m["n"], "S": m["Sany@5"], "T": m["Tany@5"], "lat_ms": statistics.median(lat) if lat else None}


# ---- golden ----
def golden(kind, model, name, rule):
    """Same ranking as memrove search: cs.merged over the livecopy chat log + records, `until` per case. rule: routed (current: dense only for Turkish) | all (dense for every query)."""
    import numpy as np
    import sqlite3
    import memrove_v3_chatsearch as cs
    import memrove_v3_embed as emb
    os.environ["MEMROVE_HYBRID_RULE"] = rule if rule == "routed" else "w2"
    cache, dvec = {}, {}

    def body(rid):
        db = sqlite3.connect(f"file:{STATE}/memory.sqlite3?mode=ro", uri=True)
        try:
            return emb.embed_text(json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0]), full=True)
        finally:
            db.close()

    def dense(state, query, live):
        ids = [rid for rid, r, _ in live if str(r.get("source") or "").startswith(cs.REC_SOURCES[0])]
        if name == "eg2":   # production vectors in the copy, production query path
            qv = model.query([query], "")[0]
            vecs = emb.VecStore(state, readonly=True).load(emb.DaemonEmbedder.name)
            sims = [(float(np.dot(qv, np.asarray(vecs[i]))), i) for i in ids if i in vecs]
        else:
            need = [i for i in ids if i not in dvec]
            if need:
                for i, v in zip(need, model.docs([body(i) for i in need])):
                    dvec[i] = v
            qv = model.query([query], QTASK["golden"])[0]
            sims = [(float(np.dot(qv, dvec[i])), i) for i in ids]
        return ["rec:" + i for _, i in sorted(sims, reverse=True)[:cs.DENSE_POOL]]

    cs._dense = dense
    out = {"tr": [], "en": []}
    lat = []
    cpath = Path("/tmp/lme/models-cache/golden-cands.json")
    cands = json.loads(cpath.read_text()) if kind == "rerank" else {}
    for line in open(GOLDEN):
        c = json.loads(line)
        if not c["expected"]:
            continue
        for lang in ("tr", "en"):
            if lang not in c["queries"]:
                continue
            q = c["queries"][lang]
            key = f"{c['id']}/{lang}"
            if kind == "rerank":   # candidates = the current production top 30 (eg2 run, routed), saved so this process never loads two models
                rows = [tuple(r) for r in cands[key]]
            else:
                rows = cs.merged(STATE, q, top=POOL, until=c["until"], hybrid=(None if name != "bm25" else False))
                if name == "eg2" and rule == "routed":
                    cands[key] = [list(r) for r in rows]
            if kind == "rerank":
                docs = [r[3] for r in rows]
                t = time.perf_counter()
                sc = model.scores(q, docs)
                lat.append((time.perf_counter() - t) * 1000)
                rows = [rows[i] for i in sorted(range(len(rows)), key=lambda i: -sc[i])]
            ids = [str(r[0]).removeprefix("rec:") for r in rows]
            rk = next((i for i, x in enumerate(ids, 1) if x in c["expected"]), None)
            out[lang].append(rk)
    if cands and kind != "rerank":
        cpath.write_text(json.dumps(cands))
    f = lambda v, k: sum(1 for r in v if r and r <= k)
    return {"tr3": f(out["tr"], 3), "tr10": f(out["tr"], 10), "ntr": len(out["tr"]), "en3": f(out["en"], 3), "en10": f(out["en"], 10), "nen": len(out["en"]), "ranks": out,
            "lat_ms": statistics.median(lat) if lat else None}


def cmd_run(a):
    lme.setup("/tmp/lme/models-work/main")
    os.environ.pop("HF_TOKEN", None)
    os.environ["HF_HUB_USER_AGENT_ORIGIN"] = "memrove-bench"
    kind, model, size = load(a.model)
    res = {"model": a.model, "kind": kind, "disk_mib": size / 2**20}
    res["lme"] = None if a.skip_lme else lme_rows(kind, model, a.every or 1)
    for rule in () if a.skip_golden else (("routed", "all") if kind in ("embed", "none") else ("routed",)):
        res[f"golden_{rule}"] = golden(kind, model, a.model, rule)
    # vmmap "Physical footprint (peak)" counts Metal/MPS memory that ru_maxrss misses; eg2 runs in the private daemon
    out = subprocess.check_output(["vmmap", "--summary", str(a.daemon_pid or os.getpid())], text=True)
    val = next(l for l in out.splitlines() if l.startswith("Physical footprint (peak)")).split()[-1]
    res["rss_mib"] = float(val[:-1]) * {"K": 1 / 1024, "M": 1, "G": 1024}[val[-1]]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{a.model}{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:3000])


def cmd_table(a):
    print("| model | LME n | S any@5 | T any@5 | golden rule | tr @3 | en @3 | tr @10 | en @10 | p50 ms/query | peak RSS MiB | disk MiB |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(RES.glob("*.json")):
        r = json.loads(f.read_text())
        L = r["lme"] or {}
        for k in sorted(k for k in r if k.startswith("golden_")):
            g = r[k]
            lat = L.get("lat_ms") or g["lat_ms"] or 0
            print(f"| {f.stem} | {L.get('n', '-')} | {L.get('S', 0):.1f} | {L.get('T', 0):.1f} | {k[7:]} | {g['tr3']}/{g['ntr']} | {g['en3']}/{g['nen']} | {g['tr10']} | {g['en10']} | {lat:.0f} | {r['rss_mib']:.0f} | {r['disk_mib']:.0f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=sorted(MODELS))
    r.add_argument("--skip-golden", action="store_true", help="no livecopy of the live state available")
    r.add_argument("--skip-lme", action="store_true")
    r.add_argument("--daemon-pid", type=int, help="eg2: pid of the private modeld, its RSS is the model's")
    r.add_argument("--tag", default="")
    r.add_argument("--every", type=int, help="LME: every Nth scored question (default 1)")
    sub.add_parser("table")
    p = sub.add_parser("prep")
    p.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    {"run": cmd_run, "table": cmd_table, "prep": cmd_prep}[a.cmd](a)
