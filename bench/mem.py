#!/usr/bin/env python3
"""Same-conditions memory bench on LoCoMo (evidence retrieval, no LLM): recall@5/@10 per category, false-memory rate, latency, tokens, peak RSS.

  python3 bench/mem.py --data /tmp/lme/locomo/locomo10.json --arm bm25|memrove-chatlog|mem0-raw [--limit N]

One arm per process, so ru_maxrss is that arm's. Corpus = every dialog turn of one conversation (id D<session>:<turn>), query = the question, gold = qa.evidence.
Recall@k = mean fraction of a question's evidence turns in the top k (categories 1-4). Category 5 (adversarial, unanswerable) gives the false-memory rate:
the share of them whose top-1 score is >= the 10th percentile of top-1 scores over answerable questions (same rule for every arm; calibrated on the test set).
Tokens/query = chars/4 of the top-5 texts. Latency = search call only, index already built. mem0-raw: Mem0 add(infer=False), local HF embedder, no LLM call.
shortcut: LoCoMo image captions are not indexed, text only; upgrade if the owner wants multimodal turns.
"""
import argparse
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lme  # noqa: E402  (bm25 scorer and setup(), shared with the LME bench)

TOP = 50


def turns(conv):
    """-> [(dia_id, speaker, text)] in conversation order."""
    out = []
    n = 1
    while f"session_{n}" in conv:
        out += [(t["dia_id"], t["speaker"], t["text"]) for t in conv[f"session_{n}"]]
        n += 1
    return out


def gold(q):
    return {e.strip() for x in q.get("evidence", []) for e in x.replace(";", " ").split() if e.strip()}


class Bm25:
    def __init__(self, ts, work):
        self.ts, self.docs = ts, [t[2].split(" ") for t in ts]

    def search(self, q):
        sc = lme.bm25_scores(self.docs, q.split(" "))
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], -i))[:TOP]
        return [(self.ts[i][0], sc[i], self.ts[i][2]) for i in order]


class Memrove:
    def __init__(self, ts, work):
        import memrove_v3_chatlog as chatlog
        self.cs = __import__("memrove_v3_chatsearch")
        self.state = tempfile.mkdtemp(dir=work, prefix="state-")
        assert str(Path(self.state).resolve()).startswith(str(Path(work).resolve())), self.state
        first = ts[0][1]
        chatlog.append(self.state, [{"kind": "user" if t[1] == first else "agent", "text": t[2], "date": ""} for t in ts], today="2023-01-01")
        self.ts = ts

    def search(self, q):
        rows = self.cs._scored(self.state, q, TOP, None, None, None)   # the `memrove search` message lane for English; bm25 is lower-is-better, so negate
        return [(self.ts[r[0]][0], -sc, r[3]) for sc, r in rows]


class Mem0Raw:
    def __init__(self, ts, work):
        self.m = lme.mem0_open(tempfile.mkdtemp(dir=work, prefix="mem0-"))
        for dia, _, text in ts:
            self.m.add(text, user_id="u", infer=False, metadata={"dia": dia})

    def search(self, q):
        r = self.m.search(q, filters={"user_id": "u"}, limit=TOP)["results"]
        return [(x["metadata"]["dia"], x["score"], x["memory"]) for x in r]


class Rerank(Memrove):
    """memrove-chatlog top POOL, re-ordered by the hs-rerank-1 cross-encoder (a local model, no LLM call); the score is the reranker's logit, so the abstention floor is the reranker's."""
    POOL, DOC_CHARS = 30, 1200   # same depth and truncation as bench/models.py on branch worktree-agent-a03a21ed30fa4c5bc

    def __init__(self, ts, work):
        super().__init__(ts, work)
        if "model" not in Rerank.__dict__:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            Rerank.model = CrossEncoder("hindsight-ai/hs-rerank-1", device=dev, max_length=512, model_kwargs={"torch_dtype": torch.float16 if dev == "mps" else torch.float32})

    def search(self, q):
        hits = super().search(q)
        pool = hits[:self.POOL]
        sc = [float(x) for x in self.model.predict([(q, h[2][:self.DOC_CHARS]) for h in pool], batch_size=16, show_progress_bar=False)] if pool else []
        return [(h[0], s, h[2]) for s, h in sorted(zip(sc, pool), key=lambda p: -p[0])] + hits[self.POOL:]   # tail keeps memrove's bm25 scores but is never top-1


ARMS = {"bm25": Bm25, "memrove-chatlog": Memrove, "mem0-raw": Mem0Raw, "memrove-rerank": Rerank}


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def run(data, arm, limit):
    work = lme.setup("/tmp/lme/work-mem")
    rows, lat, toks = [], [], []   # rows: (category, [ids], top1 score, gold)
    for conv in json.load(open(data)):
        ts = turns(conv["conversation"])
        t0 = time.time()
        a = ARMS[arm](ts, work)
        build = time.time() - t0
        for q in conv["qa"][:limit or None]:
            t0 = time.perf_counter()
            hits = a.search(q["question"])
            lat.append((time.perf_counter() - t0) * 1000)
            toks.append(sum(len(h[2]) for h in hits[:5]) / 4)
            rows.append((q["category"], [h[0] for h in hits], hits[0][1] if hits else float("-inf"), gold(q)))
        print(f"{conv['sample_id']} built {len(ts)} turns in {build:.1f}s", file=sys.stderr)
    out = {"arm": arm, "n_q": len(rows), "p50_ms": statistics.median(lat), "p95_ms": pctl(lat, 0.95), "tokens_per_query": statistics.mean(toks),
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}   # macOS: bytes
    ans = [r for r in rows if r[0] != 5 and r[3]]
    for k in (5, 10):
        rec = lambda rs: 100 * statistics.mean(len(r[3] & set(r[1][:k])) / len(r[3]) for r in rs)
        out[f"recall@{k}"] = rec(ans)
        out[f"recall@{k}_cat"] = {c: rec([r for r in ans if r[0] == c]) for c in (1, 2, 3, 4)}
    thr = pctl([r[2] for r in ans], 0.10)
    adv = [r for r in rows if r[0] == 5]
    out["fm_threshold"] = thr
    out["false_memory_rate"] = 100 * sum(r[2] >= thr for r in adv) / len(adv) if adv else None
    out["n_adv"] = len(adv)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per conversation (debug)")
    a = ap.parse_args()
    print(json.dumps(run(a.data, a.arm, a.limit), indent=1))
