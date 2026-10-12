#!/usr/bin/env python3
"""Known-item check for `memrove chatlog search`: the query is a tree summary (a paraphrase, as the model sees it), the target is any
message that summary covers. Reads a COPY of a state dir (it builds search.db there). Usage: chatsearch_check.py <state> [--n 150] [--levels 0,1,2]"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402


def nodes(state, levels):
    out = []
    for p in sorted((chatlog.root(state) / "tree").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["l"] in levels and len(r["text"]) > 60:
                out.append(r)
    return out


def ranks(state, sample, top=30, rerank=None):
    res = []
    for r in sample:
        q = r["text"].split(": ", 1)[-1]
        want = set(range(r["i"] << r["l"], (r["i"] + 1) << r["l"]))
        got = [h[0] for h in cs.hits(state, q, top, rerank)]
        res.append(next((k + 1 for k, i in enumerate(got) if i in want), None))
    return res


def report(name, rk, secs):
    n = len(rk)
    at = lambda k: sum(1 for x in rk if x and x <= k) / n
    mrr = sum(1 / x for x in rk if x) / n
    print(f"{name:34s} hit@1 {at(1):.2f}  hit@3 {at(3):.2f}  hit@8 {at(8):.2f}  MRR {mrr:.3f}  {secs / n * 1000:.0f} ms/query")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--levels", default="0,1,2")
    a = ap.parse_args()
    cs.sync(a.state)
    sample = random.Random(7).sample(nodes(a.state, {int(x) for x in a.levels.split(",")}), a.n)
    for name, stem in (("exact words", 99), ("stem 6 prefix", 6), ("stem 5 prefix", 5), ("stem 4 prefix", 4)):
        cs.STEM = stem
        t = time.time()
        report(name, ranks(a.state, sample), time.time() - t)
    cs.STEM = 5
    import memrove_v3_embed as em
    emb = em.StaticEmbedder()
    cache = {}

    def vec(i, text):
        if i not in cache:
            cache[i] = emb.embed([text])[0]
        return cache[i]

    def rrf(query, docs, w=1.0):
        q = emb.embed([query])[0]
        by_cos = sorted(range(len(docs)), key=lambda k: -em.cosine(q, vec(*docs[k])))
        score = {}
        for rank, k in enumerate(range(len(docs))):
            score[docs[k][0]] = 1 / (60 + rank)
        for rank, k in enumerate(by_cos):
            score[docs[k][0]] += w / (60 + rank)
        return sorted(score, key=lambda i: -score[i])
    t = time.time()
    report("stem 5 + static rerank (RRF, warm)", ranks(a.state, sample[:0] or sample, rerank=rrf), time.time() - t)
    t = time.time()
    cache.clear()
    report("stem 5 + static rerank (cold)", ranks(a.state, sample, rerank=rrf), time.time() - t)
