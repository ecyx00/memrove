#!/usr/bin/env python3
"""Two-stage selector: `eval_selector.py <cases.json>... --n 50 [--cards DIR] [--state S]`. Two passes over the cases (cold, then warm); prints hits, hits on non-weak cases
(WEAK ids excluded), Wilson 95% CI, p50/p95 wall time per pass and mean tokens per call."""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "template/.claude/scripts"))
sys.path.insert(0, str(HERE))
import memrove_v3_selector as sel
from eval_subrecall import default_state

WEAK = {"d28", "d29", "d30"}   # flagged weak in cases-c-notes.md


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def carded(state, case):
    import sqlite3
    db = sqlite3.connect(f"file:{Path(state) / 'ideacards.sqlite3'}?mode=ro", uri=True)
    ids = case["hit"]["message_ids"]
    return db.execute(f"SELECT count(*) FROM cards WHERE card!='none' AND id IN ({','.join('?' * len(ids))})", ids).fetchone()[0] > 0


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a[:-1] else d
    if arg("--cards"):
        sys.path.insert(0, arg("--cards"))
    state, n = arg("--state") or default_state(), int(arg("--n", 100))
    cases = []
    for f in (x for x in a if x.endswith(".json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        cases += d["cases"] if isinstance(d, dict) else d
    for label in ("cold", "warm"):
        hits, strong, times, use, by = 0, 0, [], [], {True: [0, 0], False: [0, 0]}
        for c in cases:
            want = {f"m{x}" for x in c["hit"]["message_ids"]}
            t = time.monotonic()
            r = sel.select(state, c["brief"], None, n)
            times.append(time.monotonic() - t)
            ok = bool(want & set(r.get("picks", [])))
            hits += ok
            strong += ok and c["id"] not in WEAK
            by[carded(state, c)][0] += ok
            by[carded(state, c)][1] += 1
            use.append(r.get("usage", {}))
            print(f"{label} {c['id']}: {'HIT ' if ok else 'MISS'} {times[-1]:.1f}s picks={r.get('picks', r.get('error'))} want={sorted(want)}", flush=True)
        ts = sorted(times)
        lo, hi = wilson(hits, len(cases))
        m = lambda k: sum(u.get(k) or 0 for u in use) / len(use)
        print(f"N={n} {label}: HITS {hits}/{len(cases)} (Wilson95 {lo:.2f}-{hi:.2f}) non-weak {strong}/{len(cases) - sum(c['id'] in WEAK for c in cases)} "
              f"carded {by[True][0]}/{by[True][1]} uncarded {by[False][0]}/{by[False][1]} p50 {ts[len(ts) // 2]:.1f}s p95 {ts[min(len(ts) - 1, int(0.95 * len(ts)))]:.1f}s tokens/call input {m('input_tokens'):.0f} cache_write {m('cache_creation_input_tokens'):.0f} "
              f"cache_read {m('cache_read_input_tokens'):.0f} output {m('output_tokens'):.0f}", flush=True)


if __name__ == "__main__":
    main()
