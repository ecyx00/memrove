#!/usr/bin/env python3
"""Arm table for subagent recall: `eval_arms.py <cases.json> [--top 6] [--state S] [--cards DIR_WITH_ideacards_module]`.
Arms: bm25 (raw owner text), dense (message vectors), cards (idea cards, when importable), rrf (k=60 of the available arms). Hit = a hit id in the arm's top N message ids."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "template/.claude/scripts"))
sys.path.insert(0, str(HERE))
import memrove_v3_subrecall as sr
from eval_subrecall import default_state


def rrf(*lists, k=60):
    s = {}
    for lst in lists:
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0) + 1 / (k + r + 1)
    return sorted(s, key=s.get, reverse=True)


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a[:-1] else d
    data = json.loads(Path(a[0]).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    state, top = arg("--state") or default_state(), int(arg("--top", 6))
    cards = None
    if arg("--cards"):
        sys.path.insert(0, arg("--cards"))
        import memrove_v3_ideacards as cards
    arms = {"bm25": lambda b: [r[0] for _, r in sr._messages(state, sr.query_of(b))],
            "dense": lambda b: [r[0] for _, r in sr._dense(state, b, 400) if not str(r[0]).startswith("rec:")]}
    if cards:
        arms["cards"] = lambda b: [i for i, _ in cards.search_cards(state, b, 50)]
    res = {n: [] for n in [*arms, "rrf"]}
    lat = {n: [] for n in res}
    for c in cases:
        want = {int(x) for x in c["hit"]["message_ids"]}
        got, t0 = {}, time.monotonic()
        for n, f in arms.items():
            t = time.monotonic()
            try:
                got[n] = [int(x) for x in f(c["brief"])]
            except Exception as e:
                got[n] = []
                print("arm", n, "failed:", repr(e), file=sys.stderr)
            lat[n].append(time.monotonic() - t)
        got["rrf"] = rrf(*got.values())
        lat["rrf"].append(time.monotonic() - t0)
        for n in res:
            res[n].append(bool(want & set(got[n][:top])))
    for n in res:
        p95 = sorted(lat[n])[min(len(lat[n]) - 1, int(0.95 * len(lat[n])))]
        print(f"{n:6s} HITS {sum(res[n])}/{len(cases)}  p95 {p95:.2f}s  per-case {''.join('H' if x else '.' for x in res[n])}")


if __name__ == "__main__":
    main()
