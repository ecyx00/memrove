#!/usr/bin/env python3
"""Subagent-recall eval: `eval_subrecall.py <cases.json> [--state STATE] [--project P]`.
Each case has `brief` and `hit.message_ids`; a hit is any of those ids among the injected items. Prints per case hit/miss, items, latency, then "HITS n/N" and p95 seconds."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "template/.claude/scripts"))
import memrove_v3_subrecall as sr


def default_state():
    base = Path.home() / "Library/Application Support/memrove-v3" if sys.platform == "darwin" else Path.home() / ".local/share/memrove-v3"
    return os.environ.get("MEMROVE_STATE") or str(sorted(base.glob("*/memory.sqlite3"), key=lambda p: p.stat().st_mtime)[-1].parent)


def main():
    a = sys.argv[1:]
    arg = lambda k, d=None: a[a.index(k) + 1] if k in a[:-1] else d
    data = json.loads(Path(a[0]).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    state, project = arg("--state") or default_state(), arg("--project")
    hits, times = 0, []
    for i, c in enumerate(cases, 1):
        want = {str(x) for x in c["hit"]["message_ids"]}
        t = time.monotonic()
        items = sr.recall(state, c["brief"], project)
        times.append(time.monotonic() - t)
        got = {str(x["id"]).removeprefix("msg:") for x in items}
        ok = bool(want & got)
        hits += ok
        print(f"case {i}: {'HIT ' if ok else 'MISS'} want={sorted(want)} items={len(items)} {times[-1]:.2f}s")
        for x in items:
            print(f"    {x['id']} {x['date']} {x['text'][:90]}")
    p95 = sorted(times)[min(len(times) - 1, int(0.95 * len(times)))]
    print(f"HITS {hits}/{len(cases)}  p95 {p95:.2f}s  max {max(times):.2f}s")


if __name__ == "__main__":
    main()
