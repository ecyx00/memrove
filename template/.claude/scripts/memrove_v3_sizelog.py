#!/usr/bin/env python3
"""Size log of what memrove hooks inject: one jsonl row {ts, event, source, bytes, slots} per hook event in <state>/hook-size.jsonl.

`python3 memrove_v3_sizelog.py --state <dir>` prints n, p50, p95 and max bytes per event/source. Stdlib only, never raises on write."""
import json
import sys
import time
from pathlib import Path

NAME = "hook-size.jsonl"


def log(state, event, source, nbytes, slots=None):
    try:
        if not state:
            return
        Path(state).mkdir(parents=True, exist_ok=True)
        row = {"ts": round(time.time(), 3), "event": event, "source": source, "bytes": nbytes,
               "slots": {k: len(v.encode()) if isinstance(v, str) else v for k, v in (slots or {}).items()}}
        with open(Path(state) / NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, -(-len(xs) * q // 100) - 1)]   # nearest rank


def stats(state):
    """-> {"Event/source": {"n", "p50", "p95", "max"}}"""
    groups = {}
    try:
        lines = (Path(state) / NAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for ln in lines:
        try:
            r = json.loads(ln)
            groups.setdefault(f"{r['event']}/{r.get('source') or '-'}", []).append(int(r["bytes"]))
        except (ValueError, KeyError, TypeError):
            pass
    return {k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "max": max(v)} for k, v in sorted(groups.items())}


if __name__ == "__main__":
    state = sys.argv[sys.argv.index("--state") + 1] if "--state" in sys.argv[1:-1] else None
    if not state:
        raise SystemExit("usage: memrove_v3_sizelog.py --state <dir>")
    print(f"{'event/source':28} {'n':>6} {'p50':>7} {'p95':>7} {'max':>7}")
    for k, s in stats(state).items():
        print(f"{k:28} {s['n']:>6} {s['p50']:>7} {s['p95']:>7} {s['max']:>7}")
