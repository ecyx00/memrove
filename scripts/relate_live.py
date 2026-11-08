#!/usr/bin/env python3
"""Live Jev check on synthetic pairs (tests/fixtures/relate_pairs.py). Needs JEV_API_KEY in the environment, else exits 0 with a skip message.
Sends only the synthetic texts, and only with --go (each call may use Jev credits). Usage: relate_live.py --go [max_calls=40]"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.path.insert(0, str(ROOT / "tests/fixtures"))
sys.dont_write_bytecode = True
import memrove_v3_relate as rel  # noqa: E402
from relate_pairs import PAIRS  # noqa: E402

if not os.environ.get("JEV_API_KEY"):
    print("SKIP: JEV_API_KEY is not set; live Jev run not done")
    sys.exit(0)
if "--go" not in sys.argv:
    print("SKIP: pass --go to send the synthetic pairs to Jev")
    sys.exit(0)
args = [a for a in sys.argv[1:] if a != "--go"]
cap = int(args[0]) if args else 40
ok = ab = 0
lat, conf, errs = [], Counter(), Counter()
for i, (o, n, exp) in enumerate(PAIRS[:cap]):
    t = time.time()
    d = rel.jev_decide({"text": o}, {"text": n})
    lat.append(time.time() - t)
    got = d["relation"] or "none"
    if d["relation"] is None:
        ab += 1
        if d["why"].startswith("jev error"):
            errs[d["why"][:60]] += 1
            if "402" in d["why"]:
                print("STOP: HTTP 402 no credits"); break
    conf[(exp, got)] += 1
    ok += got == exp
    print(i, exp, "->", got, d["scores"])
n = sum(conf.values())
lat.sort()
print(f"pairs {n} accuracy {ok/n:.2f} abstain_rate {ab/n:.2f} (abstain counts as 'none') errors {dict(errs)}")
print(f"latency s: median {lat[len(lat)//2]:.2f} max {lat[-1]:.2f}")
print("confusion (expected, got):", dict(conf))
