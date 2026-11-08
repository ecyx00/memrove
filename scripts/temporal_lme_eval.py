#!/usr/bin/env python3
"""R2 check on LongMemEval-S temporal-reasoning: % parsed, and of those the % whose range holds every evidence-session date.
Usage: python3 -I scripts/temporal_lme_eval.py [data.json] [-v]   (reads only; no network)"""
import json, re, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "template/.claude/scripts"))
sys.dont_write_bytecode = True
from memrove_v3_temporal import parse  # noqa: E402

def d(s):
    y, m, dd = map(int, re.match(r"(\d+)/(\d+)/(\d+)", s).groups())
    return date(y, m, dd)

a = [x for x in sys.argv[1:] if x != "-v"]
data = json.load(open(a[0] if a else "/tmp/lme/data/longmemeval_s_cleaned.json"))
qs = [q for q in data if q["question_type"] == "temporal-reasoning" and not q["question_id"].endswith("_abs")]
parsed = ok = anyk = 0
for q in qs:
    r = parse(q["question"], d(q["question_date"]))
    if not r:
        continue
    parsed += 1
    ev = [d(h) for h, i in zip(q["haystack_dates"], q["haystack_session_ids"]) if i in q["answer_session_ids"]]
    good = all(r[0] <= e <= r[1] for e in ev)
    ok += good
    anyk += any(r[0] <= e <= r[1] for e in ev)
    if "-v" in sys.argv:
        print("OK " if good else "BAD", q["question"][:70], "|", q["question_date"][:10], r[:2], [str(e) for e in ev])
print(f"questions {len(qs)}  parsed {parsed} ({100*parsed/len(qs):.1f}%)  holds-all-evidence {ok}/{parsed} ({100*ok/max(parsed,1):.1f}%)  holds-any-evidence {anyk}/{parsed} ({100*anyk/max(parsed,1):.1f}%)")
