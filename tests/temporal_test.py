#!/usr/bin/env python3
"""Router R2 (temporal arm): Turkish gold set (>=90% exact range, 0 wrong narrow windows on no-time queries) plus English spot checks."""
import json
from datetime import date
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
from memrove_v3_temporal import parse  # noqa: E402

GOLD = [json.loads(line) for line in (ROOT / 'tests/fixtures/temporal_tr.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


class TemporalTest(unittest.TestCase):
    def test_turkish_parseable_exact(self):
        rows = [g for g in GOLD if g['since']]
        self.assertGreaterEqual(len(rows), 40)
        bad = []
        for g in rows:
            r = parse(g['q'], date.fromisoformat(g['ref']))
            if not r or (r[0].isoformat(), r[1].isoformat()) != (g['since'], g['until']):
                bad.append((g['q'], r and r[:2]))
        self.assertLessEqual(len(bad), len(rows) * 0.10, bad)

    def test_turkish_no_time_never_narrow(self):
        rows = [g for g in GOLD if not g['since']]
        self.assertGreaterEqual(len(rows), 20)
        bad = []
        for g in rows:
            r = parse(g['q'], date.fromisoformat(g['ref']))
            if r and (r[1] - r[0]).days < 180:   # a wide window is allowed, a narrow one hides the answer
                bad.append((g['q'], r[:2]))
        self.assertEqual(bad, [])

    def test_english(self):
        ref = date(2023, 4, 19)   # Wednesday
        cases = {
            'what did we do last week': (date(2023, 4, 10), date(2023, 4, 16)),
            'in March 2023': (date(2023, 3, 1), date(2023, 3, 31)),
            'two weeks ago': (date(2023, 4, 2), date(2023, 4, 8)),
            'yesterday': (date(2023, 4, 18),) * 2,
            'last Saturday': (date(2023, 4, 15),) * 2,
            'in March and April': (date(2023, 3, 1), date(2023, 4, 30)),
            'since March': (date(2023, 3, 1), ref),
        }
        for q, want in cases.items():
            r = parse(q, ref)
            self.assertEqual(r and r[:2], want, q)
        for q in ('How many days ago did I meet Emma?', 'recently', 'a few days ago', 'I may go', 'all events before today'):
            self.assertIsNone(parse(q, ref), q)


if __name__ == '__main__':
    unittest.main()
