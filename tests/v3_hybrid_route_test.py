#!/usr/bin/env python3
"""Hybrid search routing: the Turkish detector, and flag off / English query = the plain bm25 path."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_chatsearch as C


class RouteTest(unittest.TestCase):
    def test_detector(self):
        for q in ("bu işi iptal et", "unit test istemiyorum ve", "NASIL yapılır", "İstanbul", "why did we drop it"):
            self.assertEqual(C.is_turkish(q), q != "why did we drop it", q)
        self.assertFalse(C.is_turkish("I told you I do not want unit tests, cancel it"))

    def test_routing(self):
        calls = []
        with patch.object(C, '_hybrid', lambda *a: calls.append(a) or []), patch.object(C, '_scored', lambda *a: []), patch.object(C, 'records', lambda *a: []):
            for flag, q, hybrid_called in (("0", "bu işi iptal et", False), ("1", "bu işi iptal et", True), ("1", "cancel the unit tests", False), (None, "cancel the unit tests", False), (None, "bu işi iptal et", True)):
                calls.clear()
                env = {k: v for k, v in os.environ.items() if k != 'MEMROVE_HYBRID_SEARCH'}
                if flag is not None:
                    env['MEMROVE_HYBRID_SEARCH'] = flag
                with patch.dict(os.environ, env, clear=True):
                    C.merged('/nowhere', q)
                self.assertEqual(bool(calls), hybrid_called, (flag, q))

    def test_hybrid_caps_records(self):
        msgs = [("m%d" % i, "2026-01-01", "", "x") for i in range(10)]
        recs = [("rec:r%d" % i, "2026-01-01", "note", "x") for i in range(30)]
        with patch.object(C, 'merged', lambda *a, **k: msgs), patch.object(C, '_live', lambda *a: [(r[0][4:], {"kind": "note", "text": "x"}, "2026-01-01") for r in recs]), \
                patch.object(C, '_dense', lambda *a: [r[0] for r in recs]):
            out = C._hybrid('/nowhere', 'bu işi iptal et', 10, None, None, None, False)
        self.assertEqual(sum(r[0].startswith("rec:") for r in out), C.HYBRID_REC_SLOTS)
        self.assertEqual(len(out), 10)
        self.assertTrue(any(not r[0].startswith("rec:") for r in out))


if __name__ == '__main__':
    unittest.main()
