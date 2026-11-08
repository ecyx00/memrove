#!/usr/bin/env python3
"""Relation edges: overlap gate, abstain, contradicts edge with prov + valid_from, supersedes only proposes, remote guard, Jev errors."""
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_relate as rel  # noqa: E402

A = {"id": "r-old", "text": "Use sqlite for the memory store in memrove_core.py with 5 retries", "valid_from": "2026-10-01"}
B = {"id": "r-new", "text": "Use postgres for the memory store in memrove_core.py with 3 retries", "valid_from": "2026-11-01"}
FAR = {"id": "r-far", "text": "Reply to the owner in Turkish always", "valid_from": "2026-11-02"}


def db():
    d = sqlite3.connect(":memory:")
    return d


def edges(d):
    try:
        return d.execute("SELECT src,dst,type,prov,valid_from FROM graph_edges").fetchall()
    except sqlite3.OperationalError:  # table never created = nothing written
        return []


class Relate(unittest.TestCase):
    def test_gate_failing_pair_never_sent(self):
        calls = []
        r = rel.relate(db(), [(A, FAR)], decider=lambda o, n: calls.append(1) or {"relation": "contradicts"})
        self.assertEqual((calls, r[0]["status"]), ([], "gated"))

    def test_abstain_writes_no_edge(self):
        d = db()
        r = rel.relate(d, [(A, B)], decider=lambda o, n: {"relation": None, "why": "unsure"})
        self.assertEqual(r[0]["status"], "abstain")
        self.assertEqual(edges(d), [])

    def test_contradicts_writes_edge_with_prov_and_valid_from(self):
        d = db()
        r = rel.relate(d, [(B, A)], decider=lambda o, n: {"relation": "contradicts", "why": "db differs"})  # order of the pair is irrelevant
        self.assertEqual(r[0]["status"], "edge")
        (src, dst, typ, prov, vf), = edges(d)
        self.assertEqual((src, dst, typ, vf), ("r-new", "r-old", "contradicts", "2026-11-01"))
        self.assertIn('"by": "jev"', prov)

    def test_supersedes_only_proposes(self):
        d = db()
        r = rel.relate(d, [(A, B)], decider=lambda o, n: {"relation": "supersedes", "why": "newer"})
        self.assertEqual(r[0]["status"], "proposed")
        self.assertEqual(edges(d), [])

    def test_remote_decider_blocked_without_remote_ok(self):
        r = rel.relate(db(), [(A, B)], decider="jev")
        self.assertEqual(r[0]["status"], "abstain")

    def test_jev_no_key_and_error_abstain(self):
        import os
        os.environ.pop("JEV_API_KEY", None)
        self.assertIsNone(rel.jev_decide(A, B)["relation"])
        os.environ["JEV_API_KEY"] = "x"
        try:
            def boom(body, key):
                raise OSError("402")
            self.assertIsNone(rel.jev_decide(A, B, post=boom)["relation"])
            ok = lambda body, key: {"answers": {"contradicts": {"noul": 0.9}, "supports": {"noul": 0.1}}}
            self.assertEqual(rel.jev_decide(A, B, post=ok)["relation"], "contradicts")
            two = lambda body, key: {"answers": {"contradicts": {"noul": 0.8}, "supports": {"noul": 0.9}}}
            self.assertEqual(rel.jev_decide(A, B, post=two)["relation"], "supports")  # nested relations: top score wins
            low = lambda body, key: {"answers": {"contradicts": {"noul": 0.5}, "supports": {"noul": 0.4}}}
            self.assertIsNone(rel.jev_decide(A, B, post=low)["relation"])
            for bad in (float("nan"), True, 5, float("inf"), -0.1):  # malformed scores never pick a relation
                junk = lambda body, key, v=bad: {"answers": {"supports": {"noul": v}}}
                self.assertIsNone(rel.jev_decide(A, B, post=junk)["relation"], bad)
            def leak(body, key):
                raise ValueError("Invalid header value b'Bearer %s'" % key)
            os.environ["JEV_API_KEY"] = "SECRETKEY123\n"
            self.assertNotIn("SECRETKEY123", rel.jev_decide(A, B, post=leak)["why"])
        finally:
            os.environ.pop("JEV_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
