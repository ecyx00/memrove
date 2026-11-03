#!/usr/bin/env python3
"""Restate tiebreak: when the two draws disagree a third draw decides (2 of 3 yes with the same scope and project). Fake evaluate only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, evaluator  # noqa: E402
from memrove_v3_projections import update_record  # noqa: E402
import memrove_v3_restate as restate  # noqa: E402


def ans(rule, scope="global", project="", sentence="s"):
    return {"standing_rule": rule, "scope": scope, "project": project, "sentence": sentence}


class TiebreakTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.draws = {}   # item id -> {salt: answers}
        self.old = evaluator.evaluate

        def fake(state, rubric_id, items, provider=None, vault=None, root=None):
            self.calls.append([(i["id"], i.get("salt", "1")) for i in items])
            out = []
            for i in items:
                x = self.draws[i["id"]][i.get("salt", "1")]
                out.append({"item_id": i["id"], "rubric_id": rubric_id, "rubric_version": 1, "provider": "haiku", "revision": 1, "answers": dict(x),
                            "state": "unclear" if x["standing_rule"] == "unclear" else "assessed", "latency_ms": 0})
            return out
        evaluator.evaluate = fake

    def tearDown(self):
        evaluator.evaluate = self.old

    def run_votes(self, **draws):
        self.draws = draws
        out = restate.two_votes("unused", [{"id": k} for k in draws])
        return {a["item_id"]: a for a in out}

    def test_yes_no_then_yes(self):
        a = self.run_votes(r={"1": ans("yes", sentence="one"), "2": ans("no"), "3": ans("yes", sentence="three")})["r"]
        self.assertEqual((a["state"], a["answers"]["standing_rule"]), ("assessed", "yes"))
        self.assertIn(a["answers"]["sentence"], ("one", "three"))

    def test_no_yes_then_yes(self):
        a = self.run_votes(r={"1": ans("no", sentence="one"), "2": ans("yes", sentence="two"), "3": ans("yes", sentence="three")})["r"]
        self.assertEqual((a["state"], a["answers"]["standing_rule"]), ("assessed", "yes"))
        self.assertIn(a["answers"]["sentence"], ("two", "three"))   # a yes draw of the agreeing pair, never the "no" draw

    def test_yes_no_then_no_is_unclear(self):
        a = self.run_votes(r={"1": ans("yes"), "2": ans("no"), "3": ans("no")})["r"]
        self.assertEqual((a["state"], a["answers"]["standing_rule"]), ("unclear", "unclear"))

    def test_scope_split_then_match(self):
        a = self.run_votes(r={"1": ans("yes", "global"), "2": ans("yes", "project", "memrove"), "3": ans("yes", "project", "Memrove")})["r"]
        self.assertEqual((a["answers"]["standing_rule"], a["answers"]["scope"]), ("yes", "project"))

    def test_three_way_split_is_unclear(self):
        a = self.run_votes(r={"1": ans("yes", "global"), "2": ans("yes", "project", "a"), "3": ans("yes", "project", "b")})["r"]
        self.assertEqual(a["state"], "unclear")

    def test_yes_and_unclear_draw_needs_a_third(self):
        out = self.run_votes(p={"1": ans("yes"), "2": ans("unclear"), "3": ans("yes")}, q={"1": ans("yes"), "2": ans("unclear"), "3": ans("no")})
        self.assertEqual((out["p"]["answers"]["standing_rule"], out["q"]["state"]), ("yes", "unclear"))

    def test_agreeing_pairs_take_no_third_draw(self):
        out = self.run_votes(n={"1": ans("no"), "2": ans("no")}, y={"1": ans("yes"), "2": ans("yes")})
        self.assertEqual((out["n"]["answers"]["standing_rule"], out["y"]["answers"]["standing_rule"]), ("no", "yes"))
        self.assertEqual(len(self.calls), 2)

    def test_third_draws_go_in_one_call(self):
        self.run_votes(a={"1": ans("yes"), "2": ans("no"), "3": ans("yes")}, b={"1": ans("no"), "2": ans("yes"), "3": ans("no")},
                       c={"1": ans("no"), "2": ans("no")})
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(sorted(self.calls[2]), [("a", "3"), ("b", "3")])


class RedoUnclearTest(Base):
    def test_pending_unclear_only(self):
        for rid, st, rs in (("u", "active", "unclear"), ("y", "active", "assessed"), ("d", "dropped_llm", "unclear"), ("n", "active", None)):
            self.engine.note_create(f"notes/captured/{rid}.md", "Commit mesajları kısa olsun.", dict(
                id=rid, kind="preference", project="repo", session="s1", verified_by="user_quote", status=st, key=rid, polarity="pos"))
            if rs:
                update_record(self.engine, rid, {"assessments": {"restate": {"state": rs, "answers": {}}}})
        with self.engine.store._connect() as db:
            self.assertEqual(sorted(r["id"] for r in restate.pending(db, redo="unclear")), ["u"])
            self.assertEqual(sorted(r["id"] for r in restate.pending(db)), ["n"])


    def test_adopt_redo_unclear_overwrites_only_unclear(self):
        import json
        import sqlite3
        for rid, rs in (("u", "unclear"), ("y", "assessed")):
            self.engine.note_create(f"notes/captured/{rid}.md", "Commit mesajları kısa olsun.", dict(
                id=rid, kind="preference", project="repo", session="s1", verified_by="user_quote", status="active", key=rid, polarity="pos"))
            update_record(self.engine, rid, {"assessments": {"restate": {"state": rs, "answers": {"standing_rule": rs}}}})
        copy = self.root / "copy"
        copy.mkdir()
        with sqlite3.connect(self.state / "memory.sqlite3") as src, sqlite3.connect(copy / "memory.sqlite3") as dst:
            src.backup(dst)
            for rid in ("u", "y"):
                r = json.loads(dst.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])
                r["assessments"]["restate"] = {"state": "assessed", "answers": {"standing_rule": "yes"}}
                dst.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(r), rid))
        log = self.root / "adopt.log"
        self.assertEqual(restate.adopt(self.engine, self.state, copy, log)["written"], 0)
        self.assertEqual(restate.adopt(self.engine, self.state, copy, log, redo="unclear")["written"], 1)
        got = {i: r["assessments"]["restate"]["answers"]["standing_rule"] for i, r in self.records().items() if i in ("u", "y")}
        self.assertEqual(got, {"u": "yes", "y": "assessed"})


if __name__ == "__main__":
    unittest.main()
