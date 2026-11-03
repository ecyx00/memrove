#!/usr/bin/env python3
"""paid_ok egress: agent-written records may go to a paid remote provider (Haiku) but never to a free-tier one. Stub RUNNER only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, evaluator  # noqa: E402
import v3_reassess_test  # noqa: E402
from memrove_v3_projections import update_record  # noqa: E402
import memrove_v3_trust as trust  # noqa: E402

AGENT = {"origin_type": "agent_inference", "source_system": "memrove-cli", "kind": "lesson"}


class PaidEgressTest(Base):
    def setUp(self):
        super().setUp()
        evaluator.PROVIDERS["freetier"] = (True, lambda state, rubric, items, vault=None, root=None: {i["id"]: {"id": i["id"], "reusable": "yes"} for i in items})
        self.addCleanup(evaluator.PROVIDERS.pop, "freetier", None)

    def item(self, egress):
        return {"id": "r1", "text": "lesson text", "prev": "reuse", "egress": egress, "revision": 1}

    def test_free_remote_refuses_paid_ok_and_haiku_assesses_it(self):
        (a,) = evaluator.evaluate(self.state, "lesson_reusable", [self.item("paid_ok")], provider="freetier")
        self.assertEqual(a["state"], "not_assessed")
        self.assertEqual(evaluator.metrics(self.state)["freetier"]["egress_refused"], 1)
        self.reply["lesson_reusable"] = {"items": [{"id": "r1", "reusable": "yes"}]}
        (b,) = evaluator.evaluate(self.state, "lesson_reusable", [self.item("paid_ok")], provider="haiku")
        self.assertEqual((b["state"], b["answers"]["reusable"]), ("assessed", "yes"))

    def test_visible_by_tier(self):
        self.assertTrue(evaluator.visible("paid_ok", "haiku"))
        self.assertFalse(evaluator.visible("paid_ok", "freetier"))
        self.assertFalse(evaluator.visible("paid_ok", "jev"))          # a remote provider that is not marked paid
        self.assertTrue(evaluator.visible("external_ok", "freetier"))
        self.assertFalse(evaluator.visible("device_only", "haiku"))
        self.assertTrue(evaluator.visible("device_only", "none"))      # local

    def test_haiku_still_refuses_device_only(self):
        (a,) = evaluator.evaluate(self.state, "lesson_reusable", [self.item("device_only")], provider="haiku")
        self.assertEqual(a["state"], "not_assessed")
        self.assertEqual(self.calls, [])

    def test_egress_of_agent_written(self):
        self.assertEqual(trust.combine_egress(["external_ok", "paid_ok"]), "paid_ok")
        self.assertEqual(trust.combine_egress(["paid_ok", "self_hosted"]), "self_hosted")
        self.assertEqual(trust.egress_of(AGENT), "paid_ok")                                    # memrove remember
        self.assertEqual(trust.egress_of(dict(AGENT, source_system="beyin-cli")), "paid_ok")   # its old name
        self.assertEqual(trust.egress_of({"origin": "agent", "origin_type": "agent_inference", "kind": "card"}), "paid_ok")  # memrove card
        self.assertEqual(trust.egress_of({"origin_type": "agent_inference", "origin": "papercut", "kind": "papercut"}), "device_only")
        self.assertEqual(trust.egress_of({"origin_type": "agent_inference", "kind": "lesson"}), "device_only")   # unknown writer
        self.assertEqual(trust.egress_of(dict(AGENT, origin_type="user_explicit")), "device_only")
        self.assertEqual(trust.egress_of({"origin_type": "agent_inference", "verified_by": "llm"}), "external_ok")
        self.assertEqual(trust.egress_of(dict(AGENT, egress="device_only")), "device_only")    # an explicit label wins
        self.engine.note_create("notes/a.md", "private source", {"id": "src", "egress": "device_only"})
        with self.engine.store._connect() as db:
            self.assertEqual(trust.egress_of(dict(AGENT, derived_from=["src"]), db), "device_only")

    def test_remember_stamps_paid_ok_and_user_stays_external(self):
        from memrove_v3_context import remember
        a = remember(self.engine, "lesson", "Run the narrow test first, the full suite once.", "faster loop", "repo")
        u = remember(self.engine, "preference", "Reply to me in Turkish.", None, "repo", user=True)
        rs = self.records()
        self.assertEqual(rs[a["id"]]["egress"], "paid_ok")
        self.assertEqual(trust.egress_of(rs[u["id"]]), "external_ok")

    def test_card_written_by_the_session_stamps_paid_ok(self):
        import memrove_v3_card as card
        card.upsert(self.engine, "c-test", {"project": "repo", "session": "s1", "origin": "agent", "origin_type": "agent_inference", "did": ["x"]})
        self.assertEqual(self.records()["c-test"]["egress"], "paid_ok")


class PairEgressTest(v3_reassess_test.ReassessTest):
    def test_reassess_pair_carries_the_stricter_class(self):
        self.seed()
        update_record(self.engine, "old", {"egress": "paid_ok"})
        seen = []
        old = evaluator.evaluate
        evaluator.evaluate = lambda state, rubric_id, items, **kw: seen.extend(items) or [
            {"item_id": i["id"], "answers": {}, "state": "not_assessed", "revision": None} for i in items]
        try:
            self.run_job()
        finally:
            evaluator.evaluate = old
        self.assertEqual([i["egress"] for i in seen], ["paid_ok"])


if __name__ == "__main__":
    unittest.main()
