#!/usr/bin/env python3
"""evidence: a record is verified only when code finds its quote verbatim in a chat message; an agent record never becomes a standing rule."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
os.environ.setdefault("MEMROVE_NO_LLM", "1")
import memrove_core as core  # noqa: E402
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_evidence as evidence  # noqa: E402
import memrove_v3_rules as rules  # noqa: E402

SAID = "Always use   SQLite for the store, never postgres."


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.state = t / "s"
        self.c = core.Core(t / "v", self.state)
        self.k = "c:msg-1"
        chatlog.append(self.state, [{"kind": "user", "text": f"ok. {SAID} thanks", "date": "2026-10-07T10:00:00+00:00", "src": "claude", "k": self.k}])

    def test_verbatim_quote_is_found_after_whitespace_and_case_normalization(self):
        r = {"kind": "decision", "text": 'The owner said "always use sqlite for the store, never POSTGRES." today'}
        self.assertEqual(evidence.verify(self.state, r), [{"kind": "msg", "ref": self.k}])

    def test_near_miss_paraphrase_is_not_verified(self):
        r = {"kind": "decision", "text": 'The owner said "always use sqlite for the store and avoid postgres".'}
        self.assertEqual(evidence.verify(self.state, r), [])

    def test_owner_quote_gate_matches_user_messages_only(self):
        chatlog.append(self.state, [{"kind": "agent", "text": "the assistant alone wrote this sentence here", "date": "2026-10-07T10:01:00+00:00", "src": "claude", "k": "c:msg-2"}])
        self.assertEqual(evidence.owner_quote(self.state, SAID.upper(), None), SAID.upper())
        self.assertEqual(evidence.owner_quote(self.state, None, f"Owner: '{SAID}'"), SAID)
        self.assertIsNone(evidence.owner_quote(self.state, "the assistant alone wrote this sentence here", None))
        self.assertIsNone(evidence.owner_quote(self.state, None, None))

    def test_agent_record_never_reaches_the_standing_rules_even_when_verified(self):
        def restated(rid, sentence):
            self.c._patch(rid, "test", assessments={"restate": {"state": "assessed", "answers": {"standing_rule": "yes", "scope": "global", "sentence": sentence}}})
        agent = self.c.ingest({"kind": "decision", "text": f'Claude wrote "{SAID}" down', "why": "w"})[0]["id"]
        owner = self.c.ingest({"kind": "decision", "text": SAID, "why": "w", "user": True})[0]["id"]
        restated(agent, "Claude must always use SQLite for the store in every project.")
        restated(owner, "The owner wants SQLite as the store in every project.")
        self.assertTrue(evidence.backfill(self.c)["verified"] >= 2)
        self.assertEqual(self.c.expand(agent)["record"]["evidence_state"], "verified")
        self.assertEqual([x[3] for x in rules.rules(self.state)], [owner])


if __name__ == "__main__":
    unittest.main()
