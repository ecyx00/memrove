"""Delivery phase 1: byte budgets of the SessionStart text, usage note size, silent per-prompt hook, size log."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v3_package_helpers import isolated_env
from v3_rules_test import judged, rec, store

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "template/.claude/scripts"
HOOK = SCRIPTS / "memrove_v3_hook.py"
sys.path.insert(0, str(SCRIPTS))
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chattree as ct  # noqa: E402
import memrove_v3_jit as jit  # noqa: E402
import memrove_v3_sizelog as sizelog  # noqa: E402

spec = importlib.util.spec_from_file_location("hook_p1", HOOK)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def oversized_state():
    """A state with 60 long rules, a huge card for project 'unknown' and a chat view; -> state dir."""
    rules = [rec(f"q-{i}", "preference", "x", "2026-10-%02d" % (i % 28 + 1), assessments=judged(f"The owner wants rule number {i} to hold with {'x' * 80} everywhere."))
             for i in range(60)]
    card = {"id": "c-1", "kind": "card", "status": "active", "project": "unknown", "valid_from": "2026-10-18", "did": ["d" * 200] * 5,
            "why": ["w" * 200] * 3, "open": ["o" * 200] * 3, "next": "n" * 300}
    state = store(rules + [card])
    view = chatlog.root(state)
    view.mkdir(parents=True, exist_ok=True)
    (view / "view.json").write_text(json.dumps({"T": 1234}))
    return state


class DeliveryP1Test(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-p1-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        home = self.root / "home"
        home.mkdir()
        p = patch.object(Path, "home", return_value=home)   # the real ~/.claude/CLAUDE.md is never read
        p.start()
        self.addCleanup(p.stop)
        self.state = oversized_state()
        words = [json.dumps({"type": "user", "message": {"content": f"owner message {i} " + "w" * 500}}) for i in range(40)]
        self.transcript = self.root / "t.jsonl"
        self.transcript.write_text("\n".join(words))

    def test_startup_total_is_at_most_4000(self):
        text, slots = hook.session_start(self.state, {"source": "startup", "cwd": str(self.vault)}, self.vault)
        self.assertGreater(len(slots["rules"]), 1000)   # the input really was oversized
        self.assertLessEqual(len(text.encode()), 4000)
        used = sum(len(v.encode()) for k, v in slots.items() if k != "rules")
        self.assertGreater(len(slots["rules"].encode()), 2500)   # rules take the room the other slots leave
        self.assertGreater(len(slots["rules"].encode()), 4000 - used - 300)
        self.assertIn("Standing rules no longer fit", slots["owner"])   # 60 rules overflow: the owner is told (systemMessage)
        self.assertNotIn("Standing rules no longer fit", text)   # upkeep stays out of the model's context
        self.assertLessEqual(len(slots["card"].encode()), 600)
        self.assertIn("handoff card c-1", slots["card"])
        self.assertIn("memrove card --did", slots["card"])   # the handoff line survives the card's cut

    def test_compact_total_is_at_most_6000_and_keeps_the_newest_words(self):
        text, slots = hook.session_start(self.state, {"source": "compact", "cwd": str(self.vault), "transcript_path": str(self.transcript)}, self.vault)
        self.assertLessEqual(len(text.encode()), 6000)
        self.assertLessEqual(len(slots["words"].encode()), 2000)
        self.assertIn("owner message 39", slots["words"])
        self.assertNotIn("owner message 0 ", slots["words"])
        self.assertNotIn("w" * 301, slots["words"])   # 300 chars per message

    def test_usage_note_is_at_most_500(self):
        self.assertLessEqual(len(ct.instruction(self.state).encode()), 500)
        self.assertLessEqual(len(jit.RULE.encode()), 500)
        self.assertLessEqual(len(jit.SUBAGENT_RULE.encode()), 500)
        note = ct.instruction(self.state)
        for must in ("memrove search", "memrove wake", "memrove zoom", "verify drift-prone facts", "beat an agent's", "newer beats older"):
            self.assertIn(must, note)

    def test_hook_logs_sizes_and_a_cueless_prompt_is_silent(self):
        env = isolated_env(self.root / "home")
        state = self.root / "state"
        base = [sys.executable, str(HOOK), "--vault", str(self.vault), "--state", str(state), "--harness", "claude", "--global"]
        for payload in ({"hook_event_name": "SessionStart", "source": "startup", "session_id": "s1", "cwd": str(self.vault)},
                        {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "cwd": str(self.vault), "prompt": "testleri çalıştır"},
                        {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "cwd": str(self.vault), "prompt": "geçen sefer ne demiştik?"},
                        {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "cwd": str(self.vault), "prompt": "geçen sefer ne demiştik?"}):
            subprocess.run(base, input=json.dumps(payload), capture_output=True, text=True, env=env, cwd=self.vault, timeout=30, check=True)
        rows = [json.loads(ln) for ln in (state / "hook-size.jsonl").read_text().splitlines()]
        prompts = [r["bytes"] for r in rows if r["event"] == "UserPromptSubmit"]
        self.assertEqual([r["event"] for r in rows], ["SessionStart", "UserPromptSubmit", "UserPromptSubmit", "UserPromptSubmit"])
        self.assertEqual((prompts[0], prompts[1] > 0, prompts[2]), (0, True, 0))   # no cue: 0 bytes; cue: one line; same session again: 0 bytes
        self.assertEqual(set(rows[0]), {"ts", "event", "source", "bytes", "slots"})
        self.assertEqual(sizelog.stats(state)["UserPromptSubmit/-"]["n"], 3)


if __name__ == "__main__":
    unittest.main()
