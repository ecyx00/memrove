#!/usr/bin/env python3
"""Real-use metric over synthetic Claude Code transcripts: pastref line fires, followed by a memrove lookup, subagent hints."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "template/.claude/scripts"
sys.path.insert(0, str(SCRIPTS))
import memrove_v3_pastref as pr  # noqa: E402
import memrove_v3_realuse as ru  # noqa: E402

DAY = "2026-10-12T10:00:00.000Z"


def user(text, **kw):
    return {"type": "user", "message": {"role": "user", "content": text}, "timestamp": DAY, "cwd": "/srv/dev/x", **kw}


def line(event, text):
    return {"type": "attachment", "timestamp": DAY, "attachment": {"type": "hook_additional_context", "hookEvent": event, "content": [text]}}


def call(name, **inp):
    return {"type": "assistant", "timestamp": DAY, "message": {"role": "assistant", "content": [{"type": "tool_use", "name": name, "input": inp}]}}


def write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


class RealUseTest(unittest.TestCase):
    def test_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / "p" / "s1.jsonl", [
                user("daha önce ne karar vermiştik"), line("UserPromptSubmit", pr.LINE), call("mcp__memrove-chat__search", query="x"),
                user("geçen hafta yaptığımız ölçüm"), line("UserPromptSubmit", pr.LINE),
                user("testleri çalıştır"), call("Bash", command="memrove search foo"),
                user("bir şey yap"),
                {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta"}, "timestamp": DAY},
            ])
            write(root / "p" / "probe.jsonl", [user("x", entrypoint="sdk-cli"), line("UserPromptSubmit", pr.LINE)])
            write(root / "p" / "tmp.jsonl", [user("x", cwd="/tmp/zz"), line("UserPromptSubmit", pr.LINE)])
            sub = root / "p" / "s1" / "subagents"
            write(sub / "agent-a.jsonl", [line("SubagentStart", "Before you answer, search first"), call("Bash", command="memrove zoom 5 1")])
            (sub / "agent-a.meta.json").write_text(json.dumps({"agentType": "coder"}))
            write(sub / "agent-b.jsonl", [line("SubagentStart", "Before you answer, search first"), call("Read", file_path="/x")])
            (sub / "agent-b.meta.json").write_text(json.dumps({"agentType": "coder"}))
            write(sub / "agent-c.jsonl", [line("SubagentStart", "other hook hint"), call("Read", file_path="/x")])
            (sub / "agent-c.meta.json").write_text(json.dumps({"agentType": "scout"}))
            rep = ru.stats(root)
            self.assertEqual(rep["days"]["2026-10-12"], {"with_line": 2, "with_line_followed": 1, "without_line": 2, "without_line_followed": 1})
            self.assertEqual(rep["subagents"]["coder"], {"total": 2, "fired": 2, "called_memrove": 1})
            self.assertEqual(rep["subagents"]["scout"], {"total": 1, "fired": 0, "called_memrove": 0})
            self.assertIn("2026-10-12", ru.render(rep))


if __name__ == "__main__":
    unittest.main()
