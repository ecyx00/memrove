"""Injected texts: no view-authority claims, one rule, and every `memrove <cmd>` they name resolves on the global wrapper."""
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.path.insert(0, str(ROOT / "scripts"))
import memrove_v3_chattree as ct  # noqa: E402
import memrove_v3_pastref as pr  # noqa: E402
import install_v3 as inst  # noqa: E402

CHAT_TS = (ROOT / "mods/memrove-chat/hooks/chat.ts").read_text()
BANNED = ("is your memory", "say in your reply", "is the truth", "Soul.md")


def wrapper_commands():
    spec = importlib.util.spec_from_file_location("memrove_cli", ROOT / "scripts/memrove_v3.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return set(next(a for a in cli.parser()._actions if a.dest == "command").choices)


class InjectedTextsTest(unittest.TestCase):
    def test_no_view_authority_claims(self):
        texts = [ct.INSTRUCTION, pr.LINE] + re.findall(r"(?:ONE_LINE|VIEW_NOTE) = '(.*)'\n", CHAT_TS)
        self.assertEqual(len(texts), 4)
        for t in texts:
            for b in BANNED:
                self.assertNotIn(b, t)

    def test_rule_is_search_first(self):
        for t in (ct.INSTRUCTION, re.search(r"ONE_LINE = '(.*)'\n", CHAT_TS).group(1)):
            self.assertIn("search first", t)
            self.assertIn("not evidence", t)

    def test_agents_block_has_no_stale_promises(self):
        src = (ROOT / "scripts/install_v3.py").read_text()
        for b in ("Soul.md", "RULES (", "THREADS (", "kendin oku"):
            self.assertNotIn(b, src)

    def test_named_commands_resolve_on_wrapper(self):
        have = wrapper_commands()
        sources = [ct.INSTRUCTION, (ROOT / "scripts/install_v3.py").read_text(),
                   (ROOT / "template/.claude/scripts/memrove_v3_health.py").read_text(), pr.LINE]
        named = {m for t in sources for m in re.findall(r"`memrove ([a-z-]+)[ `]", t)}
        self.assertTrue({"search", "show", "doctor", "zoom"} <= named)
        self.assertEqual(named - have, set())


if __name__ == "__main__":
    unittest.main()
