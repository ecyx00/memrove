#!/usr/bin/env python3
"""remember nudge: near-duplicate open records are listed on stderr, unrelated ones are not, errors never fail remember."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True


class RememberNudge(unittest.TestCase):
    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            vault, state, proj = tmp / "Vault", tmp / "state", tmp / "work" / "myproj"
            vault.mkdir(); (proj / ".git").mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}
            env.update(HOME=str(tmp), CLAUDE_CODE_SESSION_ID="s1", MEMROVE_NO_LLM="1")
            run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(vault), "--state", str(state), "remember", *a, "--cwd", str(proj)],
                                            env=env, capture_output=True, text=True)
            first = run("Use zebrafish queue for ingestion", "--kind", "decision")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotIn("may replace", first.stderr)
            unrelated = run("Paint the bike shed green", "--kind", "decision")
            self.assertNotIn("may replace", unrelated.stderr)
            second = run("Switch ingestion to the zebrafish queue v2", "--kind", "decision")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("may replace: rec:" + first.stdout.strip(), second.stderr)
            self.assertNotIn("bike shed", second.stderr)
            self.assertTrue(second.stdout.strip().startswith("obs-"))
            replaced = run("Switch ingestion to the zebrafish queue v3", "--kind", "decision", "--supersedes", first.stdout.strip())
            self.assertNotIn("may replace", replaced.stderr)  # explicit supersedes: no nudge

    def test_never_raises(self):
        from memrove_v3_nudge import nudge
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "memory.sqlite3").write_bytes(b"not a database")
            self.assertEqual(nudge(tmp, "p", "use zebrafish queue"), [])
            self.assertEqual(nudge(tmp + "/missing", "p", "use zebrafish queue"), [])


if __name__ == "__main__":
    unittest.main()
