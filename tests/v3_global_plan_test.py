import os, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GlobalPlanTest(unittest.TestCase):
    def test_plan_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"; (home / ".claude").mkdir(parents=True)
            settings = home / ".claude/settings.json"
            raw = b'{"theme": "dark"}\n'
            settings.write_bytes(raw)
            vault, state = Path(tmp) / "Vault", Path(tmp) / "state"
            r = subprocess.run([sys.executable, str(ROOT / "scripts/install_v3.py"), "--vault", str(vault),
                                "--state", str(state), "--global", "--plan"],
                               env=dict(os.environ, HOME=str(home), USERPROFILE=str(home)),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('"plan"', r.stdout)
            self.assertIn('UserPromptSubmit', r.stdout)
            self.assertEqual(settings.read_bytes(), raw)
            self.assertEqual(os.listdir(home / ".claude"), ["settings.json"])
            self.assertFalse(vault.exists() or state.exists())


if __name__ == "__main__":
    unittest.main()
