import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = str(ROOT / "scripts/install_v3.py")


def run(home, *args, **env):
    return subprocess.run([sys.executable, INSTALL, *args], env=dict(os.environ, HOME=str(home), USERPROFILE=str(home), **env),
                          capture_output=True, text=True)


class InstallOptionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.vault, self.state = Path(self.tmp.name) / "vault", Path(self.tmp.name) / "state"
        self.vault.mkdir(parents=True)
        self.home.mkdir()

    def test_workspace_root_option_is_written_once(self):
        work = Path(self.tmp.name) / "work"
        for _ in range(2):
            r = run(self.home, "--vault", str(self.vault), "--state", str(self.state), "--workspace-root", str(work))
            self.assertEqual(r.returncode, 0, r.stderr)
        cfg = json.loads((self.state / "project-config.json").read_text())
        self.assertEqual(cfg["workspace_roots"], [str(work.resolve())])


if __name__ == "__main__":
    unittest.main()
