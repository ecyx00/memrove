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

    def test_global_alone_installs_the_vault_and_workspace_root(self):
        work = Path(self.tmp.name) / "work"
        r = run(self.home, "--vault", str(self.vault), "--state", str(self.state), "--global", "--workspace-root", str(work))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.vault / ".claude/scripts/memrove_v3_hook.py").is_file())
        self.assertEqual(json.loads((self.state / "project-config.json").read_text())["workspace_roots"], [str(work.resolve())])

    def test_doc_search_scans_no_repo_by_default(self):
        (self.home / "Desktop/proj/docs/state").mkdir(parents=True)
        code = "import sys; sys.path.insert(0, %r); import memrove_v3_docsearch as d; print(d.repos())" % str(ROOT / "template/.claude/scripts")
        env = {k: v for k, v in os.environ.items() if k != "MEMROVE_DOC_REPOS"}
        r = subprocess.run([sys.executable, "-c", code], env=dict(env, HOME=str(self.home)), capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "[]", r.stderr)

    def test_python_option_is_baked_and_conda_python_warns(self):
        fake = Path(self.tmp.name) / "anaconda3/bin/python3"
        fake.parent.mkdir(parents=True)
        fake.symlink_to(sys.executable)
        r = run(self.home, "--vault", str(self.vault), "--state", str(self.state), "--python", str(fake))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("conda or venv", r.stderr)
        self.assertIn(str(fake), (self.vault / ".claude/settings.local.json").read_text())

    def test_readme_names_only_real_commands(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import re, memrove_v3
        for name in re.findall(r"`memrove(?:\.py)? (\w[\w-]*)", (ROOT / "README.md").read_text()):
            self.assertIn(name, memrove_v3.COMMANDS, name)


if __name__ == "__main__":
    unittest.main()
