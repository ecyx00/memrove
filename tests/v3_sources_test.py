#!/usr/bin/env python3
"""Source pages: src-tag and number grounding against a snapshot, and a fetch that never runs snapshot code."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_sources as sources  # noqa: E402

README = "# demo\nA tiny search tool.\nIt indexes 1,000 files in 0.5 seconds.\nk = 60\n"

BODY = """# Demo (repo)
- N1.2 Index: scans files. [src:demo:README.md:2]
- The index builds 1,000 files in 0.5 seconds, $k = 60$ (the author's measurement) [src:demo:README.md:3]
- Genel bilgi, 99 ms. [general]
- k sabiti 60, NDCG@10, #279 [src:demo:README.md:4]
"""


def make_vault(tmp):
    snap = Path(tmp) / "wiki" / "_sources" / "snap" / "demo"
    snap.mkdir(parents=True)
    (snap / "README.md").write_text(README)
    return Path(tmp)


def run_check(tmp, body):
    return sources.check_src(body, "demo", make_vault(tmp))


class SourcesTest(unittest.TestCase):
    def test_valid_page_passes(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(run_check(t, BODY), [])

    def test_missing_file_or_line_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            errs = run_check(t, BODY.replace("README.md:2", "NOPE.md:2").replace("README.md:4]", "README.md:99]"))
            self.assertTrue(any("NOPE.md" in e for e in errs), errs)
            self.assertTrue(any(":99" in e for e in errs), errs)

    def test_path_escape_and_wrong_slug_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            errs = run_check(t, "- x [src:demo:../../../../etc/hosts:1]\n- y [src:other:README.md:1]\n")
            self.assertEqual(len(errs), 2, errs)

    def test_number_not_in_cited_file_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            errs = run_check(t, BODY.replace("0.5 seconds", "7.5 seconds").replace("$k = 60$", "$k = 61$"))
            self.assertTrue(any("7.5" in e for e in errs), errs)
            self.assertTrue(any("61" in e for e in errs), errs)

    def test_fetch_never_runs_snapshot_code(self):
        with tempfile.TemporaryDirectory() as t:
            repo, snap, marker = Path(t) / "r", Path(t) / "snap", Path(t) / "PWNED"
            repo.mkdir()
            for name in ("setup.py", "conftest.py", "evil.sh", "json.py"):
                (repo / name).write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')\n")
            (repo / "README.md").write_text(README)
            env = dict(os.environ, GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@a", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@a")
            for cmd in (["init", "-q"], ["add", "."], ["commit", "-qm", "x"]):
                subprocess.run(["git", *cmd], cwd=repo, env=env, check=True, capture_output=True)
            seen = []
            real = sources._run
            sources._run = lambda cmd, *a, **k: (seen.append(cmd), real(cmd, *a, **k))[1]
            try:
                meta = sources.fetch_repo(str(repo), snap)
            finally:
                sources._run = real
            self.assertFalse(marker.exists())
            self.assertTrue((snap / "setup.py").is_file())  # kept as data
            self.assertFalse((snap / ".git").exists())
            self.assertRegex(meta["sha"], r"^[0-9a-f]{40}$")
            self.assertTrue(all(c[0] == "git" for c in seen), seen)
            self.assertTrue(all(str(snap) not in c[0] for c in seen))


if __name__ == "__main__":
    unittest.main()
