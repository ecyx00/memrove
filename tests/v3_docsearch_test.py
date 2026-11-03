#!/usr/bin/env python3
"""Docs channel: an answer that lives only in docs/state comes back as a doc handle, below the messages."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_chatsearch as cs  # noqa: E402


class DocSearch(unittest.TestCase):
    def test_state_doc_is_a_handle(self):
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t) / "repo"
            (repo / "docs/state").mkdir(parents=True)
            (repo / "docs/state/2026-10-11-e69-clef.md").write_text(
                "# e69 clef flash\n\nintro\n\n## Result\n\nThe flash reranker reached 0.81 recall on the gold set.\n")
            (repo / "docs/state/2026-10-01-other.md").write_text("# other\n\nnothing about it\n")
            old = os.environ.get("MEMROVE_DOC_REPOS")
            os.environ["MEMROVE_DOC_REPOS"] = str(repo)
            try:
                out = cs.search(Path(t) / "state", "e69 flash reranker recall durumu")
                self.assertIn("doc|2026-10-11|", out)
                self.assertIn("2026-10-11-e69-clef.md|Result|", out)
                self.assertNotIn("other.md", out.split("docs (open")[1].splitlines()[1])
            finally:
                os.environ.pop("MEMROVE_DOC_REPOS") if old is None else os.environ.update(MEMROVE_DOC_REPOS=old)


class DocOpen(unittest.TestCase):
    def test_show_section_whole_and_refusals(self):
        import memrove_v3_docsearch as ds
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t) / "repo"
            (repo / "docs/state").mkdir(parents=True)
            (repo / "docs/other").mkdir()
            f = repo / "docs/state/2026-10-11-x.md"
            f.write_text("# X\n\nintro\n\n## Result\n\nbody one\n\n## Next\n\nlater\n")
            (repo / "docs/other/y.md").write_text("# no\n")
            (repo / "docs/state/link.md").symlink_to(repo / "docs/other/y.md")
            os.environ["MEMROVE_DOC_REPOS"] = str(repo)
            try:
                self.assertIn("later", ds.show(str(f)))
                sec = ds.show(str(f), "result")
                self.assertIn("body one", sec)
                self.assertNotIn("later", sec)
                (repo / "README.md").write_text("# no\n")
                (repo / "docs/state/out.md").symlink_to(repo / "README.md")   # symlink out of docs/
                for bad in (repo / "README.md", repo / "docs/state/out.md", f.parent / ".." / ".." / "README.md", Path("/etc/hosts")):
                    with self.assertRaises(ValueError):
                        ds.show(str(bad))
                with self.assertRaises(ValueError):
                    ds.show(str(f), "nope")
            finally:
                os.environ.pop("MEMROVE_DOC_REPOS", None)


class DocFolders(unittest.TestCase):
    def test_audits_and_handoff_are_indexed_and_openable(self):
        import memrove_v3_docsearch as ds
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t) / "repo"
            for sub, word in (("handoff", "zebrafinch"), ("audits", "quokkaaudit")):
                (repo / "docs" / sub).mkdir(parents=True)
                (repo / "docs" / sub / f"{sub}-x.md").write_text(f"# {sub}\n\n## Status\n\n{word} lives here\n")
            os.environ["MEMROVE_DOC_REPOS"] = str(repo)
            try:
                for sub, word in (("handoff", "zebrafinch"), ("audits", "quokkaaudit")):
                    self.assertIn(f"{sub}-x.md", cs.search(Path(t) / "state", word))
                    self.assertIn(word, ds.show(str(repo / "docs" / sub / f"{sub}-x.md")))
            finally:
                os.environ.pop("MEMROVE_DOC_REPOS", None)


class RepoGlobs(unittest.TestCase):
    def test_glob_entry_finds_new_repos_with_docs_only(self):
        import memrove_v3_docsearch as ds
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "a" / "docs" / "state").mkdir(parents=True)
            (root / "b").mkdir()                                   # no docs/: skipped
            (root / "solo" / "docs").mkdir(parents=True)
            os.environ["MEMROVE_DOC_REPOS"] = os.pathsep.join([str(root / "*"), str(root / "solo")])
            try:
                self.assertEqual(sorted(r.name for r in ds.repos()), ["a", "solo"])
                (root / "c" / "docs" / "design").mkdir(parents=True)   # added later: seen without a config change
                self.assertEqual(sorted(r.name for r in ds.repos()), ["a", "c", "solo"])
            finally:
                os.environ.pop("MEMROVE_DOC_REPOS", None)


class DocIndexOrphan(unittest.TestCase):
    def test_orphan_fts_row_does_not_break_refresh(self):
        import memrove_v3_docsearch as ds
        with tempfile.TemporaryDirectory() as t:
            repo = Path(t) / "repo"
            (repo / "docs/state").mkdir(parents=True)
            (repo / "docs/state/2026-10-18-a.md").write_text("# alpha\n\nfirst body\n")
            old = os.environ.get("MEMROVE_DOC_REPOS")
            os.environ["MEMROVE_DOC_REPOS"] = str(repo)
            try:
                state = Path(t) / "state"
                ds.chatlog.root(state).mkdir(parents=True, exist_ok=True)
                ds.handles(state, "alpha body")
                db = ds._db(state)
                nxt = db.execute("SELECT max(id) FROM s").fetchone()[0] + 1
                db.execute("INSERT INTO fts(rowid, head, body) VALUES (?, 'stale', 'stale')", (nxt,))
                db.commit(); db.close()
                (repo / "docs/state/2026-10-18-b.md").write_text("# beta\n\nsecond body\n")
                self.assertTrue(any("2026-10-18-b.md" in h for h in ds.handles(state, "beta second")))
            finally:
                os.environ.pop("MEMROVE_DOC_REPOS") if old is None else os.environ.update(MEMROVE_DOC_REPOS=old)


if __name__ == "__main__":
    unittest.main()
