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
                for bad in (repo / "docs/other/y.md", repo / "docs/state/link.md", f.parent / ".." / "other" / "y.md", Path("/etc/hosts")):
                    with self.assertRaises(ValueError):
                        ds.show(str(bad))
                with self.assertRaises(ValueError):
                    ds.show(str(f), "nope")
            finally:
                os.environ.pop("MEMROVE_DOC_REPOS", None)


if __name__ == "__main__":
    unittest.main()
