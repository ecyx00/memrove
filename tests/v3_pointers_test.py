#!/usr/bin/env python3
"""Prompt pointers: matching terms -> up to 3 `path — decision:` lines, cwd repo first; no match -> none. Doc roots: flat docs/, no worktrees."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_docsearch as ds  # noqa: E402


def doc(repo, rel, decision, body="the reranker flash recall gold set"):
    p = repo / "docs" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\ntopic: x\nstatus: done\ndecision: {decision}\nnext: y\n---\n# T\n\n{body}\n")
    return p


class Pointers(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        root = Path(self.t.name)
        self.a, self.b, self.state = root / "a", root / "b", root / "state"
        self.state.mkdir()
        doc(self.a, "state/2026-10-01-one.md", "use flash reranker " + "x" * 200)
        for i in range(4):
            doc(self.b, f"state/2026-10-1{i}-n{i}.md", f"b decision {i}")
        doc(self.a, "flat.md", "flat docs count")
        os.environ["MEMROVE_DOC_REPOS"] = os.pathsep.join([str(self.a), str(self.b)])

    def tearDown(self):
        os.environ.pop("MEMROVE_DOC_REPOS", None)
        self.t.cleanup()

    def test_match_capped_cwd_first_cut(self):
        out = ds.pointers(self.state, "what about the flash reranker recall on the gold set", str(self.a / "sub"))
        self.assertEqual(len(out), 3)
        self.assertIn("/a/docs/", out[0])   # cwd repo first
        self.assertTrue(all(" — decision: " in o for o in out))
        self.assertLessEqual(max(len(o.split(" — decision: ")[1]) for o in out), 101)
        self.assertIn("flat.md", "".join(out))                           # docs/*.md outside the old subdirs is indexed

    def test_no_match_none(self):
        self.assertEqual(ds.pointers(self.state, "completely unrelated zebra banana question", str(self.a)), [])

    def test_worktree_excluded(self):
        wt = Path(self.t.name) / "w" / "a-wt-1"
        doc(wt, "state/2026-10-02-wt.md", "worktree copy")
        (wt / ".git").write_text("gitdir: /x")
        (Path(self.t.name) / "w" / "real").mkdir()
        doc(Path(self.t.name) / "w" / "real", "state/r.md", "real")
        os.environ["MEMROVE_DOC_REPOS"] = str(Path(self.t.name) / "w" / "*")
        self.assertEqual([r.name for r in ds.repos()], ["real"])


class PastrefPointersGate(unittest.TestCase):
    """pastref_line adds doc pointers only with MEMROVE_PASTREF_POINTERS=1 (they cost 2.3 s against the 2 s hook timeout)."""
    def test_pointers_only_with_env(self):
        import memrove_v3_hook as hook
        import memrove_v3_pastref as pr
        from unittest.mock import patch
        payload = {"prompt": "geçen sefer ne demiştik?", "session_id": "s1"}
        with patch.dict("os.environ", {"MEMROVE_PASTREF": "1"}), patch.object(hook, "doc_pointers", return_value="\nPOINTERS"):
            os.environ.pop("MEMROVE_PASTREF_POINTERS", None)
            plain = hook.pastref_line(payload)
            self.assertTrue(plain.startswith(pr.LINE))
            os.environ["MEMROVE_PASTREF_POINTERS"] = "1"
            self.assertEqual(hook.pastref_line(payload), plain + "\nPOINTERS")


if __name__ == "__main__":
    unittest.main()
