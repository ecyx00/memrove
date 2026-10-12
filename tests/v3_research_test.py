#!/usr/bin/env python3
"""Research shelf: the 0-token catalog checker and the merge rule."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_research as research  # noqa: E402

DOC = "doc.md"
TEXT = "We trialled semble (MinishLab) for search. See https://github.com/MinishLab/semble and https://arxiv.org/abs/2401.01234 .\nstars 6180"
DOCS = {DOC: TEXT}


def mention(**kw):
    m = {"research": "r", "verdict": "inspired", "why": "x", "doc": DOC, "name": "semble", "quote": "We trialled semble"}
    m.update(kw)
    return m


def item(url=None, mentions=None, **kw):
    it = {"id": research.item_id(url, "semble"), "kind": "repo", "title": "semble", "url": url, "one_liner": "x",
          "facts": {}, "mentions": mentions or [mention()], "concepts": [], "projects": []}
    it.update(kw)
    return it


class ResearchTest(unittest.TestCase):
    def test_valid_entry_passes(self):
        self.assertEqual(research.check_item(item(), DOCS), [])
        self.assertEqual(research.check_item(item("https://github.com/MinishLab/semble", [mention(url="https://github.com/MinishLab/semble")]), DOCS), [])

    def test_name_not_verbatim_rejected(self):
        errs = research.check_item(item(mentions=[mention(name="cqs", quote="We trialled semble")]), DOCS)
        self.assertTrue(any("cqs" in e for e in errs))

    def test_url_not_in_doc_rejected(self):
        url = "https://github.com/Invented/semble"
        errs = research.check_item(item(url, [mention(url=url)]), DOCS)
        self.assertTrue(any("Invented" in e for e in errs))

    def test_same_url_merges(self):
        a = {"name": "semble", "kind": "repo", "url": "https://github.com/MinishLab/semble", "one_liner": "a"}
        b = {"name": "semble (MinishLab)", "kind": "repo", "url": "https://github.com/minishlab/semble/", "one_liner": "b"}
        self.assertEqual(research.item_id(a["url"], a["name"]), research.item_id(b["url"], b["name"]))
        self.assertEqual(research.item_id(a["url"], a["name"]), "github:minishlab/semble")

    def test_different_owner_same_repo_does_not_merge(self):
        self.assertNotEqual(research.item_id("https://github.com/a/tool", "tool"), research.item_id("https://github.com/b/tool", "tool"))

    def test_name_only_id_and_no_owner_invented(self):
        self.assertEqual(research.item_id(None, "semble (MinishLab)"), "name:semble")
        merged = research.merge([("r1", DOC, {"name": "semble", "kind": "repo", "url": None, "verdict": "inspired", "why": "", "one_liner": "", "quote": "semble"}),
                                 ("r2", DOC, {"name": "semble (MinishLab)", "kind": "repo", "url": None, "verdict": "adopted", "why": "", "one_liner": "", "quote": "semble"})])
        self.assertEqual(len(merged), 1)
        self.assertIsNone(merged[0]["url"])


if __name__ == "__main__":
    unittest.main()
