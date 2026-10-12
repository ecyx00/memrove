#!/usr/bin/env python3
"""Concept wiki tools: checker (new section shape, number rules), publish (index writer), evidence pack."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_recall as recall  # noqa: E402
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as chatsearch  # noqa: E402
import memrove_v3_wiki as wiki  # noqa: E402

CONCEPT = """---
project: demo
root: {root}
{roots}title: K
group: Concepts
updated: 2026-10-10
---
# K
One sentence. Second sentence.
Analogy: x
Spine: A → B
## Problem
- {basics} [general]
## Skeleton
- A part. [general]
## Flow
- A step. [general]
- Breaks: a → b → c [general]
## Solution and cost
- a → b → c [general]
## Key distinctions
- X differs from Y. [general]
## What we researched
### A source (repo)
- What they did. [doc:docs/state/2026-10-09-x.md]
## Ours
- Sorun. [m:7]
Our measurement:
| measurement | source | what theory says |
|---|---|---|
{own}
## Where you see it
- [f](code:a.py:1)
## Subtopics to know
- [[other]] — neighbouring topic. [general]
## Compression
- One sentence. [general]
<details><summary>Test yourself</summary>
1. question?
</details>
<details><summary>Layer 4</summary>
- 40% [doc:docs/state/2026-10-09-x.md]
</details>
"""

OLD = """---
project: demo
root: {root}
title: K
---
# K
## Temel
- x [general]
## How it works
- x [general]
"""


class WikiCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        git = lambda *a: subprocess.run(['git', '-C', str(self.tmp), *a], check=True, capture_output=True)
        git('init'); (self.tmp / 'docs/state').mkdir(parents=True)
        (self.tmp / 'docs/state/2026-10-09-x.md').write_text('bm25 rrf notes\nunrelated\n'); (self.tmp / 'a.py').write_text('x = 1\n')
        git('add', '.'); git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-m', 'init')
        self.sha = subprocess.run(['git', '-C', str(self.tmp), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        recs = {'rec-1': {'id': 'rec-1', 'valid_from': '2026-10-09', 'status': 'active'}}
        def show(store, rid):
            if rid not in recs:
                raise ValueError(rid)
            return recs[rid]
        self.saved = recall.show, chatlog.get, recall.search
        recall.show = show
        recall.search = lambda store, q, n: []
        chatlog.get = lambda state, i: {'date': '2026-10-09T10:00:00+00:00', 'kind': 'user', 'text': 'bm25 or rrf?'} if i == 7 else None

    def tearDown(self):
        recall.show, chatlog.get, recall.search = self.saved

    def page(self, name='k', **kw):
        (self.tmp / 'other.md').write_text('x')
        d = self.tmp / 'wiki' / '_concepts'; d.mkdir(parents=True, exist_ok=True)
        (d / 'other.md').write_text('x')
        p = d / f'{name}.md'
        args = dict(basics='A cache is an intermediate memory.', own='| measurement | [m:7] | theory |', roots='')
        args.update(kw)
        p.write_text(CONCEPT.format(root=self.tmp, **args))
        return p

    def run_concept(self, **kw):
        return wiki.check(self.page(**kw), None, None)

    def test_old_sections_rejected(self):
        p = self.tmp / 'old.md'
        p.write_text(OLD.format(root=self.tmp))
        errs = wiki.check(p, None, None)
        self.assertTrue(any('sections must be exactly' in e for e in errs), errs)

    def test_valid_new_shape_passes(self):
        self.assertEqual(self.run_concept(), [])

    def test_real_tags_pass(self):
        self.assertEqual(self.run_concept(own=f'| measurement | [commit:{self.sha}] [memrove:rec-1] | theory |'), [])

    def test_fabricated_commit_and_memrove_id_fail(self):
        errs = self.run_concept(own='| measurement | [commit:deadbeef1] [memrove:rec-fake] | theory |')
        self.assertTrue(any('commit:deadbeef1' in e for e in errs) and any('memrove:rec-fake' in e for e in errs), errs)

    def test_general_line_with_number_fails(self):
        errs = self.run_concept(basics='The cache is 10 times faster.')
        self.assertTrue(any('[general] line has a number' in e for e in errs), errs)

    def test_backticked_pure_number_in_general_fails(self):
        for tok in ('1024', '3 token', '40%', '16x', '1/61 + 1/63', '2.5ms'):
            errs = self.run_concept(basics=f'Size is `{tok}`.')
            self.assertTrue(any('[general] line has a number' in e for e in errs), (tok, errs))

    def test_backticked_identifier_with_digits_passes(self):
        for tok in ('potion-code-16M', 'k1', 'BM25', 'Qwen3'):
            self.assertEqual(self.run_concept(basics=f'Name is `{tok}`.'), [], tok)

    def test_general_name_with_digit_passes_but_quantity_fails(self):
        self.assertEqual(self.run_concept(basics='BM25 is a scoring function.'), [])
        self.assertTrue(self.run_concept(basics='The BM25 score is 25.'))

    def test_general_number_inside_math_passes(self):
        self.assertEqual(self.run_concept(basics='Cost is $O(n^2)$.'), [])
        self.assertTrue(self.run_concept(basics='Cost is \\$5.'))

    def test_own_measurement_with_only_general_fails(self):
        errs = self.run_concept(own='| measurement | none | theory [general] |')
        self.assertTrue(any('Our measurement line' in e for e in errs), errs)

    def test_src_tag_resolves_through_snapshot(self):
        vault = self.tmp / 'v'
        snap = vault / 'wiki' / '_sources' / 'snap' / 'semble'
        snap.mkdir(parents=True); (snap / 'a.py').write_text('K = 16\nx\n')
        line = '- Constant 16. [src:semble:a.py:1]'
        text = self.page().read_text().replace('- What they did. [doc:docs/state/2026-10-09-x.md]', line)
        p = self.tmp / 'wiki' / '_concepts' / 's.md'
        p.write_text(text)
        self.assertEqual(wiki.check(p, None, None, vault), [])
        p.write_text(text.replace('Constant 16.', 'Constant 17.'))
        self.assertTrue(any('17' in e for e in wiki.check(p, None, None, vault)))
        p.write_text(text.replace('a.py:1', 'a.py:9'))
        self.assertTrue(wiki.check(p, None, None, vault))

    def test_cross_project_code_links(self):
        memrove = f'roots: {{"memrove": "{ROOT}"}}\n'
        good = CONCEPT.replace('- [f](code:a.py:1)', '- [d](code:@memrove/template/.claude/scripts/memrove_v3_hybrid.py#decide) [general]')
        bad = CONCEPT.replace('- [f](code:a.py:1)', '- [d](code:@unknown/x.py#f) [general]')
        d = self.page().parent
        for name, text, want in (('g', good, False), ('b', bad, True)):
            (d / f'{name}.md').write_text(text.format(root=self.tmp, basics='x', own='| a | [m:7] | b |', roots=memrove))
            errs = wiki.check(d / f'{name}.md', None, None)
            self.assertEqual(bool(errs), want, errs)

    # ---- publish ----
    def test_publish_two_roots_idempotent(self):
        vault = self.tmp / 'v'
        other = self.tmp / 'other'; other.mkdir()
        p = self.page(roots=f'roots: {{"demo": "{self.tmp}", "other": "{other}"}}\n')
        self.assertEqual(wiki.publish(p, None, None, vault), [])
        idx = {n: vault / 'wiki' / n / 'index.json' for n in ('demo', 'other')}
        first = {n: f.read_text() for n, f in idx.items()}
        e = json.loads(first['other'])['groups'][0]['pages'][0]
        self.assertEqual((e['slug'], e['title'], e['one_liner'], e['file']), ('k', 'K', 'One sentence.', '../_concepts/k.md'))
        self.assertEqual(json.loads(first['other'])['root'], str(other))
        self.assertEqual(wiki.publish(p, None, None, vault), [])
        self.assertEqual(first, {n: f.read_text() for n, f in idx.items()})
        self.assertEqual(len(json.loads(first['demo'])['groups'][0]['pages']), 1)

    def test_publish_failing_page_writes_nothing(self):
        vault = self.tmp / 'v'
        p = self.page(basics='The cache is 10 times faster.')
        self.assertTrue(wiki.publish(p, None, None, vault))
        self.assertFalse((vault / 'wiki').exists())

    def test_publish_general_project(self):
        vault = self.tmp / 'v'
        p = self.page()
        t = p.read_text().replace('project: demo', 'project: _general').replace(f'root: {self.tmp}\n', '')
        t = t.replace('[doc:docs/', f'[doc:{self.tmp}/docs/').replace('- [f](code:a.py:1)', '- None. [general]')
        p.write_text(t)
        self.assertEqual(wiki.publish(p, None, None, vault), [])
        idx = json.loads((vault / 'wiki' / '_general' / 'index.json').read_text())
        self.assertIsNone(idx['root'])
        self.assertEqual(idx['groups'][0]['pages'][0]['slug'], 'k')

    # ---- evidence ----
    def test_evidence_pack_blocks(self):
        vault = self.tmp / 'v'
        doc = self.tmp / 'docs/state/2026-10-09-x.md'
        cat = vault / 'wiki' / '_sources'
        (cat / 'snap' / 'semble').mkdir(parents=True)
        (cat / 'catalog.json').write_text(json.dumps({'items': [{
            'id': 'github:minishlab/semble', 'kind': 'repo', 'title': 'semble', 'url': 'https://github.com/MinishLab/semble',
            'mentions': [{'research': 'x', 'verdict': 'inspired', 'why': 'w', 'doc': str(doc), 'name': 'semble', 'quote': 'We trialled semble'}]},
            {'id': 'github:other/other', 'kind': 'repo', 'title': 'other', 'url': None,
             'mentions': [{'research': 'y', 'verdict': 'adopted', 'why': 'w', 'doc': '/nowhere.md', 'name': 'other', 'quote': 'NOT HERE'}]}]}))
        saved = wiki.RESEARCH_DIR
        wiki.RESEARCH_DIR = self.tmp / 'no-research'
        try:
            topic = wiki.parse_topic(f'Hybrid::terms=bm25,rrf::roots=demo={self.tmp}::msg=7')
            out = wiki.evidence_pack(None, None, vault, topic)
        finally:
            wiki.RESEARCH_DIR = saved
        self.assertIn('[m:7] user 2026-10-09', out)
        self.assertIn(f'[doc:{doc}]', out)
        self.assertIn('bm25 rrf notes', out)
        self.assertNotIn('unrelated', out)
        self.assertIn('github:minishlab/semble', out)
        self.assertIn('We trialled semble', out)
        self.assertIn('snapshot: semble', out)
        self.assertNotIn('NOT HERE', out)
        self.assertLessEqual(len(out), wiki.PACK_CAP)

    def test_pack_is_capped(self):
        self.assertEqual(len(wiki._cap('x' * 100000)) <= wiki.PACK_CAP, True)


if __name__ == '__main__':
    unittest.main()
