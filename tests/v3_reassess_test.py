#!/usr/bin/env python3
"""E17: new decision/lesson weakening an older cross-project assumption -> reassess marker (never supersede); stub RUNNER only."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, llm, cap  # noqa: E402
import memrove_v3_inject as inject  # noqa: E402
import v3_recall_test as rt  # noqa: E402

OLD = 'Haiku is enough: let us use Haiku for task X, no pricier model needed.'
NEW = dict(situation='Haiku for task X', approach='X with Haiku', result='failed', why='Haiku failed on task X, Sonnet was used',
           reuse='Haiku was not enough for X.')


class ReassessTest(Base):
    answer = {'weakens': 'yes', 'reason': 'Haiku sufficiency'}

    def runner(self, system, payload, vault=None):
        self.calls.append(payload)
        if self.answer is None:
            return None, 0.0, 'boom'
        return json.dumps({'items': [dict(self.answer, id=i['id']) for i in json.loads(payload)]}), 0.0, None

    def seed(self, new_text=None, lesson=None):
        self.engine.note_create('notes/captured/old.md', OLD, dict(id='old', kind='decision', project='appA', session='s1', verified_by='user_quote',
                                status='active', key=cap.key_of(OLD), polarity='pos', valid_from='2026-08-01'))
        lesson = lesson or NEW
        self.engine.note_create('notes/inferred/new.md', 'lesson', dict(id='new', kind='lesson', project='appB', session='s2', verified_by='llm', origin='llm',
                                status='active', key=cap.key_of(lesson['situation'] + ' ' + lesson['approach']), valid_from='2026-10-01', **lesson))

    def run_job(self):
        return llm.reassess(self.engine, self.vault, self.state)

    def test_cross_project_weakening_flagged_not_superseded(self):
        self.seed()
        self.assertEqual(self.run_job()['flagged'], 1)
        old = self.records()['old']
        self.assertEqual(old['reassess']['by'], 'new')
        self.assertEqual(old['status'], 'active')
        self.assertNotIn('valid_to', old)
        self.assertEqual(self.records()['new']['supersedes'], [])
        self.assertIn('[reassess -> #new: Haiku sufficiency]', inject.reassess_tag(old))
        self.assertEqual(len(self.calls), 1)
        self.run_job()  # new record is checked once
        self.assertEqual(len(self.calls), 1)

    def test_unrelated_not_flagged(self):
        self.seed(lesson=dict(NEW, situation='Postgres indeks', approach='partial index', why='query got faster', reuse='ok'))
        self.run_job()
        self.assertNotIn('reassess', self.records()['old'])
        self.assertEqual(self.calls, [])

    def test_evaluator_no_is_not_flagged(self):
        self.seed()
        self.answer = {'weakens': 'no', 'reason': ''}
        self.run_job()
        self.assertNotIn('reassess', self.records()['old'])

    def test_evaluator_failure_no_flag_and_retried(self):
        self.seed()
        self.answer = None
        self.run_job()
        self.assertNotIn('reassess', self.records()['old'])
        self.assertNotIn('reassess_checked', self.records()['new'])
        self.answer = {'weakens': 'yes', 'reason': 'Haiku sufficiency'}
        self.assertEqual(self.run_job()['flagged'], 1)

    def test_hub_link_uses_projections_slug(self):
        import re
        tp = self.root / 't.jsonl'
        tp.write_text(json.dumps({'type': 'user', 'message': {'content': 'hello world'}}) + '\n', encoding='utf-8')
        self.runner = lambda *a, **k: (json.dumps({'did': ['x'], 'decisions': [], 'rejected': [], 'open': [], 'outcomes': []}), 0.0, None)
        llm.RUNNER = self.runner
        ev = {'event': 'SessionEnd', 'session': 's9', 'transcript_path': str(tp), 'project': 'Şirket ıüş', 'at': 1}
        llm.session_notes(self.engine, self.vault, self.state, [ev])
        import memrove_v3_projections as pr
        pr.project_graph(self.engine, budget_s=30)
        note = next((self.vault / 'notes/sessions').glob('*.md')).read_text(encoding='utf-8')
        slug = re.search(r'\[\[memrove/projects/([^\]]+)\]\]', note).group(1)
        self.assertTrue((self.vault / 'memrove/projects' / (slug + '.md')).exists(), slug)

    def test_run_jobs_wires_reassess(self):
        self.seed()
        self.runner = lambda system, payload, vault=None: (json.dumps({'items': [dict(self.answer, id=i['id']) for i in json.loads(payload)]}), 0.0, None)
        llm.RUNNER = self.runner
        out = llm.run_jobs(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': 'x', 'project': 'appB'}])
        self.assertEqual(out['reassess']['flagged'], 1)
        self.assertEqual(self.records()['old']['reassess']['by'], 'new')


class RecallTagTest(rt.Base):
    def test_tag_in_recall(self):
        self.add('l1', **dict(rt.LESSON, reassess={'by': 'newer1', 'reason': 'idempotency changed', 'at': '2026-10-05'}))
        out = self.recall()
        self.assertIn('[reassess -> #newer1: idempotency changed]', out)


if __name__ == '__main__':
    unittest.main()
