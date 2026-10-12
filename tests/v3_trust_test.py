#!/usr/bin/env python3
"""E14: lesson quality (reuse line, evidence folding, reusability policy, recall boost) and the section-15 trust fields. Stub RUNNER only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, line, llm, cap, evaluator  # noqa: E402
from v3_lessons_test import OUT, EVID  # noqa: E402
import memrove_v3_trust as trust  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / 'template/.claude/scripts'
OUT = dict(OUT, reuse='Any small tool can skip the server when state is local.')


class LessonQualityTest(Base):
    def run_j2(self, sid, outcomes, text):
        tp = self.root / f'{sid}.jsonl'
        tp.write_text(line(type='user', message={'content': text}), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': [], 'outcomes': outcomes}
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': sid, 'transcript_path': str(tp), 'project': 'repo', 'at': 1}])

    def lessons(self):
        return [r for r in self.records().values() if r.get('kind') == 'lesson']

    def test_evidence_survives_unicode_punctuation_and_case(self):
        text = 'Decision “Redis – instead of SQLite” used, it’s FINE and no problem came up'
        ev = 'decision "redis - instead of sqlite" used, it\'s fine and no problem came up'
        self.run_j2('sA', [dict(OUT, evidence=ev)], text)
        self.assertEqual(len(self.lessons()), 1)

    def test_lesson_without_reuse_is_dropped_and_logged(self):
        self.run_j2('sA', [dict(OUT, reuse='')], 'x ' + EVID)
        self.assertEqual(self.lessons(), [])
        self.assertIn('no-reuse', (self.state / 'llm-drops.jsonl').read_text(encoding='utf-8'))

    def test_prompt_demands_reuse(self):
        self.assertIn('reuse is required', llm.SYSTEM['j2'])
        self.assertIn('"reuse"', llm.SYSTEM['j2'])

    def test_new_lesson_carries_trust_fields(self):
        self.run_j2('sA', [OUT], 'x ' + EVID)
        (r,) = self.lessons()
        self.assertEqual((r['reuse'], r['extraction_confidence'], r['source_reliability'], r['assessment_state'], r['egress']),
                         (OUT['reuse'], 0.5, 0.4, 'not_assessed', 'external_ok'))

    def test_judge_lessons_yes_no_and_failure(self):
        for i, (a, b) in enumerate((('alfa elma', 'masa kalem'), ('bravo armut', 'deniz bulut'), ('charlie kiraz', 'orman yagmur'))):
            self.run_j2(f's{i}', [dict(OUT, situation=f'Durum {i} {a}', approach=b)], 'x ' + EVID)
        by = {r['situation'][:7]: r['id'] for r in self.lessons()}
        self.reply['lesson_reusable'] = {'items': [{'id': by['Durum 0'], 'reusable': 'yes'}, {'id': by['Durum 1'], 'reusable': 'no'},
                                                   {'id': by['Durum 2'], 'reusable': 'unclear'}]}
        out = llm.judge_lessons(self.engine, self.vault, self.state)
        self.assertEqual((out['assessed'], out['dropped']), (3, 1))
        rs = self.records()
        self.assertEqual(rs[by['Durum 0']]['assessment_state'], 'assessed')
        self.assertEqual(rs[by['Durum 1']]['status'], 'dropped_llm')
        self.assertEqual(rs[by['Durum 2']]['assessment_state'], 'insufficient')
        self.assertEqual(trust.lesson_admission(rs[by['Durum 0']]), 'admit')
        self.assertEqual(trust.lesson_admission(rs[by['Durum 2']]), 'own_project')
        self.assertEqual(llm.judge_lessons(self.engine, self.vault, self.state)['pending'], 0)  # nothing unassessed left
        self.assertEqual(llm.judge_lessons(self.engine, self.vault, self.state, redo=True, dry_run=True)['pending'], 2)  # dropped one stays out

    def test_judge_lessons_failure_is_not_assessed_and_device_only_never_leaves(self):
        self.run_j2('sA', [OUT], 'x ' + EVID)
        self.reply.pop('lesson_reusable', None)  # runner errors
        self.assertEqual(llm.judge_lessons(self.engine, self.vault, self.state)['assessed'], 0)
        (r,) = self.lessons()
        self.assertEqual((r['assessment_state'], r['status']), ('not_assessed', 'active'))
        from memrove_v3_projections import update_record
        update_record(self.engine, r['id'], {'egress': 'device_only'})
        self.reply['lesson_reusable'] = {'items': [{'id': r['id'], 'reusable': 'yes'}]}
        n = len(self.calls)
        llm.judge_lessons(self.engine, self.vault, self.state)
        self.assertEqual(len(self.calls), n)



    def inferred(self, decisions, rejected=()):
        self.run_j2('sI', [], 'x')
        tp = self.root / 'sI2.jsonl'
        tp.write_text(line(type='user', message={'content': 'y'}), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'decisions': list(decisions), 'rejected': list(rejected), 'open': [], 'outcomes': []}
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': 'sI2', 'transcript_path': str(tp), 'project': 'repo', 'at': 2}])
        return {r['text'].strip(): r['id'] for r in self.records().values() if r.get('origin_type') == 'agent_inference' and r.get('kind') in ('decision', 'rejected')}

    def test_inferred_directive_dropped_standing_kept_failure_keeps(self):
        ids = self.inferred(['Detect the errors and fix them', 'Skills will be generic'])
        self.assertEqual(len(ids), 2)
        self.reply['j1'] = {'items': [dict(id=ids['Detect the errors and fix them'], standing_rule='no', durable='no', scope='project', rule_name='x'),
                                      dict(id=ids['Skills will be generic'], standing_rule='yes', durable='yes', scope='user', rule_name='generic skills')]}
        self.assertEqual(llm.judge_inferred(self.engine, self.vault, self.state), {'assessed': 2, 'dropped': 1, 'pending': 0})
        rs = self.records()
        self.assertEqual(rs[ids['Detect the errors and fix them']]['status'], 'dropped_llm')
        self.assertEqual(rs[ids['Skills will be generic']]['status'], 'active')

    def test_inferred_evaluator_failure_keeps_record(self):
        ids = self.inferred(['Review the other models'])
        self.reply.pop('j1', None)  # runner errors
        self.assertEqual(llm.judge_inferred(self.engine, self.vault, self.state)['assessed'], 0)
        self.assertEqual(self.records()[ids['Review the other models']]['status'], 'active')

    def test_worker_j1_batches_inferred_and_prompt_forbids_directives(self):
        ids = self.inferred(['Keep fixing'])
        self.reply['j1'] = {'items': [dict(id=i, standing_rule='no', durable='no', scope='project', rule_name='x') for i in ids.values()]}
        self.assertEqual(llm.judge_quotes(self.engine, self.vault, self.state), 1)
        self.assertEqual(self.records()[ids['Keep fixing']]['status'], 'dropped_llm')
        self.assertIn('task directive', llm.SYSTEM['j2'])



class TrustFieldsTest(Base):
    def test_split_fields_read_old_confidence(self):
        old = {'confidence': 0.7, 'origin_type': 'user_explicit'}
        self.assertEqual((trust.extraction_confidence(old), trust.source_reliability(old)), (0.7, 0.95))
        self.assertEqual(trust.extraction_confidence({'extraction_confidence': 0.2, 'confidence': 0.9}), 0.2)
        self.assertIsNone(trust.source_reliability({}))

    def test_assessment_state(self):
        a = lambda st: {'assessments': {'j1': {'state': st}}}
        self.assertEqual([trust.assessment_state(a('assessed')), trust.assessment_state(a('unclear'))], ['assessed', 'insufficient'])
        self.assertEqual(trust.assessment_state({'llm_verdict': 'keep'}), 'assessed')  # legacy
        self.assertEqual(trust.assessment_state({'verified_by': 'llm'}), 'not_assessed')
        self.assertEqual(trust.assessment_state({}), 'unknown')

    def test_egress_most_restrictive_wins(self):
        self.assertEqual(trust.combine_egress(['external_ok', 'self_hosted']), 'self_hosted')
        self.assertEqual(trust.combine_egress(['external_ok', 'device_only', 'self_hosted']), 'device_only')
        self.assertEqual(trust.egress_of({'verified_by': 'user_quote'}), 'external_ok')  # Claude Code transcript
        self.assertEqual(trust.egress_of({}), 'device_only')
        self.engine.note_create('notes/a.md', 'private source', {'id': 'src', 'egress': 'device_only'})
        self.engine.note_create('notes/b.md', 'open source', {'id': 'pub', 'egress': 'external_ok'})
        with self.engine.store._connect() as db:
            self.assertEqual(trust.egress_of({'egress': 'external_ok', 'derived_from': ['pub', 'src']}, db), 'device_only')
            self.assertEqual(trust.egress_of({'egress': 'external_ok', 'derived_from': ['pub', 'gone']}, db), 'device_only')

    def test_device_only_quote_never_reaches_remote_judge(self):
        self.quote('q1', 'decision', 'Always use tabs here please', egress='device_only')
        self.reply['j1'] = {'items': []}
        llm.judge_quotes(self.engine, self.vault, self.state)
        self.assertEqual(self.calls, [])

    def test_corroboration_counts_distinct_sessions_not_our_notes(self):
        mk = lambda **kw: dict({'kind': 'decision'}, **kw)
        recs = [mk(seen_sessions=['s1', 's2']), mk(session='s2'), mk(kind='session', origin='llm', session='s3')]
        self.assertEqual(trust.corroboration(recs), 2)
        self.assertEqual(trust.corroboration([mk(seen_sessions=['s1', 's1'])]), 1)

    def test_lesson_admission_policy(self):
        a = lambda ans, st='assessed': {'assessments': {'lesson_reusable': {'state': st, 'answers': {'reusable': ans}}}}
        self.assertEqual(trust.lesson_admission(a('yes')), 'admit')
        self.assertEqual(trust.lesson_admission(a('no'), sessions=5), 'drop')
        self.assertEqual(trust.lesson_admission({}, sessions=2), 'admit')
        self.assertEqual(trust.lesson_admission({}, sessions=1), 'own_project')
        self.assertEqual(trust.lesson_admission({'origin_type': 'user_explicit'}), 'admit')
        self.assertEqual(trust.lesson_admission(a('unclear', 'unclear')), 'own_project')

    def test_source_missing_marks_derived_records(self):
        self.engine.note_create('notes/src1.md', 'doc one', {'id': 'src1'})
        self.engine.note_create('notes/src2.md', 'doc two', {'id': 'src2'})
        self.engine.note_create('notes/inferred/d1.md', 'only from one', {'id': 'd1', 'kind': 'lesson', 'status': 'active', 'derived_from': ['src1']})
        self.engine.note_create('notes/inferred/d2.md', 'from two', {'id': 'd2', 'kind': 'lesson', 'derived_from': ['src1', 'src2']})
        (self.vault / 'notes/src1.md').unlink()
        self.engine.sync()
        rs = self.records()
        self.assertEqual((rs['d1']['status'], rs['d1']['status_before']), ('source_missing', 'active'))
        self.assertNotEqual(rs['d2'].get('status'), 'source_missing')  # src2 survives
        (self.vault / 'notes/src2.md').unlink()
        self.engine.sync()
        self.assertEqual(self.records()['d2']['status'], 'source_missing')


if __name__ == '__main__':
    unittest.main()
