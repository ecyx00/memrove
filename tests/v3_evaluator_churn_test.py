#!/usr/bin/env python3
"""Evaluator churn: blocked items are not candidates, dropped_llm is not restated, doctor metrics measure state not retries. Stub RUNNER only."""
import json
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, llm, evaluator  # noqa: E402
from v3_trust_test import OUT, EVID, line  # noqa: E402
from memrove_v3_projections import update_record  # noqa: E402
import memrove_v3_restate as restate  # noqa: E402


class ChurnTest(Base):
    def run_j2(self, sid, outcomes, text):
        tp = self.root / f'{sid}.jsonl'
        tp.write_text(line(type='user', message={'content': text}), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': [], 'outcomes': outcomes}
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': sid, 'transcript_path': str(tp), 'project': 'repo', 'at': 1}])

    def lessons(self):
        return [r for r in self.records().values() if r.get('kind') == 'lesson']

    def test_blocked_newest_lessons_do_not_starve_external_ok(self):
        for i, (a, b) in enumerate((('alfa elma', 'masa kalem'), ('bravo armut', 'deniz bulut'), ('charlie kiraz', 'orman yagmur'))):
            self.run_j2(f's{i}', [dict(OUT, situation=f'Durum {i} {a}', approach=b)], 'x ' + EVID)
        by = {r['situation'][:7]: r['id'] for r in self.lessons()}
        time.sleep(0.01)
        for k in ('Durum 1', 'Durum 2'):  # updated last = newest
            update_record(self.engine, by[k], {'egress': 'device_only'})
            time.sleep(0.01)
        self.reply['lesson_reusable'] = {'items': [{'id': by['Durum 0'], 'reusable': 'yes'}]}
        out = llm.judge_lessons(self.engine, self.vault, self.state, limit=2)
        self.assertEqual(out['assessed'], 1)
        self.assertEqual(self.records()[by['Durum 0']]['assessment_state'], 'assessed')
        m = evaluator.metrics(self.state).get('haiku', {})
        self.assertEqual(m.get('egress_refused', 0), 0)  # blocked ones were never offered

    def test_j1_blocked_newest_do_not_take_slots(self):
        for rid in ('a', 'b', 'c', 'd'):  # a, b oldest and visible; c, d newest and device_only
            self.engine.note_create(f'notes/captured/{rid}.md', 'Commit mesajları kısa olsun.', dict(
                id=rid, kind='preference', project='repo', session='s1', verified_by='user_quote', status='active', key=rid, polarity='pos'))
            time.sleep(0.01)
        for rid in ('c', 'd'):
            update_record(self.engine, rid, {'egress': 'device_only', 'updated_at': '2099-01-01'})  # newest
        self.reply['j1'] = {'items': [{'id': i, 'standing_rule': 'yes', 'durable': 'yes', 'scope': 'user', 'rule_name': 'r'} for i in 'abcd']}
        old, llm.MAX_J1 = llm.MAX_J1, 2
        try:
            self.assertEqual(llm.judge_quotes(self.engine, self.vault, self.state), 2)
        finally:
            llm.MAX_J1 = old
        st = {i: r.get('assessment_state') for i, r in self.records().items()}
        self.assertEqual((st['a'], st['b']), ('assessed', 'assessed'))
        self.assertNotEqual(st.get('c'), 'assessed')
        self.assertEqual(evaluator.metrics(self.state).get('haiku', {}).get('egress_refused', 0), 0)

    def test_restate_pending_skips_dropped_llm(self):
        for rid, st in (('q1', 'active'), ('q2', 'dropped_llm')):
            self.engine.note_create(f'notes/captured/{rid}.md', 'Commit mesajları kısa olsun.', dict(
                id=rid, kind='preference', project='repo', session='s1', verified_by='user_quote', status=st, key=rid, polarity='pos'))
        with self.engine.store._connect() as db:
            self.assertEqual([r['id'] for r in restate.pending(db)], ['q1'])

    def test_metrics_window_and_refusals(self):
        now = time.time()
        with evaluator._db(self.state) as db:
            rows = [(now - 30 * 86400, 'haiku', 'x', 'item_not_assessed', 1, 0)] * 10  # old churn
            rows += [(now, 'haiku', 'x', 'egress_refused', 1, 0)] * 5 + [(now, 'haiku', 'x', 'item_not_assessed', 1, 0)] * 5  # refusals also log item_not_assessed
            rows += [(now, 'haiku', 'x', 'item_assessed', 1, 0)] * 6 + [(now, 'haiku', 'x', 'item_not_assessed', 1, 0)] * 2
            db.executemany('INSERT INTO log VALUES(?,?,?,?,?,?)', rows)
        m = evaluator.metrics(self.state)['haiku']
        self.assertEqual((m['items'], m['not_assessed_rate'], m['egress_refused'], m['egress_refused_total']), (8, 0.25, 5, 5))


if __name__ == '__main__':
    unittest.main()
