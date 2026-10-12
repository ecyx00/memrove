#!/usr/bin/env python3
"""E13c: System-One evaluator: providers, no fallback, egress, cache, stale revision, advisory-only policy, legacy fields."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (capture cues, evidence heuristics).
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3_evaluator as ev  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_inject as inject  # noqa: E402
from memrove_v3_projections import update_record  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402
import memrove_v3_capture as cap  # noqa: E402

OK = {'standing_rule': 'yes', 'durable': 'yes', 'scope': 'user', 'rule_name': 'short commits'}


class T(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-eval-')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.vault, self.state = root / 'vault', root / 'state'
        self.vault.mkdir(), self.state.mkdir()
        self.engine = SyncEngine(self.vault, self.state)
        self.calls, self.reply = [], lambda items: {'items': [dict(OK, id=i['id']) for i in items]}
        llm.RUNNER = self.runner
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)
        for k in ('MEMROVE_NO_LLM', 'MEMROVE_INVOKED_BY', 'MEMROVE_EVALUATOR'):
            saved = os.environ.pop(k, None)
            self.addCleanup(os.environ.pop, k, None)
            if saved:
                self.addCleanup(os.environ.__setitem__, k, saved)

    def runner(self, system, payload, vault=None):
        self.calls.append(payload)
        r = self.reply(json.loads(payload)) if self.reply else None
        return (json.dumps(r) if r else None), 0.0, (None if r else 'boom')

    def item(self, i='a', text='Commit mesajları kısa olsun.', **kw):
        return dict({'id': i, 'text': text, 'revision': 1}, **kw)

    def ev(self, items, **kw):
        return ev.evaluate(self.state, 'j1', items, vault=self.vault, **kw)

    def quote(self, rid, text, **extra):
        self.engine.note_create(f'notes/captured/{rid}.md', text, dict(id=rid, kind='preference', project='repo', session='s1', verified_by='user_quote',
                                status='active', key=cap.key_of(text), polarity='pos', **extra))

    def test_haiku_assessed(self):
        a = self.ev([self.item()])[0]
        self.assertEqual((a['state'], a['provider'], a['rubric_version'], a['answers']['scope']), ('assessed', 'haiku', 1, 'user'))

    def test_failure_is_not_assessed_without_second_provider(self):
        self.reply = None
        a = self.ev([self.item()])[0]
        self.assertEqual(a['state'], 'not_assessed')
        self.assertEqual(len(self.calls), 1)  # one haiku call, nothing else tried
        os.environ['MEMROVE_NO_LLM'] = '1'  # disabled provider
        self.assertEqual(self.ev([self.item('b', 'other text')])[0]['state'], 'not_assessed')
        self.assertEqual(len(self.calls), 1)

    def test_none_and_jev_stub_never_call_haiku(self):
        for p in ('none', 'jev'):
            self.assertEqual(self.ev([self.item()], provider=p)[0]['state'], 'not_assessed')
        os.environ['MEMROVE_EVALUATOR'] = 'none'
        self.assertEqual(self.ev([self.item()])[0]['provider'], 'none')
        self.assertEqual(self.calls, [])

    def test_egress_refusal(self):
        r = self.ev([self.item('a', egress='device_only'), self.item('b', 'x y', egress='self_hosted'), self.item('c', 'z w', egress='external_ok')])
        self.assertEqual([a['state'] for a in r], ['not_assessed', 'not_assessed', 'assessed'])
        self.assertEqual([i['id'] for i in json.loads(self.calls[0])], ['c'])

    def test_cache_hit_and_metrics(self):
        self.ev([self.item()])
        a = self.ev([self.item()])[0]
        self.assertEqual((a['state'], len(self.calls)), ('assessed', 1))
        m = ev.metrics(self.state)['haiku']
        self.assertEqual((m['calls'], m['cache_hit_rate'], m['not_assessed_rate']), (1, 0.5, 0.0))

    def test_stale_revision_discarded(self):
        self.quote('q1', 'Commit mesajları kısa olsun.')
        with self.engine.store._connect() as db:
            rev = json.loads(db.execute("SELECT payload FROM records WHERE id='q1'").fetchone()[0])['revision']
        res = self.ev([self.item('q1', revision=rev)])
        update_record(self.engine, 'q1', {'weight': 0.5})  # record moved on while the judge ran
        self.assertEqual(ev.apply(self.engine, self.state, res), [])
        self.assertEqual(ev.metrics(self.state)['haiku']['stale_revision'], 1)
        res = self.ev([self.item('q1', revision=rev + 1)])
        self.assertEqual(ev.apply(self.engine, self.state, res), ['q1'])

    def test_judge_quotes_policy_drops_not_evaluator(self):
        self.quote('q1', 'Bu arada yeni sürümü kullanalım hemen şimdi.')
        self.reply = lambda items: {'items': [dict(OK, id=i['id'], standing_rule='no', durable='no') for i in items]}
        res = ev.evaluate(self.state, 'j1', [self.item('q1', revision=1)], vault=self.vault)
        ev.apply(self.engine, self.state, res)
        with self.engine.store._connect() as db:
            r = json.loads(db.execute("SELECT payload FROM records WHERE id='q1'").fetchone()[0])
        self.assertEqual(r['status'], 'active')  # evaluator alone never sets status
        os.remove(self.state / 'evaluator.db')
        self.assertEqual(llm.judge_quotes(self.engine, self.vault, self.state), 0)  # already assessed -> skipped

    def test_gate_new_and_legacy_fields_identical(self):
        mk = lambda rid, **kw: dict(id=rid, kind='preference', verified_by='user_quote', project='p', quote='Commit mesajları kısa olsun ve net.', **kw)
        new = mk('n', assessments={'j1': {'state': 'assessed', 'answers': {'durable': 'yes', 'standing_rule': 'yes'}}})
        old = mk('o', llm_verdict='keep', llm_durable=True)
        no = mk('x', assessments={'j1': {'state': 'not_assessed', 'answers': {}}})
        unassessed = mk('u')
        self.assertEqual([r['id'] for r in (new, old, no, unassessed) if inject.j1_view(r)['durable']], ['n', 'o'])
        self.assertTrue(inject.j1_view(mk('d', assessments={'j1': {'state': 'assessed', 'answers': {'standing_rule': 'no'}}}))['drop'])
        self.assertTrue(inject.j1_view(mk('e', llm_verdict='drop'))['drop'])


if __name__ == '__main__':
    unittest.main()
