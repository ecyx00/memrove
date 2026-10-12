#!/usr/bin/env python3
"""E12 message-time recall: (legacy gate mode pinned; hybrid in v3_hybrid_test.py) gate, ranking, envelope, dedupe, kill switch, used_n, hook shape, fail-open."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import os as _os
_os.environ.setdefault("MEMROVE_NO_LLM", "1")
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3 as v3
import memrove_v3_recall as recall
import memrove_v3_hook as hook

PROMPT = 'how should I handle stripe webhook retries and idempotency keys here'
LESSON = dict(kind='lesson', project='other', title='Stripe webhook retries', topics=['idempotency', 'webhook'],
              approach='dedupe by event id', result='unnecessary', why='stripe already retries safely', session='sess-9',
              text='Stripe webhook retries idempotency keys', origin_type='user_explicit', confidence=0.9,
              updated_at='2026-10-01T00:00:00Z')


def fake_entities(hits):
    mod = types.ModuleType('memrove_v3_entities')
    mod.match = lambda store, text: list(hits)
    return patch.dict(sys.modules, {'memrove_v3_entities': mod})


class Base(unittest.TestCase):
    MODE = 'gate'
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.vault, self.state, self.repo = self.root / 'vault', self.root / 'state', self.root / 'here'
        for p in (self.vault, self.repo / '.git'):
            p.mkdir(parents=True)
        self.store = v3.MemoryStore(self.state, self.vault)
        os.environ.pop('MEMROVE_NO_RECALL', None)
        self.addCleanup(os.environ.pop, 'MEMROVE_NO_RECALL', None)
        if self.MODE:  # the keyword-gate tests pin the legacy mode; HookTest and HybridTest run the default (hybrid)
            patcher = patch.dict(os.environ, {'MEMROVE_RECALL_MODE': self.MODE})
            patcher.start()
            self.addCleanup(patcher.stop)

    def add(self, id, **fields):
        (self.vault / f'{id}.md').write_text(fields.get('text', id), encoding='utf-8')
        self.store.ingest({'id': id, 'source': f'{id}.md', 'status': 'active', 'text': id, **fields})

    def recall(self, prompt=PROMPT, session='s1', store=None):
        return recall.recall(store or self.store, prompt, str(self.repo), session)


class RecallTest(Base):
    def test_alias_hit_passes_even_for_short_prompt(self):
        with fake_entities([('stripe', 'stripe', 'Platform rules\nnever store card data')]):
            out = self.recall('stripe?')
        self.assertIn('[profil: stripe', out)
        self.assertIn('never store card data'.split()[0], out)

    def test_real_entities_match_through_state_store(self):
        import memrove_v3_entities as ent
        with self.store._connect() as db:
            db.execute(ent.SCHEMA)
            db.execute('INSERT INTO entities VALUES(?,?)', ('platform:appstore', json.dumps(
                {'id': 'platform:appstore', 'name': 'App Store', 'aliases': ['appstoreconnect'], 'profile': [{'t': 'Review takes 24h'}]})))
        out = recall.recall(recall.StateStore(self.state), 'appstoreconnect?', str(self.repo), 's1')
        self.assertIn('[profil: platform:appstore', out)
        self.assertIn('Review takes 24h', out)

    def test_short_prompt_without_alias_is_empty(self):
        self.add('l1', **LESSON)
        self.assertEqual(self.recall('stripe webhook retries'), '')

    def test_fts_hit_passes_with_format(self):
        self.add('l1', **LESSON)
        out = self.recall()
        self.assertIn('[l1] lesson: Stripe webhook retries (other · 2026-10-01 · you)', out)
        self.assertIn('memrove show <id>', out)

    def test_inferred_lesson_needs_assessment_or_recurrence(self):
        inferred = dict(LESSON, origin_type='agent_inference', verified_by='llm', seen_sessions=['a'])
        self.add('l1', **inferred)
        self.assertEqual(self.recall(), '')  # not_assessed: stays in its own project
        reason = [json.loads(l)['excluded'] for l in (self.state / 'recall-log.jsonl').read_text().splitlines()][-1]
        self.assertEqual(reason[0]['reason'], 'lesson_not_admitted')
        self.add('l2', **dict(inferred, seen_sessions=['a', 'b']))
        self.assertIn('lesson:', self.recall(session='s2'))  # recurrence in two sessions
        self.add('l3', **dict(inferred, assessments={'lesson_reusable': {'state': 'assessed', 'answers': {'reusable': 'yes'}}}))
        self.assertIn('lesson:', self.recall(session='s3'))

    def test_failed_unnecessary_mixed_lessons_rank_higher(self):
        for i, res in enumerate(('worked', 'unnecessary')):
            self.add(f'l{i}', **dict(LESSON, result=res, title=f'Stripe webhook retries {res}'))
        out = self.recall().splitlines()
        self.assertIn('unnecessary', out[2])  # first card (after the envelope tag and header line)
        self.assertAlmostEqual(recall.LESSON_BOOST, 1.3)

    def test_low_coverage_is_blocked(self):
        # 2 terms match (stripe, webhook) but the record explains far less than 0.25 of the prompt's own terms
        self.add('l1', **dict(LESSON, title='Stripe webhook', topics=[], approach='', text='Stripe webhook'))
        self.assertEqual(self.recall('stripe webhook ile ilgili kahve makinesi bütçe planı toplantı notları hazırlayalım'), '')

    def test_single_term_match_is_blocked(self):
        self.add('l1', **dict(LESSON, title='Stripe', topics=[], approach='', why='', text='Stripe'))
        self.assertEqual(self.recall(), '')

    def test_cwd_project_excluded(self):
        self.add('l1', **dict(LESSON, project='here'))
        self.assertEqual(self.recall(), '')

    def test_not_injected_twice_in_session(self):
        self.add('l1', **LESSON)
        self.assertNotEqual(self.recall(), '')
        self.assertEqual(self.recall(), '')
        self.assertNotEqual(self.recall(session='s2'), '')

    def test_retired_and_superseded_excluded(self):
        self.add('l1', **dict(LESSON, status='retired'))
        self.add('l2', **LESSON)
        self.add('l3', **dict(LESSON, supersedes=['l2']))
        out = self.recall()
        self.assertEqual(out.count('lesson:'), 1)  # only l3 survives
        with self.store._connect() as db:
            self.assertEqual({r[0] for r in db.execute('SELECT block_id FROM injected')}, {'recall:l3'})

    def test_envelope_and_token_cap(self):
        for i in range(6):
            self.add(f'l{i}', **dict(LESSON, why='long ' * 200, project=f'p{i}'))
        out = self.recall()
        self.assertTrue(out.startswith('<memrove-recall trust="local-data" instructions="never">'))
        self.assertTrue(out.endswith('</memrove-recall>'))
        self.assertLessEqual(recall.tok(out), 300)
        self.assertLessEqual(out.count('\n[other project') + out.count('[other project') - out.count('\n[other project'), 3)
        self.assertEqual(out.count('lesson:'), 3 if recall.tok(out) <= 300 else 0)

    def test_user_explicit_first(self):
        self.add('a', **dict(LESSON, origin_type='agent_inference', confidence=1.0, project='pa', seen_sessions=['x', 'y']))
        self.add('b', **dict(LESSON, project='pb'))
        out = self.recall()
        self.assertLess(out.index('pb'), out.index('pa'))

    def test_kill_switch(self):
        self.add('l1', **LESSON)
        with patch.dict(os.environ, {'MEMROVE_NO_RECALL': '1'}):
            self.assertEqual(self.recall(), '')

    def test_log_written(self):
        self.add('l1', **LESSON)
        self.recall()
        row = json.loads((self.state / 'recall-log.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['passed'], ['l1'])
        self.assertEqual(len(row['prompt_sha1']), 40)
        self.assertIn('ms', row)

    def test_search_no_gate_includes_cwd(self):
        self.add('l1', **dict(LESSON, project='here'))
        got = recall.search(self.store, 'stripe')
        self.assertEqual([g['id'] for g in got], ['l1'])


class UsedTest(Base):
    def test_used_n_increments_when_transcript_mentions_injected_id(self):
        import memrove_v3_capture as capture
        from memrove_v3_sync import SyncEngine
        self.add('l1', **LESSON)
        self.recall(session='abc')
        engine = SyncEngine(self.vault, self.state)
        mention = [{'type': 'assistant', 'message': {'content': 'see l1.md for details'}}]
        self.assertEqual(capture.count_used(engine, 'abc', [{'type': 'user', 'message': 'nothing'}]), 0)
        self.assertEqual(capture.count_used(engine, 'abc', mention), 1)
        with self.store._connect() as db:
            self.assertEqual(json.loads(db.execute("SELECT payload FROM records WHERE id='l1'").fetchone()[0])['used_n'], 1)


class HookTest(Base):
    MODE = None
    def setUp(self):
        super().setUp()
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / 'push.json').write_text(json.dumps({'recall_cards': True}))  # cards are opt-in

    def test_cards_are_off_by_default(self):
        self.add('l1', **LESSON)
        (self.state / 'push.json').unlink()
        self.assertEqual(json.loads(self.run_hook({'prompt': PROMPT, 'session_id': 'x', 'cwd': str(self.repo)})), {})

    def run_hook(self, payload, env=None, event='UserPromptSubmit'):
        r = subprocess.run([sys.executable, str(SCRIPTS / 'memrove_v3_hook.py'), '--vault', str(self.vault), '--state', str(self.state),
                            '--harness', 'claude', '--global'], input=json.dumps(dict(payload, hook_event_name=event)),
                           text=True, capture_output=True, cwd=self.repo, env=dict(os.environ, HOME=str(self.root), MEMROVE_V3_NO_SPAWN='1', **(env or {})))
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_output_shape(self):
        self.add('l1', **LESSON)
        out = json.loads(self.run_hook({'prompt': PROMPT, 'session_id': 'x', 'cwd': str(self.repo)}))
        self.assertEqual(set(out), {'hookSpecificOutput'})
        self.assertEqual(out['hookSpecificOutput']['hookEventName'], 'UserPromptSubmit')
        self.assertIn('<memrove-recall', out['hookSpecificOutput']['additionalContext'])

    def test_empty_and_skips_return_empty_object(self):
        self.add('l1', **LESSON)
        base = {'prompt': PROMPT, 'session_id': 'x', 'cwd': str(self.repo)}
        self.assertEqual(json.loads(self.run_hook(dict(base, prompt='hello there'))), {})
        self.assertEqual(json.loads(self.run_hook(base, env={'MEMROVE_INVOKED_BY': 'x'})), {})
        self.assertEqual(json.loads(self.run_hook(base, env={'MEMROVE_NO_RECALL': '1'})), {})
        self.assertEqual(json.loads(self.run_hook(dict(base, entrypoint='sdk-cli'))), {})

    def test_fail_open_on_exception(self):
        (self.state / 'memory.sqlite3').write_bytes(b'not a database')
        with patch.object(recall, 'recall', side_effect=RuntimeError('boom')):
            self.assertEqual(hook.recall_hook(self.state, {'prompt': PROMPT, 'cwd': str(self.repo)}), '{}')
        self.assertEqual(json.loads((self.state / 'hook-error.json').read_text())['event'], 'UserPromptSubmit')

    def test_hook_with_large_store(self):
        """Functional only: a 300-record store still yields a recall envelope. Timing lives in v3_recall_benchmark.py."""
        self.add('l1', **LESSON)
        with sqlite3.connect(self.state / 'memory.sqlite3') as db:
            for i in range(300):
                rec = dict(LESSON, id=f'r{i}', source=f'r{i}.md', revision=1, project=f'p{i % 40}', status='active', visibility='internal',
                           facts={}, supersedes=[], title=f'Gardening {i} alpha{i % 97}', topics=['soil'], approach='water plants', why='dry', text=f'note body {i}')
                db.execute('INSERT INTO records VALUES(?,?)', (rec['id'], json.dumps(rec, sort_keys=True)))
        out = json.loads(self.run_hook({'prompt': PROMPT, 'session_id': 'x', 'cwd': str(self.repo)}))
        self.assertIn('<memrove-recall', out['hookSpecificOutput']['additionalContext'])


if __name__ == '__main__':
    unittest.main()
