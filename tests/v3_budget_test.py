#!/usr/bin/env python3
"""E29: per-job daily budget (J2 > J1 > rest), J4 evidence-hash dedupe, claude failure detail, per-job usage in status/health."""
import json
from datetime import date
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
sys.dont_write_bytecode = True
import v3_entities_test as ents  # noqa: E402
import memrove_v3_entities as ent  # noqa: E402
import memrove_v3_health as health  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_queue as queue  # noqa: E402


class BudgetTest(ents.Base):
    def usage(self, **by):
        (self.state / 'llm-usage.json').write_text(json.dumps({'day': date.today().isoformat(), 'calls': sum(by.values()), 'by_job': by}))

    def test_priority_order_against_the_cap(self):
        (self.state / 'llm-config.json').write_text(json.dumps({'daily_cap': 20}))  # j2 reserve 10, j1 reserve 3, j4 share 5
        self.usage(j4=5)
        self.assertFalse(queue.allowed(self.state, 'j4', {'calls': 5, 'by_job': {'j4': 5}}))  # own share spent
        self.assertTrue(queue.allowed(self.state, 'j3', {'calls': 5, 'by_job': {'j4': 5}}))
        # 7 used, none by j2/j1: others may use up to cap - 10 - 3 = 7 calls
        busy = {'calls': 7, 'by_job': {'j3': 7}}
        self.assertFalse(queue.allowed(self.state, 'j3', busy))
        self.assertFalse(queue.allowed(self.state, 'reassess', busy))
        self.assertTrue(queue.allowed(self.state, 'j1', busy))  # j1 only leaves j2's reserve
        self.assertTrue(queue.allowed(self.state, 'j2', busy))
        full = {'calls': 20, 'by_job': {'j2': 10, 'j1': 3, 'j3': 7}}
        self.assertFalse(queue.allowed(self.state, 'j2', full))  # the hard cap still holds for j2
        # j2 may take everything the others left
        self.assertTrue(queue.allowed(self.state, 'j2', {'calls': 19, 'by_job': {'j3': 7, 'j2': 12}}))

    def test_call_refuses_low_priority_with_reserved_reason_and_counts_per_job(self):
        self.usage(j3=llm.DAILY_CAP * 2 // 3)  # j3 alone already eats past what is left after reserves
        st = {}
        self.reply['j4'] = {'sections': {}}
        self.assertIsNone(llm.call(self.state, self.vault, 'j4', '{}', status=st))
        self.assertEqual(st['reason'], 'reserved')
        self.assertFalse((queue.status(self.state))['cap_hit_today'])  # not the hard cap
        self.reply['j2'] = {}
        self.assertIsNotNone(llm.call(self.state, self.vault, 'j2', '[]'))
        self.assertEqual(queue.status(self.state)['by_job']['j2'], 1)

    def test_j4_same_evidence_costs_nothing(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.assertEqual(self.compile([f'RLS enabled. [{ents.A}]']), 1)
        # a re-tag marks the entity stale again, but the evidence text is unchanged
        ent.record_doc_entities(self.store, ents.A, [{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.assertTrue(self.ents()['entity:supabase']['stale'])
        before = sum(1 for _ in (self.state / 'llm-log.jsonl').read_text().splitlines())
        self.assertEqual(self.compile([f'other. [{ents.A}]']), 0)
        after = sum(1 for _ in (self.state / 'llm-log.jsonl').read_text().splitlines())
        self.assertEqual(after, before)
        e = self.ents()['entity:supabase']
        self.assertFalse(e['stale'])
        self.assertEqual(e['profile'][0]['t'], f'RLS enabled. [{ents.A}]')
        # changed evidence -> one new call
        (self.root / 'docs/supabase.md').write_text('# Supabase\n\nRLS is off.\n', encoding='utf-8')
        ents.docs.scan(self.store, self.root)
        self.assertEqual(self.compile([f'RLS disabled. [{ents.A}]']), 1)

    def test_j4_empty_answer_is_not_retried_and_failure_is_(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.assertEqual(self.compile(['Unsourced line.']), 0)  # answered, nothing citable
        before = len((self.state / 'llm-log.jsonl').read_text().splitlines())
        self.compile(['Unsourced line.'])
        self.assertEqual(len((self.state / 'llm-log.jsonl').read_text().splitlines()), before)
        ent.record_doc_entities(self.store, ents.B, [{'name': 'Tailwind', 'aliases': ['tailwind'], 'kind': 'tool'}])
        self.reply.pop('j4')  # runner fails
        ent.compile_entities(self.store, self.vault, self.state, self.root)
        self.assertTrue(self.ents()['entity:tailwind']['stale'])  # failed call: retried next run

    def test_j4_stops_when_budget_is_reserved(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}, {'name': 'Tailwind', 'aliases': ['tailwind'], 'kind': 'tool'}])
        (self.state / 'llm-usage.json').unlink()
        (self.state / 'llm-config.json').write_text(json.dumps({'daily_cap': 4}))  # j2 reserve 2, j1 reserve 1 -> others get 1
        self.compile([f'x. [{ents.A}]'])
        self.assertEqual(queue.status(self.state)['by_job'], {'j4': 1})


class ErrorDetailTest(unittest.TestCase):
    def run_claude(self, **kw):
        r = subprocess.CompletedProcess([], 1, **kw)
        with mock.patch.object(llm.shutil, 'which', return_value='/bin/claude'), mock.patch.object(llm.subprocess, 'run', return_value=r):
            return llm._claude('sys', 'payload')

    def test_stderr_is_in_the_error(self):
        _, _, err = self.run_claude(stdout='', stderr='Error: rate limit\nreached, retry later')
        self.assertEqual(err, 'claude-exit-1: Error: rate limit reached, retry later')

    def test_envelope_result_when_stderr_is_empty(self):
        _, _, err = self.run_claude(stdout=json.dumps({'is_error': True, 'result': 'Credit balance is too low'}), stderr='')
        self.assertEqual(err, 'claude-exit-1: Credit balance is too low')

    def test_failure_detail_reaches_record_error_and_health(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            state, vault = Path(d) / 's', Path(d) / 'v'
            state.mkdir()
            llm.RUNNER = lambda s, p, v=None: (None, 0.0, 'claude-exit-1: boom detail')
            try:
                llm.call(state, vault, 'j3', '[]')
            finally:
                llm.RUNNER = llm._claude
            err = queue.status(state)['last_error']
            self.assertEqual((err['where'], err['error']), ('j3', 'RuntimeError'))
            self.assertIn('boom detail', err['msg'])
            queue.window_failed(state, 'sid123456789', 0, 5, 'p', 'claude-exit-1: boom detail')
            self.assertEqual(queue.status(state)['last_error']['error'], 'claude-exit-1: boom detail')

    def test_health_shows_j2_starved_with_per_job_usage(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            state = Path(d)
            (state / 'llm-config.json').write_text(json.dumps({'daily_cap': 10}))
            (state / 'llm-usage.json').write_text(json.dumps({'day': date.today().isoformat(), 'calls': 10, 'by_job': {'j4': 8, 'j2': 2}}))
            with mock.patch.object(queue, '_behind', return_value=['s1', 's2']):
                sigs = health.signals(state)
            self.assertTrue(any(s.startswith('J2 starved') and 'j4 8, j2 2' in s and '2 session' in s for s in sigs), sigs)


if __name__ == '__main__':
    unittest.main()
