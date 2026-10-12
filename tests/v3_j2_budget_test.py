#!/usr/bin/env python3
"""E35: J2 fragmentation guards (min-size gate for PreCompact/idle, SessionEnd flush, one transcript under two ids, subagent skip) and J4 behind J2."""
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import v3_llm_test as base  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_queue as queue  # noqa: E402

GOOD = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': []}


class J2BudgetTest(base.Base):
    def transcript(self, name, n=1, size=40):
        tp = self.root / f'{name}.jsonl'
        tp.write_text(''.join(base.line(type='user', message={'content': f'{name} {i} ' + 'x' * size}) for i in range(n)), encoding='utf-8')
        return tp

    def ev(self, sid, tp, event='PreCompact', at=None):
        return {'event': event, 'session': sid, 'transcript_path': str(tp), 'project': 'repo', 'at': at or time.time()}

    def runner(self, system, payload, vault=None):
        self.calls.append(('j2', payload))
        return json.dumps(GOOD), 0.0006, None

    def notes(self, events):
        return llm.session_notes(self.engine, self.vault, self.state, events)

    def test_small_precompact_waits_and_session_end_flushes(self):
        tp = self.transcript('s1', n=2)
        self.assertEqual(self.notes([self.ev('s1', tp)]), 0)
        self.assertEqual(queue.status(self.state)['pending'], 0)  # deferred text is not "pending": the worker is not woken for it
        self.assertEqual(self.notes([self.ev('s1', tp, 'SessionEnd')]), 1)

    def test_enough_text_runs_on_precompact_and_quiet_tail_is_flushed(self):
        big = self.transcript('s2', n=8, size=500)
        self.assertEqual(self.notes([self.ev('s2', big)]), 1)
        tp = self.transcript('s3', n=1)
        old = self.ev('s3', tp, at=time.time() - queue.DEFAULT_TAIL_S - 5)
        queue.register(self.state, [old])
        self.assertEqual(queue.status(self.state)['pending'], 1)

    def test_one_transcript_under_two_ids_is_sent_once(self):
        tp = self.transcript('s4', n=1)
        self.notes([self.ev('raw-uuid', tp, 'SessionEnd', at=1)])
        n = len(self.calls)
        tp.write_text(tp.read_text() + base.line(type='user', message={'content': 'second window text'}), encoding='utf-8')
        self.notes([self.ev('hashed', tp, 'SessionEnd', at=2), self.ev('raw-uuid', tp, 'SessionEnd', at=3)])
        self.assertEqual(len(self.calls) - n, 1)
        cur = json.loads((self.state / 'llm-cursors.json').read_text())
        self.assertEqual(cur['hashed'], cur['raw-uuid'])
        # a drain from the registry sees only one id for the transcript
        queue.register(self.state, [self.ev('raw-uuid', tp, 'SessionEnd', at=1), self.ev('hashed', tp, 'SessionEnd', at=2)])
        tp.write_text(tp.read_text() + base.line(type='user', message={'content': 'third'}), encoding='utf-8')
        self.assertEqual(queue.status(self.state)['pending'], 1)

    def test_subagent_transcripts_are_never_sessions(self):
        sub = self.root / 'agent-ab12cd34.jsonl'
        sub.write_text(base.line(type='user', message={'content': 'delegated prompt'}), encoding='utf-8')
        nested = self.root / 'abc' / 'subagents'
        nested.mkdir(parents=True)
        (nested / 'agent-x.jsonl').write_text(sub.read_text(), encoding='utf-8')
        evs = [self.ev('agent-ab12cd34', sub, 'SessionEnd'), self.ev('other', nested / 'agent-x.jsonl', 'SessionEnd')]
        queue.register(self.state, evs)
        self.assertFalse((self.state / 'llm-sessions.json').exists())
        self.assertEqual(self.notes(evs), 0)
        self.assertEqual(self.calls, [])

    def test_j4_waits_while_a_j2_window_is_due(self):
        tp = self.transcript('s5', n=1)
        queue.register(self.state, [self.ev('s5', tp, 'SessionEnd')])
        import memrove_v3_entities as ent
        seen, orig, notes = [], ent.compile_entities, llm.session_notes
        ent.compile_entities = lambda *a, **k: seen.append(1) or 0
        llm.session_notes = lambda *a, **k: 0  # J2 could not run (cap, error): its window stays due
        try:
            self.assertTrue(queue.sweep_due(self.state))
            self.assertEqual(llm.run_jobs(self.engine, self.vault, self.state, [], str(self.root))['j4'], 'deferred-j2')
            self.assertEqual(seen, [])
            llm.session_notes = notes
            self.notes([])  # drained
            self.assertFalse(queue.sweep_due(self.state))
            llm.run_jobs(self.engine, self.vault, self.state, [self.ev('s5', tp, 'SessionEnd')], str(self.root))
            self.assertEqual(seen, [1])
        finally:
            ent.compile_entities, llm.session_notes = orig, notes

    def test_call_log_carries_session_meta_and_j4_share_default(self):
        tp = self.transcript('s6', n=1)
        self.notes([self.ev('s6', tp, 'SessionEnd')])
        row = json.loads((self.state / 'llm-log.jsonl').read_text().splitlines()[-1])
        self.assertEqual((row['job'], row['session'], row['users']), ('j2', 's6', 1))
        self.assertEqual(queue.config(self.state)['j4_share'], 0.10)


if __name__ == '__main__':
    unittest.main()
