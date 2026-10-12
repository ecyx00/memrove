#!/usr/bin/env python3
"""E16: papercuts from transcripts (retry, noise filter, subagents), J2 papercuts, dedupe across sessions, projection. Stub RUNNER only."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, line, llm  # noqa: E402
import memrove_v3_papercuts as pc  # noqa: E402
from memrove_v3_projections import project_graph  # noqa: E402


def use(i, name, **inp):
    return line(type='assistant', message={'content': [{'type': 'tool_use', 'id': i, 'name': name, 'input': inp}]})


def res(i, text, err=True):
    return line(type='user', message={'content': [{'type': 'tool_result', 'tool_use_id': i, 'content': text, 'is_error': err}]})


MAIN = (use('a', 'Edit', file_path='/x/y.py') + res('a', '<tool_use_error>File has not been read yet. Read it first.</tool_use_error>')
        + use('b', 'Edit', file_path='/x/y.py') + res('b', 'ok', False)
        + use('c', 'Bash', command='python3 -m unittest') + res('c', 'Exit code 1\nFAILED (failures=1)')
        + use('d', 'Bash', command='grep foo bar.txt') + res('d', 'Exit code 1')
        + use('e', 'Bash', command='ls') + res('e', "The user doesn't want to proceed with this tool use.")
        + use('f', 'Bash', command='python3 s.py') + res('f', 'Exit code 1\nTraceback (most recent call last):\n  File "/tmp/s.py", line 3\nValueError: bad 42'))
SUB = use('g', 'Bash', command='node x.js') + res('g', 'Exit code 127\nnode: command not found') + use('h', 'Read', file_path='/a') + res('h', 'x', False)


class PapercutTest(Base):
    def session(self, sid, name):
        tp = self.root / f'{name}.jsonl'
        tp.write_text(line(type='user', message={'content': 'hi'}) + MAIN, encoding='utf-8')
        sub = self.root / name / 'subagents'
        sub.mkdir(parents=True, exist_ok=True)
        (sub / 'agent-abc.jsonl').write_text(SUB, encoding='utf-8')
        return {'event': 'SessionEnd', 'session': sid, 'transcript_path': str(tp), 'project': 'repo', 'at': 1}

    def cuts(self):
        return [r for r in self.records().values() if r.get('kind') == 'papercut']

    def test_extract_retry_and_noise(self):
        ev = self.session('s1', 't1')
        got = {(e['tool'], e['source']): e for e in pc.extract(ev['transcript_path'])}
        self.assertEqual(set(got), {('Edit', 'main'), ('Bash', 'main'), ('Bash', 'abc')})  # test failure, grep miss, declined prompt dropped
        self.assertTrue(got[('Edit', 'main')]['retry'] and got[('Edit', 'main')]['harness'])
        self.assertEqual(got[('Bash', 'main')]['excerpt'], 'ValueError: bad 42')
        self.assertFalse(got[('Bash', 'abc')]['retry'])

    def test_dedupe_across_sessions_and_projection(self):
        for sid, name in (('s1', 't1'), ('s2', 't2')):
            pc.collect(self.engine, [self.session(sid, name)])
        pc.collect(self.engine, [self.session('s2', 't2')])  # same session again: no bump
        cuts = self.cuts()
        self.assertEqual(len(cuts), 3)
        self.assertTrue(all(r['count'] == 2 and r['scope'] == 'technical' for r in cuts))
        self.assertEqual(sum(bool(r.get('harness')) for r in cuts), 1)
        self.assertFalse(list(self.vault.glob('notes/**/pc-*.md')))
        self.engine.sync()
        self.assertEqual(len(self.cuts()), 3)  # a sync leaves store-only records alone
        project_graph(self.engine, budget_s=30)
        proj = (self.vault / 'memrove/papercuts/repo.md').read_text(encoding='utf-8')
        har = (self.vault / 'memrove/papercuts/harness.md').read_text(encoding='utf-8')
        self.assertIn('2 sessions, 2 hits [transcript] Bash: ValueError: bad 42', proj)
        self.assertIn('2 sessions, 2 hits [retry, transcript] Edit: File has not been read yet', har)

    def test_j2_papercuts_weigh_more_and_run_without_llm(self):
        tp = self.root / 'j.jsonl'
        tp.write_text(line(type='user', message={'content': 'merhaba nasilsin'}), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'papercuts': ['Running tests -> pytest not found -> venv not active', 'short']}
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': 's3', 'transcript_path': str(tp), 'project': 'repo', 'at': 1}])
        (c,) = self.cuts()
        self.assertEqual((c['rank_weight'], c['cause'], c['excerpt']), (2.0, 'venv not active', 'pytest not found'))
        self.assertFalse(list(self.vault.glob('notes/**/pc-*.md')))  # store-only: no per-record note
        import os
        os.environ['MEMROVE_NO_LLM'] = '1'
        self.addCleanup(os.environ.pop, 'MEMROVE_NO_LLM', None)
        out = llm.run_jobs(self.engine, self.vault, self.state, [self.session('s4', 't4')])
        self.assertEqual(out, {'papercuts': 3})


if __name__ == '__main__':
    unittest.main()
