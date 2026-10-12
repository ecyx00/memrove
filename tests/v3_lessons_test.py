#!/usr/bin/env python3
"""E10: schema alignment, J2 outcomes -> lessons (evidence gate, corroboration), backfill CLI core. Stub RUNNER only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, line, llm, cap  # noqa: E402

# Turkish on purpose: the evidence quote must pass the Turkish evidence heuristics.
EVID = 'Redis cache yerine SQLite kullandik ve sorun cikmadi'
OUT = {'situation': 'Choosing a local state store', 'approach': 'SQLite instead of Redis', 'result': 'worked', 'why': 'No extra server needed',
       'reuse': 'Local state rarely needs a server.', 'evidence': EVID, 'topics': ['storage'], 'entities': ['SQLite']}


class LessonTest(Base):
    def transcript(self, name='t.jsonl', text='Hello. ' + EVID + ' in this project.'):
        tp = self.root / name
        tp.write_text(line(type='user', message={'content': text}), encoding='utf-8')
        return tp

    def run_j2(self, sid, outcomes, tp=None):
        tp = tp or self.transcript()
        self.reply['j2'] = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': [], 'outcomes': outcomes}
        ev = {'event': 'SessionEnd', 'session': sid, 'transcript_path': str(tp), 'project': 'repo', 'at': 1}
        llm.session_notes(self.engine, self.vault, self.state, [ev])

    def lessons(self):
        return [r for r in self.records().values() if r.get('kind') == 'lesson']

    def test_schema_mapping_and_supersede_valid_to(self):
        self.assertEqual(cap.origin_type_of({'verified_by': 'user_quote'}), 'user_explicit')
        self.assertEqual(cap.origin_type_of({'origin': 'llm'}), 'agent_inference')
        self.assertEqual(cap.origin_type_of({'source': 'docs/a.md'}), 'derived')
        self.assertEqual(cap.origin_type_of({'origin_type': 'external_source', 'origin': 'llm'}), 'external_source')
        self.quote('q-old', 'decision', 'Always use tabs in this repo')
        tp = self.transcript('c.jsonl', 'Hayir, artik always use spaces in this repo, tabs yok.')
        cap.run(self.engine, {'event': 'SessionEnd', 'session': 's9', 'transcript_path': str(tp), 'project': 'repo'})
        rs = self.records()
        new = [r for r in rs.values() if r.get('supersedes') == ['q-old']]
        self.assertTrue(new, rs.keys())
        self.assertEqual(new[0]['origin_type'], 'user_explicit')
        self.assertTrue(new[0]['valid_from'])
        self.assertEqual(rs['q-old']['valid_to'], new[0]['valid_from'])

    def test_fake_evidence_dropped_and_logged(self):
        self.run_j2('sA', [dict(OUT, evidence='bu cumle transcriptte yok kesinlikle')])
        self.assertEqual(self.lessons(), [])
        self.assertIn('evidence-not-in-transcript', (self.state / 'llm-drops.jsonl').read_text(encoding='utf-8'))

    def test_real_evidence_admitted_with_fields(self):
        self.run_j2('sA', [dict(OUT, evidence='Redis cache yerine   SQLite\nkullandik')])  # whitespace-normalised match
        (r,) = self.lessons()
        self.assertEqual((r['origin_type'], r['confidence'], r['verified_by'], r['result']), ('agent_inference', 0.5, 'llm', 'worked'))
        self.assertEqual((r['project'], r['session'], r['model_version']), ('repo', 'sA', 'haiku'))
        self.assertTrue(r['valid_from'] and r['extractor_version'])
        self.assertTrue(r['source'].startswith('notes/inferred/'))
        self.assertIn('inferred', (self.vault / r['source']).read_text(encoding='utf-8'))
        self.assertIn('## Lessons', next((self.vault / 'notes/sessions').glob('*.md')).read_text(encoding='utf-8'))

    def test_duplicate_lesson_corroborates(self):
        self.run_j2('sA', [OUT])
        tp2 = self.transcript('t2.jsonl', 'Again: ' + EVID + ' again.')
        self.run_j2('sB', [dict(OUT, why='Another rationale')], tp2)
        (r,) = self.lessons()
        self.assertEqual(r['seen_sessions'], ['sA', 'sB'])
        self.assertEqual(r['confidence'], 0.6)
        for i in range(10):  # cap 0.9
            self.run_j2('sC%d' % i, [OUT], self.transcript('t%d.jsonl' % (i + 3)))
        self.assertEqual(self.lessons()[0]['confidence'], 0.9)


class BackfillTest(Base):
    def setUp(self):
        super().setUp()
        self.tdir = self.root / 'transcripts'
        self.tdir.mkdir()
        for i in range(3):
            p = self.tdir / f's{i}.jsonl'
            p.write_text(line(type='user', message={'content': f'Oturum {i}: SQLite ile devam edelim.'}, cwd='/x/proj'), encoding='utf-8')
            os.utime(p, (1_700_000_000 + i * 86400, 1_700_000_000 + i * 86400))
        self.reply['j2'] = {'did': ['x'], 'decisions': ['Use SQLite'], 'rejected': [], 'open': [], 'outcomes': []}

    def bf(self, **kw):
        return llm.backfill(self.engine, self.vault, self.state, self.tdir, **kw)

    def test_recursive_depth2_skips_subagents(self):
        proj = self.tdir / 'proj-a'
        (proj / 'subagents').mkdir(parents=True)
        (self.tdir / 'deep' / 'x').mkdir(parents=True)
        for p in (proj / 's9.jsonl', proj / 'subagents' / 'agent-1.jsonl', self.tdir / 'deep' / 'x' / 'd.jsonl'):
            p.write_text(line(type='user', message={'content': 'Oturum: SQLite ile devam.'}, cwd='/x/proj'), encoding='utf-8')
        self.assertEqual(self.bf(dry_run=True)['pending'], 4)  # 3 top-level + proj/s9; not subagents, not depth 3

    def test_dry_run_calls_nothing(self):
        r = self.bf(dry_run=True)
        self.assertEqual((r['pending'], r['would_call']), (3, 3))
        self.assertGreater(r['est_cost_usd'], 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.bf(dry_run=True, since='2099-01-01')['pending'], 0)

    def test_max_calls_oldest_first_and_resumes(self):
        r = self.bf(max_calls=2)
        self.assertEqual((r['calls'], r['windows'], r['remaining']), (2, 2, 1))
        self.assertGreater(r['records_created'], 0)
        self.assertAlmostEqual(r['cost_usd'], 0.0012)
        self.assertFalse((self.state / 'llm-usage.json').exists())  # daily cap untouched
        self.assertEqual(sorted(json.loads((self.state / 'llm-cursors.json').read_text())), ['s0', 's1'])
        self.assertEqual(self.bf(max_calls=5)['calls'], 1)  # s2 only
        self.assertEqual(self.bf(dry_run=True)['pending'], 0)

    def test_cli_dry_run(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/memrove_v3.py'
        out = subprocess.run([sys.executable, str(script), '--vault', str(self.vault), '--state', str(self.state), 'backfill',
                              '--transcripts', str(self.tdir), '--dry-run'], capture_output=True, text=True, env=dict(os.environ, MEMROVE_NO_LLM='1'))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)['would_call'], 3)


if __name__ == '__main__':
    unittest.main()
