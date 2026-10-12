#!/usr/bin/env python3
"""Background Haiku jobs with a stubbed runner: gates, J1/J2/J3, decay, auto-supersede."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (capture cues, evidence heuristics).
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_evaluator as evaluator  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402
import memrove_v3_capture as cap  # noqa: E402


def line(**kw):
    return json.dumps(kw, ensure_ascii=False) + '\n'


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-llm-')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.vault, self.state, self.root = root / 'vault', root / 'state', root / 'repo'
        for p in (self.vault, self.state, self.root / '.git'):
            p.mkdir(parents=True)
        self.engine = SyncEngine(self.vault, self.state)
        self.calls = []
        self.reply = {}
        llm.RUNNER = self.runner
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)
        saved = os.environ.pop('MEMROVE_NO_LLM', None)
        self.addCleanup(lambda: saved and os.environ.__setitem__('MEMROVE_NO_LLM', saved))
        os.environ.pop('MEMROVE_INVOKED_BY', None)

    def runner(self, system, payload, vault=None):
        job = next((k for k, v in evaluator.RUBRICS.items() if v['system_prompt'] == system), None) or next(k for k, v in llm.SYSTEM.items() if v == system)
        self.calls.append((job, payload))
        r = self.reply.get(job)
        return (json.dumps(r) if r is not None else None), 0.0006, (None if r is not None else 'boom')

    def records(self):
        with self.engine.store._connect() as db:
            return {json.loads(r[0])['id']: json.loads(r[0]) for r in db.execute('SELECT payload FROM records')}

    def quote(self, rid, kind, text, **extra):
        meta = dict(id=rid, kind=kind, project='repo', session='s1', verified_by='user_quote', status='active', key=cap.key_of(text),
                    polarity='neg' if kind == 'rejected' else 'pos', **extra)
        self.engine.note_create(f'notes/captured/{rid}.md', text, meta)


class CallGateTest(Base):
    def test_cap_kill_switch_and_log(self):
        self.reply['j2'] = {}
        self.assertIsNotNone(llm.call(self.state, self.vault, 'j2', '[]'))
        self.assertEqual(json.loads((self.state / 'llm-log.jsonl').read_text().splitlines()[0])['job'], 'j2')
        os.environ['MEMROVE_NO_LLM'] = '1'
        self.assertIsNone(llm.call(self.state, self.vault, 'j2', '[]'))
        del os.environ['MEMROVE_NO_LLM']
        (self.root / '.memrove').mkdir()
        (self.root / '.memrove' / 'no-llm').write_text('')
        self.assertIsNone(llm.call(self.state, self.vault, 'j2', '[]', root=self.root))
        (self.root / '.memrove' / 'no-llm').unlink()
        (self.state / 'llm-usage.json').write_text(json.dumps({'day': date.today().isoformat(), 'calls': llm.DAILY_CAP}))
        self.assertIsNone(llm.call(self.state, self.vault, 'j2', '[]'))
        self.assertEqual(len(self.calls), 1)

    def test_recursion_guard_env_and_fenced_json(self):
        os.environ['MEMROVE_INVOKED_BY'] = 'memrove-scripts'
        self.addCleanup(os.environ.pop, 'MEMROVE_INVOKED_BY', None)
        self.assertTrue(llm.disabled(self.vault))
        self.assertEqual(llm._parse('sure!\n```json\n{"a": 1}\n```'), {'a': 1})


class JobsTest(Base):
    def test_j1_drop_keep_and_gate(self):
        self.quote('q1', 'preference', 'Testleri her zaman gerçek veritabanıyla çalıştır.', durable=True, prev='mock kullanma')
        self.quote('q2', 'decision', 'Bu arada yeni sürümü kullanalım hemen şimdi.')
        self.quote('q3', 'preference', 'Commit mesajları kısa olsun ve net.')
        self.reply['j1'] = {'items': [{'id': 'q1', 'standing_rule': 'yes', 'durable': 'yes', 'scope': 'user', 'rule_name': 'real db'},
                                      {'id': 'q2', 'standing_rule': 'no', 'durable': 'no', 'scope': 'project', 'rule_name': ''},
                                      {'id': 'q3', 'standing_rule': 'yes', 'durable': 'yes', 'scope': 'user', 'rule_name': 'short commits'}]}
        self.assertEqual(llm.judge_quotes(self.engine, self.vault, self.state), 3)
        self.assertEqual(len(self.calls), 1)
        rs = self.records()
        self.assertEqual(rs['q2']['status'], 'dropped_llm')
        self.assertEqual(rs['q3']['assessments']['j1']['answers']['durable'], 'yes')
        self.assertEqual(rs['q3']['rule'], 'short commits')
        self.assertEqual(llm.judge_quotes(self.engine, self.vault, self.state), 0)  # judged once
    def test_j2_note_and_auto_admitted_inference_scope(self):
        tp = self.root / 't.jsonl'
        tp.write_text(line(type='user', message={'content': 'Postgres yerine SQLite ile devam edelim bu projede.'}), encoding='utf-8')
        self.reply['j2'] = {'did': ['schema'], 'decisions': ['Use SQLite for local state'], 'rejected': ['Postgres server'], 'open': ['add indexes']}
        ev = {'event': 'SessionEnd', 'session': 'sA', 'transcript_path': str(tp), 'project': 'repo', 'at': 1}
        self.assertEqual(llm.session_notes(self.engine, self.vault, self.state, [ev]), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(list((self.vault / 'notes/sessions').glob('*.md')))
        inf = [r for r in self.records().values() if r.get('verified_by') == 'llm']
        self.assertEqual(len(inf), 2)
        self.assertTrue(all(r['origin'] == 'llm' for r in inf))
    def test_j2_matching_user_quote_is_corroborated_not_duplicated(self):
        self.quote('q1', 'decision', 'Use SQLite for local state')
        tp = self.root / 't.jsonl'
        tp.write_text(line(type='user', message={'content': 'SQLite ile devam edelim bu projede.'}), encoding='utf-8')
        self.reply['j2'] = {'did': [], 'decisions': ['Use SQLite for local state'], 'rejected': [], 'open': []}
        ev = {'event': 'SessionEnd', 'session': 'sZ', 'transcript_path': str(tp), 'project': 'repo', 'at': 1}
        llm.session_notes(self.engine, self.vault, self.state, [ev])
        self.assertFalse((self.vault / 'notes/inferred').exists() and list((self.vault / 'notes/inferred').glob('*.md')))
        self.assertIn('sZ', self.records()['q1']['seen_sessions'])
        self.assertIn('## Decisions', next((self.vault / 'notes/sessions').glob('*.md')).read_text(encoding='utf-8'))

    def test_update_record_survives_full_sync_and_body_edit(self):
        from memrove_v3_projections import update_record
        self.quote('q2', 'decision', 'Keep tabs everywhere')
        self.assertTrue(update_record(self.engine, 'q2', {'status': 'dropped_llm', 'weight': 0.5, 'llm_verdict': 'drop'}))
        self.engine.sync(full=True)
        r = self.records()['q2']
        self.assertEqual((r['status'], r['weight'], r['llm_verdict']), ('dropped_llm', 0.5, 'drop'))
        f = self.vault / 'notes/captured/q2.md'
        f.write_text(f.read_text(encoding='utf-8') + '\nuser addition\n', encoding='utf-8')
        self.engine.sync(full=True)
        r = self.records()['q2']
        self.assertEqual((r['status'], r['weight'], r['llm_verdict']), ('dropped_llm', 0.5, 'drop'))

    def test_j3_kinds_and_topic_edges(self):
        import memrove_v3_docs as docs
        (self.root / 'docs').mkdir()
        (self.root / 'docs' / 'zzz.md').write_text('# Zzz thoughts\n\nSome words about caching layers.\n', encoding='utf-8')
        docs.scan(self.engine.store, self.root)
        doc_id = f'doc:repo/docs/zzz.md'
        self.reply['j3'] = {'items': [{'id': doc_id, 'kind': 'reference', 'topics': ['Caching', 'perf']}]}
        self.assertEqual(llm.tag_docs(self.engine, self.vault, self.state), 1)
        with self.engine.store._connect() as db:
            self.assertEqual(db.execute('SELECT kind FROM docs WHERE id=?', (doc_id,)).fetchone()[0], 'reference')
            edges = db.execute("SELECT dst,w FROM links WHERE src=? AND rel='topic' ORDER BY dst", (doc_id,)).fetchall()
        self.assertEqual(edges, [('topic:caching', 0.5), ('topic:perf', 0.5)])
        self.assertEqual(llm.tag_docs(self.engine, self.vault, self.state), 0)

    def test_failed_call_leaves_records_unjudged(self):
        self.quote('q1', 'preference', 'Hep kısa yanıt ver bana.')
        self.assertEqual(llm.judge_quotes(self.engine, self.vault, self.state), 0)
        self.assertNotIn('llm_verdict', self.records()['q1'])


class TrustTest(Base):
    def test_decay_halves_then_retires_never_deletes(self):
        old = (date.today() - timedelta(days=40)).isoformat()
        self.engine.note_create('notes/inferred/p-1.md', 'Use tabs', {'id': 'p-1', 'kind': 'decision', 'project': 'repo', 'verified_by': 'llm', 'origin': 'llm',
                                                                      'status': 'active', 'last_seen': old, 'session': 's1'})
        llm.decay(self.engine)
        r = self.records()['p-1']
        self.assertEqual((r['status'], r['weight']), ('active', 0.5))
        llm.decay(self.engine, today=date.today() + timedelta(days=30))
        r = self.records()['p-1']
        self.assertEqual((r['status'], r['weight']), ('retired', 0.0))

    def test_user_quote_auto_supersedes_opposite_polarity(self):
        self.quote('q-old', 'decision', 'Redis cache ile devam edelim sunucuda.')
        old = cap.opposed(self.engine, 'repo', 'k', 'neg', 'new', 'Yanlış, Redis cache sunucuda istemiyorum artık.')
        self.assertEqual(old, ['q-old'])
        tp = self.root / 't.jsonl'
        tp.write_text(line(type='user', message={'content': 'Yanlış, Redis cache sunucuda istemiyorum artık.'}), encoding='utf-8')
        cap.run(self.engine, {'transcript_path': str(tp), 'session': 's2', 'project': 'repo', 'event': 'SessionEnd', 'cwd': str(self.root)})
        rs = self.records()
        new = [r for r in rs.values() if r.get('polarity') == 'neg']
        self.assertTrue(new and new[0].get('supersedes') == ['q-old'], rs.keys())


class LongSessionTest(Base):
    def test_every_user_message_reaches_one_window_across_runs(self):
        tp = self.root / 'long.jsonl'
        msgs = [f'msg{i:03d} ' + 'karar ver ' * 40 for i in range(120)]  # ~400 chars each -> 48k chars, several windows
        with tp.open('w', encoding='utf-8') as f:
            for m in msgs:
                f.write(line(type='user', message={'content': m}))
                f.write(line(type='assistant', message={'content': [{'type': 'text', 'text': 'tamam ' * 300}]}))
        self.reply['j2'] = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': []}
        ev = {'event': 'SessionEnd', 'session': 'sL', 'transcript_path': str(tp), 'project': 'repo', 'at': 1}
        budget = {'left': 2, 'calls': 0, 'cost': 0.0}
        llm.session_notes(self.engine, self.vault, self.state, [ev], budget=budget)  # run 1: only 2 windows
        self.assertEqual(len(self.calls), 2)
        self.assertLess(json.loads((self.state / 'llm-cursors.json').read_text())['sL'], tp.stat().st_size)
        llm.session_notes(self.engine, self.vault, self.state, [ev])  # run 2: the rest
        self.assertEqual(llm.session_notes(self.engine, self.vault, self.state, [ev]), 0)  # run 3: nothing left, no re-send
        payloads = [p for _, p in self.calls]
        self.assertTrue(all(len(p) <= llm.MAX_J2_CHARS for p in payloads))
        for i in range(120):
            self.assertEqual(sum(f'msg{i:03d} ' in p for p in payloads), 1, i)
        self.assertEqual(len(list((self.vault / 'notes/sessions').glob('*.md'))), len(payloads))

    def test_later_run_drains_other_sessions_left_behind(self):
        tps = []
        for n in ('sX', 'sY'):
            tp = self.root / f'{n}.jsonl'
            tp.write_text(''.join(line(type='user', message={'content': f'{n} mesaj {i:02d} ' + 'karar ver ' * 40}) for i in range(60)), encoding='utf-8')
            tps.append(tp)
        self.reply['j2'] = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': []}
        evx = {'event': 'SessionEnd', 'session': 'sX', 'transcript_path': str(tps[0]), 'project': 'repo', 'at': 1}
        evy = {'event': 'SessionEnd', 'session': 'sY', 'transcript_path': str(tps[1]), 'project': 'repo', 'at': 2}
        llm.session_notes(self.engine, self.vault, self.state, [evx], budget={'left': 1, 'calls': 0, 'cost': 0.0})  # sX left partly behind
        n1 = len(self.calls)
        llm.session_notes(self.engine, self.vault, self.state, [evy])  # event for sY drains sX's rest too
        payloads = ''.join(p for _, p in self.calls)
        for i in range(60):
            self.assertEqual(payloads.count(f'sX mesaj {i:02d} '), 1)
            self.assertEqual(payloads.count(f'sY mesaj {i:02d} '), 1)
        self.assertGreater(len(self.calls), n1 + 1)

    def test_rescan_reprocesses_without_doubling_records(self):
        d = self.root / 'tr'
        d.mkdir()
        (d / 'sR.jsonl').write_text(line(type='user', message={'content': 'Postgres yerine SQLite ile devam edelim bu projede.'}, cwd=str(self.root)), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'decisions': ['Use SQLite for local state'], 'rejected': ['Postgres server'], 'open': []}
        r = llm.backfill(self.engine, self.vault, self.state, d)
        self.assertEqual(r['calls'], 1)
        before = len(self.records())
        self.assertEqual(llm.backfill(self.engine, self.vault, self.state, d)['calls'], 0)  # cursor at end: nothing
        r = llm.backfill(self.engine, self.vault, self.state, d, rescan='sR')
        self.assertEqual(r['calls'], 1)
        self.assertEqual(len(self.records()), before)  # same answers -> corroborated, not doubled
        self.reply['j2'] = dict(self.reply['j2'], decisions=['Use SQLite for local state', 'Index every foreign key column'])
        llm.backfill(self.engine, self.vault, self.state, d, rescan='all')
        self.assertEqual(len(self.records()), before + 1)  # only the genuinely new decision


if __name__ == '__main__':
    unittest.main()
