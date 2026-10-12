#!/usr/bin/env python3
"""E18: derived_from written where records are derived (transcript soft state, entity hard state) and a backup that restores
and still works (capture -> sync -> recall). Temp dirs and stub runners only."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (capture cues, evidence heuristics).
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base, line, llm, cap  # noqa: E402
from v3_lessons_test import OUT, EVID  # noqa: E402
from v3_package_helpers import inherited_env  # noqa: E402
import memrove_v3_entities as ent  # noqa: E402
import memrove_v3_recall as recall  # noqa: E402
import memrove_v3_trust as trust  # noqa: E402
import v3_entities_test as et  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class TranscriptSourceTest(Base):
    def session(self, sid='sA'):
        tp = self.root / f'{sid}.jsonl'
        tp.write_text(line(type='user', message={'content': 'Merhaba. ' + EVID}), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'decisions': ['Use SQLite for local state'], 'rejected': ['Redis for local state'], 'open': [],
                            'outcomes': [OUT], 'papercuts': ['doing a sync -> flaky tool call -> stale cache here']}
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': sid, 'transcript_path': str(tp), 'project': 'repo', 'at': 1}])
        return tp

    def derived(self):
        return {r['kind']: r for r in self.records().values() if r.get('derived_from')}

    def test_derived_from_written_and_pruned_is_soft(self):
        tp = self.session()
        d = self.derived()
        self.assertEqual(set(d), {'lesson', 'decision', 'rejected', 'papercut'})
        for r in d.values():
            self.assertEqual(r['derived_from'], ['transcript:' + str(tp)])
        self.assertTrue(recall.search(self.engine.store, 'Redis SQLite local state'))
        tp.unlink()  # Claude Code pruned the session
        self.assertEqual(trust.reconcile_transcripts(self.engine), 4)
        for r in (r for r in self.records().values() if r.get('derived_from')):
            self.assertEqual((r['status'], r['source_state']), ('active', 'transcript_pruned'))  # soft: still active, still recalled
        self.assertTrue(recall.search(self.engine.store, 'Redis SQLite local state'))
        self.assertEqual(trust.reconcile_transcripts(self.engine), 0)  # idempotent
        tp.write_text('back', encoding='utf-8')
        self.assertEqual(trust.reconcile_transcripts(self.engine), 4)
        self.assertEqual({r['source_state'] for r in self.records().values() if r.get('derived_from')}, {'ok'})

    def test_second_session_keeps_record_when_one_transcript_survives(self):
        a = self.session('sA')
        self.reply['j2']['decisions'] = ['Use SQLite for local state']
        b = self.root / 'sB.jsonl'
        b.write_text(line(type='user', message={'content': 'Merhaba. ' + EVID}), encoding='utf-8')
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': 'sB', 'transcript_path': str(b), 'project': 'repo', 'at': 2}])
        lesson = self.derived()['lesson']
        self.assertEqual(len(lesson['derived_from']), 2)
        a.unlink()
        trust.reconcile_transcripts(self.engine)
        self.assertNotEqual(self.derived()['lesson'].get('source_state'), 'transcript_pruned')

    def test_doc_source_deleted_is_hard_and_transcript_does_not_count_as_doc(self):
        self.engine.note_create('notes/src.md', 'doc', {'id': 'src'})
        self.engine.note_create('notes/inferred/d.md', 'from a doc', {'id': 'd', 'kind': 'lesson', 'status': 'active', 'derived_from': ['src']})
        (self.vault / 'notes/src.md').unlink()
        self.engine.sync()
        self.assertEqual(self.records()['d']['status'], 'source_missing')
        self.assertFalse([h for h in recall.search(self.engine.store, 'from a doc') if h['id'] == 'd'])
        self.assertEqual(trust.egress_of({'egress': 'external_ok', 'derived_from': ['transcript:/x']}), 'external_ok')


class EntitySourceTest(et.Base):
    def test_all_sources_gone_marks_source_missing_and_revives(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.reply['j4'] = {'sections': {'what': [f'RLS acik. [{et.A}]']}}
        self.assertEqual(ent.match(self.store, 'supabase nasil')[0][0], 'entity:supabase')
        (self.root / 'docs/supabase.md').unlink()
        et.docs.scan(self.store, self.root)
        ent.refresh_stale(self.store)
        with self.store._connect() as db:
            e = ent._load(db)['entity:supabase']
            self.assertEqual((e['status'], e['derived_from']), ('source_missing', [et.A]))
            self.assertEqual(ent.all_entities(db), [])
        self.assertEqual(ent.match(self.store, 'supabase nasil'), [])
        (self.root / 'docs/supabase.md').write_text('# Supabase again\n', encoding='utf-8')
        et.docs.scan(self.store, self.root)
        ent.refresh_stale(self.store)
        with self.store._connect() as db:
            self.assertNotIn('status', ent._load(db)['entity:supabase'])


class RestoreTest(Base):
    def cli(self, vault, state, *args):
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(vault), '--state', str(state), *args],
                           capture_output=True, text=True, encoding='utf-8', env=inherited_env())
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_backup_wipe_restore_then_real_cycle(self):
        # populated: a decision quote, a lesson with an assessment, a user-deleted note (tombstone)
        self.quote('q1', 'decision', 'Always use tabs in this repo')
        tp = self.root / 't.jsonl'
        tp.write_text(line(type='user', message={'content': EVID}), encoding='utf-8')
        self.reply['j2'] = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': [], 'outcomes': [OUT]}
        llm.session_notes(self.engine, self.vault, self.state, [{'event': 'SessionEnd', 'session': 'sA', 'transcript_path': str(tp), 'project': 'repo', 'at': 1}])
        lid = next(r['id'] for r in self.records().values() if r.get('kind') == 'lesson')
        from memrove_v3_projections import update_record
        update_record(self.engine, lid, {'assessments': {'lesson_reusable': {'state': 'assessed', 'answers': {'reusable': 'yes'}}}})
        self.quote('q-gone', 'decision', 'Ship on Fridays always please')
        (self.vault / 'notes/captured/q-gone.md').unlink()
        self.engine.sync()
        before = self.records()
        self.assertEqual(before['q-gone']['status'], 'deleted_by_user')

        dest = self.root.parent / 'bak'
        out = self.cli(self.vault, self.state, 'backup', '--dest', str(dest))
        self.assertEqual(out['status'], 'backed_up')
        self.assertEqual(out['consistency']['missing'], [])

        shutil.rmtree(self.vault)
        shutil.rmtree(self.state)
        self.vault.mkdir()
        res = self.cli(self.vault, self.state, 'restore', '--from', str(dest))
        self.assertEqual(res['status'], 'restored', res)
        self.assertEqual(res['consistency']['missing'], [])
        # restore never overwrites
        again = subprocess.run([sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(self.vault), '--state', str(self.state), 'restore', '--from', str(dest)],
                               capture_output=True, text=True, env=inherited_env())
        self.assertNotEqual(again.returncode, 0)

        from memrove_v3_sync import SyncEngine
        engine = SyncEngine(self.vault, self.state)
        with engine.store._connect() as db:
            after = {json.loads(r[0])['id']: json.loads(r[0]) for r in db.execute('SELECT payload FROM records')}
        for rid in ('q1', 'q-gone', lid):
            self.assertEqual(after[rid], before[rid], rid)  # records, tombstone and assessment survived byte for byte
        self.assertEqual(after[lid]['assessments']['lesson_reusable']['state'], 'assessed')

        # real write/read path: capture -> sync -> recall
        tp2 = self.root / 'c.jsonl'
        tp2.write_text(line(type='user', message={'content': 'Hayir, artik always use spaces in this repo, tabs yok.'}), encoding='utf-8')
        cap.run(engine, {'event': 'SessionEnd', 'session': 's9', 'transcript_path': str(tp2), 'project': 'repo'})
        engine.sync()
        with engine.store._connect() as db:
            new = [json.loads(r[0]) for r in db.execute('SELECT payload FROM records')]
        self.assertTrue([r for r in new if r.get('supersedes') == ['q1']], 'capture after restore did not write')
        self.assertEqual(next(r for r in new if r['id'] == 'q-gone')['status'], 'deleted_by_user')  # sync did not resurrect the tombstone
        self.assertTrue(recall.search(engine.store, 'Redis SQLite local state'))
        self.assertTrue(recall.search(engine.store, 'always use spaces repo'))

    def test_restore_into_other_paths_rebinds_and_cycle_works(self):
        import base64
        V, S = self.vault.resolve(), self.state.resolve()  # the runtime pins resolved paths
        self.quote('q1', 'decision', 'Always use tabs in this repo')
        self.engine.sync()
        hook = self.vault / '.claude/settings.local.json'
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(f'{{"command": "py {V}/h.py --vault {V} --state {S}"}}', encoding='utf-8')
        enc = base64.b64encode(f"& 'py' '--vault' '{V}'".encode('utf-16-le')).decode()
        (self.state / 'v3-install.json').write_text(json.dumps({'files': {'.claude/settings.local.json': {
            'installed_hash': 'x', 'installed_content': 'x'}}, 'commands': [f'py --vault {V}', 'powershell.exe -EncodedCommand ' + enc]}), encoding='utf-8')
        dest = self.root.parent / 'bak2'
        self.cli(self.vault, self.state, 'backup', '--dest', str(dest))
        vault2, state2 = self.root.parent / 'other/Vault', self.root.parent / 'other/state'
        vault2.mkdir(parents=True)
        res = self.cli(vault2, state2, 'restore', '--from', str(dest))
        self.assertEqual(res['status'], 'restored', res)
        self.assertIn('store vault_root', res['rebound'])
        self.assertIn('.claude/settings.local.json', res['rebound'])
        v2, s2 = str(vault2.resolve()), str(state2.resolve())
        self.assertIn(f'--vault {v2} --state {s2}', (vault2 / '.claude/settings.local.json').read_text(encoding='utf-8'))
        rec = json.loads((state2 / 'v3-install.json').read_text(encoding='utf-8'))
        self.assertEqual(rec['commands'][0], f'py --vault {v2}')
        self.assertIn(v2, base64.b64decode(rec['commands'][1].split()[-1]).decode('utf-16-le'))
        self.assertNotIn(str(self.vault.resolve()), json.dumps(rec['files']) + json.dumps(rec['commands']))
        file_bytes = (vault2 / '.claude/settings.local.json').read_bytes()
        self.assertEqual(base64.b64decode(rec['files']['.claude/settings.local.json']['installed_content']), file_bytes)
        # the old pair is untouched and the restored pair works through the CLI: sync -> capture -> inject -> recall -> search
        self.assertTrue(self.cli(vault2, state2, 'search', '--records', 'always use tabs repo'))
        self.cli(vault2, state2, 'sync')
        from memrove_v3_sync import SyncEngine
        engine = SyncEngine(vault2, state2)
        tp = self.root / 'c2.jsonl'
        tp.write_text(line(type='user', message={'content': 'Hayir, artik always use spaces in this repo, tabs yok.'}), encoding='utf-8')
        cap.run(engine, {'event': 'SessionEnd', 'session': 's8', 'transcript_path': str(tp), 'project': 'repo'})
        engine.sync()
        self.assertTrue(recall.search(engine.store, 'always use spaces repo'))
        self.assertTrue(self.cli(vault2, state2, 'search', '--records', 'always use spaces repo'))

    def test_search_compact_line_and_show_keep_turkish(self):
        self.quote('q1', 'decision', 'Android desteği yok, App Store merkezli')
        self.engine.sync()
        def run(*a):
            r = subprocess.run([sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(self.vault), '--state', str(self.state), *a],
                               capture_output=True, text=True, encoding='utf-8', env=inherited_env())
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout
        out = run('search', '--records', '--compact', 'android desteği').splitlines()[0]
        self.assertRegex(out, r'^#\S+ \[[^\[\]·]+ · [^\[\]·]* · (you|inferred)\] \w+: ')
        self.assertNotIn('other project', out)
        self.assertEqual(out.count('['), 1)
        shown = run('show', 'q1')
        self.assertIn('desteği', shown)
        self.assertNotIn('\\u', shown)


if __name__ == '__main__':
    unittest.main()
