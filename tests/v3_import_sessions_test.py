#!/usr/bin/env python3
"""import-sessions: injected context dropped, project by cwd, dedupe across transcript files, idempotent rerun, dry-run writes nothing, older-only supersede."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import os as _os
_os.environ.setdefault("MEMROVE_NO_LLM", "1")
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True


def line(**kw):
    return json.dumps(kw, ensure_ascii=False) + '\n'


def umsg(text, ts='2026-05-02T10:00:00Z'):
    return line(type='response_item', timestamp=ts, payload={'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]})


class ImportSessionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v3-import-sessions-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.home, self.vault, self.state = root / 'home', root / 'vault', root / 'state'
        self.repo = self.home / 'Desktop' / 'proj'
        for p in (self.vault, self.repo / '.git', self.home / '.codex' / 'sessions' / '2026'):
            p.mkdir(parents=True)
        self.old_home = _os.environ.get('HOME')
        _os.environ['HOME'] = str(self.home)
        self.addCleanup(lambda: _os.environ.__setitem__('HOME', self.old_home))

    def codex(self):
        rec = line(type='session_meta', payload={'id': 'sess-1', 'cwd': str(self.repo), 'timestamp': '2026-05-02T09:00:00Z', 'source': 'cli'})
        rec += umsg('<skill> <name>x</name> Bundan sonra her zaman skill metni.</skill>')
        rec += umsg('<environment_context> <cwd>/x</cwd> Always ignore this. </environment_context>')
        rec += umsg('Bundan sonra testleri her zaman önce çalıştır.')
        rec += line(type='response_item', payload={'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Bundan sonra her zaman model metni.'}]})
        (self.home / '.codex' / 'sessions' / '2026' / 'rollout-a.jsonl').write_text(rec, encoding='utf-8')
        sub = line(type='session_meta', payload={'id': 'sub-1', 'cwd': str(self.repo), 'source': {'subagent': {}}, 'thread_source': 'subagent'})
        sub += umsg('Bundan sonra her zaman subagent mesajı.')
        (self.home / '.codex' / 'sessions' / '2026' / 'rollout-b.jsonl').write_text(sub, encoding='utf-8')

    def test_codex_dry_run_then_import_idempotent(self):
        import memrove_v3_sessions as S
        self.codex()
        r = S.run('codex', self.vault, self.state, dry_run=True, home=self.home)
        self.assertEqual((r['sessions'], r['user_turns'], r['records']), (1, 1, 1))
        self.assertEqual(r['by_project'], {'proj': 1})
        self.assertEqual(r['j_jobs']['calls_made_by_import'], 0)
        self.assertFalse(self.state.exists(), 'dry-run must not create state')
        first = S.run('codex', self.vault, self.state, home=self.home)
        self.assertEqual((first['created'], first['already_imported']), (1, 0))
        again = S.run('codex', self.vault, self.state, home=self.home)
        self.assertEqual((again['created'], again['already_imported']), (0, 1))
        notes = list((self.vault / 'notes' / 'captured').glob('q-*.md'))
        self.assertEqual(len(notes), 1)
        text = notes[0].read_text(encoding='utf-8')
        self.assertIn('imported_from', text)
        self.assertIn('2026-05-02', text)

    def test_antigravity_dedupes_step_index_and_unwraps(self):
        import memrove_v3_sessions as S
        logs = self.home / '.gemini' / 'antigravity' / 'brain' / 'abc' / '.system_generated' / 'logs'
        logs.mkdir(parents=True)
        step = line(step_index=0, source='USER_EXPLICIT', type='USER_INPUT', created_at='2026-06-19T21:07:38Z',
                    content='<USER_REQUEST>\nBundan sonra her zaman önce plan yap.\n</USER_REQUEST>\n<ADDITIONAL_METADATA>always x</ADDITIONAL_METADATA>')
        model = line(step_index=1, source='MODEL', type='PLANNER_RESPONSE', content='Bundan sonra her zaman model.')
        (logs / 'transcript.jsonl').write_text(step + model, encoding='utf-8')
        (logs / 'transcript_full.jsonl').write_text(step, encoding='utf-8')
        r = S.run('antigravity', self.vault, self.state, dry_run=True, home=self.home)
        self.assertEqual((r['sessions'], r['user_turns'], r['records']), (1, 1, 1))
        self.assertEqual(r['by_project'], {'unknown': 1})

    def test_id_and_turn_cleaning(self):
        import memrove_v3_sessions as S
        q = {'project': 'p', 'type': 'preference', 'quote': 'x'}
        self.assertTrue(S.rid_of(q).startswith('q-'))
        self.assertEqual(S.rid_of(q), S.rid_of(dict(q)))
        self.assertIsNone(S.clean_turn('<turn_aborted> The user interrupted'))
        self.assertEqual(S.clean_turn('# Files mentioned by the user: a\n\n## My request for Codex:\nYap bunu.'), 'Yap bunu.')

    def test_pasted_prompts_dropped(self):
        import memrove_v3_sessions as S
        self.assertEqual(S.pasted_reason('You are a senior engineer. Always write tests.'), 'role-opener')
        self.assertEqual(S.pasted_reason('Rules:\n- do not touch a\n- never touch b\n- always c'), 'markdown-structure')
        self.assertEqual(S.pasted_reason('\n'.join('satır %d' % i for i in range(20))), 'many-lines')
        self.assertIsNone(S.pasted_reason('Bundan sonra testleri her zaman önce çalıştır.'))
        self.assertTrue(S.english_paste('Always start by requesting the project context from the manager.'))
        self.assertFalse(S.english_paste('Her zaman AGENTS.md lere uy.'))

    def test_imports_are_judged_last(self):
        import memrove_v3_llm as llm
        from memrove_v3_sync import SyncEngine
        engine = SyncEngine(self.vault, self.state)
        for rid, extra in (('q-live', {}), ('q-imp', {'imported_from': 'codex-session'})):
            engine.note_create(f'notes/captured/{rid}.md', 'Bundan sonra her zaman ' + rid,
                               dict({'id': rid, 'kind': 'preference', 'project': 'p', 'verified_by': 'user_quote', 'status': 'active'}, **extra))
        seen = []
        old = llm._j1_batch
        llm._j1_batch = lambda engine, vault, state, root, rows: (seen.extend(r['id'] for r in rows) or ([], 0))
        self.addCleanup(lambda: setattr(llm, '_j1_batch', old))
        llm.MAX_J1, keep = 1, llm.MAX_J1
        self.addCleanup(lambda: setattr(llm, 'MAX_J1', keep))
        llm.judge_quotes(engine, self.vault, self.state)
        self.assertEqual(seen, ['q-live'])


if __name__ == '__main__':
    unittest.main()
