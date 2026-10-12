#!/usr/bin/env python3
"""End to end on a temp HOME: global install, hooks for one real-transcript session, vault projection, doctor.

Runs with a stubbed Haiku runner. MEMROVE_E2E_REAL=1 uses the real CLI instead (a few real calls).
MEMROVE_E2E_TRANSCRIPT overrides the transcript (default: a mid-size main session under ~/.claude/projects).
"""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
REAL = os.environ.get('MEMROVE_E2E_REAL') == '1'
SOURCE = Path(os.environ.get('MEMROVE_E2E_TRANSCRIPT') or Path.home() / '.claude/projects/-Users-dev/session.jsonl')

EXTRA = [
    'Bundan sonra testleri her zaman gerçek veritabanıyla çalıştır, mock kullanma.',
    'Karar verdim: kalıcı durum için SQLite kullanılacak, Postgres sunucusu istemiyorum.',
    'Commit mesajları her zaman kısa ve net olsun.',
]
STUB = {
    'j1': lambda items: {'items': [{'id': i['id'], 'standing_rule': 'yes', 'durable': 'yes', 'scope': 'user', 'rule_name': 'rule ' + i['id'][-4:]} for i in items]},
    'j2': lambda _: {'did': ['wired capture'], 'decisions': ['Keep state in SQLite'], 'rejected': ['Run a Postgres server'], 'open': ['write docs']},
    'j3': lambda items: {'items': [{'id': i['id'], 'kind': 'reference', 'topics': ['memory', 'notes']} for i in items]},
}


@unittest.skipUnless(SOURCE.is_file(), 'no local transcript to copy')
class E2ETest(unittest.TestCase):
    def test_flow(self):
        import memrove_v3_llm as llm
        with tempfile.TemporaryDirectory(prefix='v3-e2e-') as tmp:
            home = Path(tmp)
            vault, state, repo = home / 'vault', home / 'state', home / 'repo'
            (repo / '.git').mkdir(parents=True)
            (repo / 'docs').mkdir()
            (repo / 'docs' / 'architecture.md').write_text('# Architecture\n\nOverview of src/app layout.\n', encoding='utf-8')
            (repo / 'docs' / 'problem-login.md').write_text('# Problem: login fails\n\nUsers cannot sign in.\n', encoding='utf-8')
            (repo / 'docs' / 'brainstorm.md').write_text('# Brainstorm\n\nLoose ideas about caching.\n', encoding='utf-8')
            shutil.copytree(ROOT / 'template/.claude', vault / '.claude', ignore=shutil.ignore_patterns('.state', '__pycache__'))
            tp = home / '.claude/projects/-tmp-repo/sess-e2e.jsonl'
            tp.parent.mkdir(parents=True)
            shutil.copy(SOURCE, tp)
            with tp.open('a', encoding='utf-8') as f:
                for t in EXTRA:
                    f.write(json.dumps({'type': 'user', 'message': {'content': t}}, ensure_ascii=False) + '\n')
            before = SOURCE.stat().st_mtime
            saved = os.environ.pop('MEMROVE_NO_LLM', None)
            self.addCleanup(lambda: saved and os.environ.__setitem__('MEMROVE_NO_LLM', saved))
            env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), MEMROVE_V3_NO_SPAWN='1')
            for k in ('CLAUDE_CODE_ENTRYPOINT', 'MEMROVE_NO_LLM', 'MEMROVE_INVOKED_BY'):
                env.pop(k, None)
            r = subprocess.run([sys.executable, str(ROOT / 'scripts/install_v3.py'), '--vault', str(vault), '--state', str(state), '--global'],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            hooks = json.loads((home / '.claude/settings.json').read_text())['hooks']
            self.assertEqual(sorted(hooks), ['PreCompact', 'SessionEnd', 'SessionStart', 'Stop', 'UserPromptSubmit'])

            def fire(event, **extra):
                payload = dict(hook_event_name=event, session_id='sess-e2e', transcript_path=str(tp), cwd=str(repo), **extra)
                cmd = hooks[event][-1]['hooks'][0]['command']
                p = subprocess.run(cmd, shell=True, input=json.dumps(payload), text=True, capture_output=True, env=env, cwd=repo, timeout=30)
                self.assertEqual(p.returncode, 0, p.stderr)
                return json.loads(p.stdout or '{}')

            calls = []
            if not REAL:
                def runner(system, payload, vault=None):
                    import memrove_v3_evaluator as evaluator
                    job = 'j1' if system == evaluator.RUBRICS['j1']['system_prompt'] else next(k for k, v in llm.SYSTEM.items() if v == system)
                    calls.append(job)
                    arg = json.loads(payload) if job != 'j2' else payload
                    return json.dumps(STUB[job](arg)), 0.0006, None
                llm.RUNNER = runner
                self.addCleanup(setattr, llm, 'RUNNER', llm._claude)
            from memrove_v3_hook import drain_queue

            fire('SessionStart', source='startup')
            fire('Stop')
            fire('PreCompact')
            drain_queue(vault, state)
            out = fire('SessionStart', source='compact')
            block = out['hookSpecificOutput']['additionalContext']
            self.assertLessEqual(len(block) // 4, 1200)
            self.assertIn('gerçek veritabanıyla', block)
            self.assertIn('[you]', block)
            fire('SessionEnd')
            drain_queue(vault, state)

            b = vault / 'memrove'
            for sub in ('decisions', 'preferences', 'sessions', 'docs/repo'):
                self.assertTrue(list((b / sub).glob('*.md')), sub)
            hub = (b / 'projects/repo.md').read_text(encoding='utf-8')
            self.assertIn('[[memrove/preferences/', hub)
            self.assertIn('Docs:', hub)
            self.assertTrue(list((vault / 'notes/sessions').glob('*.md')))
            import graf_kontrol
            generated = graf_kontrol.tara(vault)
            orphans = [str(o) for o in generated[2] if str(o).startswith(('memrove/', 'notes/sessions/'))]
            self.assertEqual(orphans, [], orphans)
            d = subprocess.run([sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(vault), '--state', str(state), 'doctor'],
                               capture_output=True, text=True, env=env)
            doc = json.loads(d.stdout)
            self.assertGreaterEqual(doc['llm']['calls_total'], 3 if REAL else 1)
            self.assertEqual(doc['docs']['docs'], 3)
            self.assertIn('orphan_pct', doc['graph'])
            import hashlib
            import memrove_v3_docs
            from memrove_v3_sync import SyncEngine
            self.assertIsNotNone(memrove_v3_docs._find_transcript(hashlib.sha256(b'sess-e2e').hexdigest()[:24], SyncEngine(vault, state).store))
            self.assertEqual(SOURCE.stat().st_mtime, before)
            if not REAL:
                self.assertEqual(sorted(set(calls)), ['j1', 'j2', 'j3'])
            else:
                print('\nREAL RUN doctor.llm =', json.dumps(doc['llm']))
                print('REAL RUN block:\n' + block)
                print('REAL RUN notes:', [p.read_text(encoding='utf-8') for p in (vault / 'notes/sessions').glob('*.md')])
                print('REAL RUN j1:', [(r.name, r.read_text()[-0:]) for r in (vault / 'notes/captured').glob('*.md')][:0])


if __name__ == '__main__':
    unittest.main()
