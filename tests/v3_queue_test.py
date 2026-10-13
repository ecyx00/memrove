#!/usr/bin/env python3
"""J2 queue (poison windows, locked SessionEnd, sessions without SessionEnd, cap) and sticky project resolution."""
import json
import os
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
import memrove_v3_project as project  # noqa: E402

GOOD = {'did': ['x'], 'decisions': [], 'rejected': [], 'open': []}


class QueueTest(base.Base):
    def transcript(self, name, marker=''):
        tp = self.root / f'{name}.jsonl'
        tp.write_text(base.line(type='user', message={'content': f'{name} {marker} let us make a decision'}), encoding='utf-8')
        return tp

    def ev(self, name, tp, at, event='SessionEnd'):
        return {'event': event, 'session': name, 'transcript_path': str(tp), 'project': 'repo', 'at': at}

    def runner(self, system, payload, vault=None):
        self.calls.append(('j2', payload))
        if 'POISON' in payload:
            return None, 0.0, 'boom'
        return json.dumps(GOOD), 0.0006, None

    def run_jobs(self, events):
        return llm.run_jobs(self.engine, self.vault, self.state, events, str(self.root))

    def test_poison_window_is_quarantined_and_later_sessions_run(self):
        bad, good = self.transcript('sBad', 'POISON'), self.transcript('sGood')
        evs = [self.ev('sBad', bad, 1), self.ev('sGood', good, 2)]
        llm.session_notes(self.engine, self.vault, self.state, evs)
        self.assertTrue(any('sGood' in p for _, p in self.calls), 'the session after the poison window must still run')
        st = queue.status(self.state)
        self.assertEqual((st['quarantined'], st['last_error']['where']), (0, 'j2'))
        for _ in range(queue.DEFAULT_TRIES - 1):
            llm.session_notes(self.engine, self.vault, self.state, [evs[0]])
        st = queue.status(self.state)
        self.assertEqual(st['quarantined'], 1)
        self.assertEqual(st['quarantine'][0]['session'], 'sBad')
        n = len(self.calls)
        llm.session_notes(self.engine, self.vault, self.state, [evs[0]])
        self.assertEqual(len(self.calls), n, 'a quarantined window is not retried')

    def test_session_end_while_lock_held_is_queued_not_dropped(self):
        tp = self.transcript('sLock')
        (self.state / 'llm.lock').write_text('')
        out = self.run_jobs([self.ev('sLock', tp, time.time())])
        self.assertEqual(out.get('deferred'), 'worker-running')
        self.assertEqual(self.calls, [])
        self.assertEqual(queue.status(self.state)['pending'], 1)
        (self.state / 'llm.lock').unlink()
        self.run_jobs([{'event': 'Stop', 'session': 'other', 'at': time.time()}])  # any later worker run drains it
        self.assertTrue(any('sLock' in p for _, p in self.calls))
        self.assertEqual(queue.status(self.state)['pending'], 0)

    def test_session_without_session_end_is_swept_once_idle(self):
        old, live = self.transcript('sOld'), self.transcript('sLive')
        queue.register(self.state, [self.ev('sOld', old, time.time() - queue.DEFAULT_TAIL_S - 60, 'Stop'), self.ev('sLive', live, time.time(), 'Stop')])
        self.assertEqual(queue.status(self.state)['pending'], 1)  # only the idle one is due
        self.run_jobs([{'event': 'Stop', 'session': 'sLive', 'at': time.time()}])
        joined = ''.join(p for _, p in self.calls)
        self.assertIn('sOld', joined)
        self.assertNotIn('sLive', joined)

    def test_cap_reached_is_visible_and_configurable(self):
        tp = self.transcript('sCap')
        cap = queue.config(self.state)['daily_cap']
        (self.state / 'llm-usage.json').write_text(json.dumps({'day': time.strftime('%Y-%m-%d'), 'calls': cap}))
        self.assertEqual(llm.session_notes(self.engine, self.vault, self.state, [self.ev('sCap', tp, 1)]), 0)
        st = queue.status(self.state)
        self.assertTrue(st['cap_hit_today'])
        self.assertEqual((st['cap_used'], st['cap'], st['quarantined']), (cap, cap, 0))  # capped, not poisoned: cursor stays
        (self.state / 'llm-config.json').write_text(json.dumps({'daily_cap': cap + 5}))
        self.assertEqual(queue.status(self.state)['cap'], cap + 5)
        self.assertEqual(llm.session_notes(self.engine, self.vault, self.state, [self.ev('sCap', tp, 1)]), 1)

    def test_job_exception_is_recorded(self):
        orig = llm.judge_quotes
        llm.judge_quotes = lambda *a, **k: (_ for _ in ()).throw(ValueError('nope'))
        self.addCleanup(setattr, llm, 'judge_quotes', orig)
        o3 = llm.tag_docs
        llm.tag_docs = lambda *a, **k: 0
        self.addCleanup(setattr, llm, 'tag_docs', o3)
        out = self.run_jobs([self.ev('s', self.transcript('s'), time.time())])
        self.assertEqual(out['j1'], 'ValueError')
        self.assertEqual(queue.status(self.state)['last_error']['where'], 'j1')


class StickyProjectTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory(prefix='v3-proj-')
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name).resolve()
        self.state = self.home / 'state'
        saved = os.environ.get('HOME')
        os.environ['HOME'] = str(self.home)
        self.addCleanup(lambda: os.environ.__setitem__('HOME', saved))
        self.repo = self.home / 'code' / 'app'
        (self.repo / '.git').mkdir(parents=True)
        (self.repo / 'docs').mkdir()

    def test_cwd_change_within_a_session_keeps_the_first_project(self):
        other = self.home / 'code' / 'other'
        (other / '.git').mkdir(parents=True)
        self.assertEqual(project.sticky(self.state, 's1', self.repo)[0], 'app')
        self.assertEqual(project.sticky(self.state, 's1', other)[0], 'app')
        self.assertEqual(project.sticky(self.state, 's2', other)[0], 'other')

    def test_hook_events_carry_the_sticky_project(self):
        import memrove_v3_hook as hook
        (self.state / 'hook-queue').mkdir(parents=True)
        other = self.home / 'code' / 'other'
        (other / '.git').mkdir(parents=True)
        for cwd in (self.repo, other):
            hook.enqueue_event(self.home / 'vault', self.state, {'hook_event_name': 'Stop', 'session_id': 'S', 'cwd': str(cwd), 'transcript_path': '/t.jsonl'}, 'claude')
        names = {json.loads(p.read_text())['project'] for p in (self.state / 'hook-queue').glob('*.json')}
        self.assertEqual(names, {'app'})

    def test_home_is_never_a_project(self):
        (self.home / '.git').mkdir()
        self.assertEqual(project.resolve(self.home), (None, None))
        self.assertEqual(project.project_name(self.home), 'unknown')
        self.assertEqual(project.resolve(self.home / 'code'), (None, None))  # nothing above home counts either

    def test_unmarked_subdirectory_is_not_a_project(self):
        self.assertEqual(project.project_name(self.repo / 'docs'), 'app')
        (self.home / 'plain' / 'sub').mkdir(parents=True)
        self.assertEqual(project.resolve(self.home / 'plain' / 'sub'), (None, None))
        nested = self.repo / 'vendor' / 'lib'
        (nested / '.git').mkdir(parents=True)
        self.assertEqual(project.project_name(nested), 'lib')  # nearest git root, no special case

    def test_workspace_child_is_the_project_of_an_unmarked_tree(self):
        pi = self.home / 'Desktop' / 'notes-hub'
        (pi / 'docs' / 'state').mkdir(parents=True)
        self.assertEqual(project.resolve(pi / 'docs' / 'state'), ('notes-hub', pi))
        self.assertEqual(project.resolve(pi), ('notes-hub', pi))
        self.assertEqual(project.resolve(self.home / 'Desktop'), (None, None))  # the workspace root itself is not a project
        (self.home / 'Desktop' / 'CLAUDE.md').write_text('x')
        self.assertEqual(project.resolve(self.home / 'Desktop'), (None, None))
        (pi / 'CLAUDE.md').write_text('x')  # a marker file works too
        self.assertEqual(project.project_name(pi / 'docs'), 'notes-hub')
        (pi / 'docs' / 'AGENTS.md').write_text('x')  # nearest marker wins
        self.assertEqual(project.project_name(pi / 'docs' / 'state'), 'docs')

    def test_configured_workspace_root(self):
        ws = self.home / 'work'
        (ws / 'proj' / 'sub').mkdir(parents=True)
        self.assertEqual(project.resolve(ws / 'proj' / 'sub', self.state), (None, None))
        self.state.mkdir(exist_ok=True)
        (self.state / 'project-config.json').write_text(json.dumps({'workspace_roots': [str(ws)]}))
        self.assertEqual(project.project_name(ws / 'proj' / 'sub', self.state), 'proj')

    def test_configured_root_child_beats_deeper_markers_and_root_marker(self):
        ws = self.home / 'work'
        (ws / 'a' / 'deeper').mkdir(parents=True)
        (ws / 'b').mkdir()
        (ws / 'CLAUDE.md').write_text('map')
        (ws / 'a' / 'deeper' / 'package.json').write_text('{}')
        self.state.mkdir(exist_ok=True)
        (self.state / 'project-config.json').write_text(json.dumps({'workspace_roots': [str(ws)]}))
        self.assertEqual(project.project_name(ws / 'a' / 'deeper', self.state), 'a')
        self.assertEqual(project.project_name(ws / 'b', self.state), 'b')
        self.assertEqual(project.resolve(ws, self.state), (None, None))

    def tool_transcript(self, paths, cds=()):
        tp = self.home / 'tool.jsonl'
        lines = [base.line(type='assistant', message={'content': [{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': p}}]}) for p in paths]
        lines += [base.line(type='assistant', message={'content': [{'type': 'tool_use', 'name': 'Bash', 'input': {'command': f'cd {c} && ls'}}]}) for c in cds]
        tp.write_text(''.join(lines), encoding='utf-8')
        return tp

    def test_session_from_home_takes_the_project_its_paths_point_at(self):
        tp = self.tool_transcript([str(self.repo / 'a.py'), str(self.repo / 'docs' / 'b.md'), '/tmp/x/y.txt'], cds=[str(self.repo)])
        self.assertEqual(project.infer(tp), 'app')
        self.assertEqual(project.sticky(self.state, 'h1', self.home, transcript=tp), ('app', None))
        other = self.home / 'code' / 'other'
        (other / '.git').mkdir(parents=True)
        tp2 = self.tool_transcript([str(other / 'z.py')] * 5)  # later evidence for another project does not move a sticky session
        self.assertEqual(project.sticky(self.state, 'h1', self.home, transcript=tp2)[0], 'app')

    def test_infer_is_incremental_and_bounded_per_call(self):
        tp = self.tool_transcript([str(self.repo / 'a.py')] * 2)
        old = project.CHUNK
        project.CHUNK = len(tp.read_bytes()) // 2 + 1  # a call reads about half
        self.addCleanup(setattr, project, 'CHUNK', old)
        self.assertIsNone(project.infer(tp, self.state, session='i1'))
        cur1 = json.loads((self.state / 'session-paths.json').read_text())['i1']['cursor']
        self.assertLess(cur1, tp.stat().st_size)  # bounded: only whole lines of the first chunk
        with open(tp, 'a') as f:  # more evidence arrives; later calls catch up and only read new bytes
            f.write(base.line(type='assistant', message={'content': [{'type': 'tool_use', 'name': 'Read', 'input': {'file_path': str(self.repo / 'z.py')}}]}))
        names = [project.infer(tp, self.state, session='i1') for _ in range(4)]
        self.assertEqual(names[-1], 'app')
        self.assertEqual(json.loads((self.state / 'session-paths.json').read_text())['i1']['cursor'], tp.stat().st_size)

    def test_session_from_home_without_majority_stays_recallable_user_scope(self):
        other = self.home / 'code' / 'other'
        (other / '.git').mkdir(parents=True)
        tp = self.tool_transcript([str(self.repo / 'a.py'), str(other / 'b.py'), str(self.repo / 'c.py'), str(other / 'd.py')])
        self.assertIsNone(project.infer(tp))  # 2 / 2: no clear majority
        self.assertIsNone(project.infer(self.tool_transcript([str(self.repo / 'a.py')] * 2)))  # under 3 hits
        import memrove_v3_hook as hook
        (self.state / 'hook-queue').mkdir(parents=True)
        hook.enqueue_event(self.home / 'vault', self.state, {'hook_event_name': 'Stop', 'session_id': 'P', 'cwd': str(self.home), 'transcript_path': str(tp)}, 'claude')
        self.assertEqual({json.loads(x.read_text())['project'] for x in (self.state / 'hook-queue').glob('*.json')}, {'unknown'})
        import memrove_v3_kinds as kinds
        scope = kinds.effective_scope({'kind': 'decision', 'project': 'unknown'})
        self.assertEqual(scope, 'user')
        self.assertTrue(kinds.in_scope(scope, kinds.client_scopes(self.state)))  # not filtered out of claude_code recall
        self.assertNotIn('personal', kinds.client_scopes(self.state))

    def test_vault_is_not_a_code_project(self):
        vault = self.home / 'vault'
        (vault / '.git').mkdir(parents=True)
        (vault / 'sub').mkdir()
        self.assertEqual(project.resolve(vault / 'sub', vault=vault), (None, None))

    def test_relabel_dry_run_lists_changes_and_writes_nothing(self):
        from memrove_v3_sync import SyncEngine
        vault = self.home / 'vault'
        vault.mkdir()
        engine = SyncEngine(vault, self.state)
        tp = self.home / 't.jsonl'
        tp.write_text(base.line(type='user', cwd=str(self.repo / 'docs'), message={'content': 'x'}), encoding='utf-8')
        (self.state / 'session-map.json').write_text(json.dumps({'h1': {'raw': 'r', 'path': str(tp)}}))
        engine.note_create('notes/captured/r1.md', 'a note', dict(id='r1', kind='note', project='docs', session='h1'))
        engine.note_create('notes/captured/r2.md', 'baska not', dict(id='r2', kind='note', project='docs', session='gone'))
        before = (self.state / 'memory.sqlite3').read_bytes()
        plan = project.relabel_plan(self.state)
        self.assertEqual(plan['changes'], {'docs -> app': 1})
        self.assertEqual(plan['unresolvable'], 1)
        self.assertEqual((self.state / 'memory.sqlite3').read_bytes(), before)

    def test_relabel_apply_is_idempotent_and_undoable(self):
        from memrove_v3_sync import SyncEngine
        vault = self.home / 'vault'
        vault.mkdir()
        engine = SyncEngine(vault, self.state)
        tp = self.home / 't.jsonl'
        tp.write_text(base.line(type='user', cwd=str(self.repo / 'docs'), message={'content': 'x'}), encoding='utf-8')
        (self.state / 'session-map.json').write_text(json.dumps({'h1': {'raw': 'r', 'path': str(tp)}}))
        engine.note_create('notes/captured/r1.md', 'a note', dict(id='r1', kind='note', project='docs', session='h1'))
        engine.note_create('notes/captured/r2.md', 'baska not', dict(id='r2', kind='note', project='docs', session='gone'))
        engine.note_create('notes/captured/r3.md', 'real', dict(id='r3', kind='note', project='other', session='gone'))
        out = project.relabel_apply(engine, self.state, vault)
        self.assertEqual(out['moved_records'], 2)
        self.assertEqual(project.relabel_apply(engine, self.state, vault)['moved_records'], 0)
        with engine.store._connect() as db:
            rows = {i: json.loads(p) for i, p in db.execute('SELECT id,payload FROM records')}
        self.assertEqual([(rows[i]['project'], rows[i].get('relabeled_from')) for i in ('r1', 'r2', 'r3')],
                         [('app', 'docs'), ('app', 'docs'), ('other', None)])
        self.assertEqual(project.relabel_apply(engine, self.state, vault, undo=True, only='nope')['undone_records'], 0)
        self.assertEqual(project.relabel_apply(engine, self.state, vault, undo=True, only='docs')['undone_records'], 2)
        with engine.store._connect() as db:
            self.assertEqual({json.loads(p)['project'] for i, p in db.execute('SELECT id,payload FROM records')}, {'docs', 'other'})


if __name__ == '__main__':
    unittest.main()
