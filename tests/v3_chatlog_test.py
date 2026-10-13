#!/usr/bin/env python3
"""Chat memory core: push-order equivalence, log, ingest, tree, view sawtooth + persistence, pagination, batch parser, compactor with a stub runner."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
from datetime import datetime, timedelta
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
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chattree as ct  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402


def push(new, states):
    """Taelin's rollback push (rollback_state_list.js), life fixed at 0 as under push alone."""
    if states is None:
        return {'keep': 0, 'life': 0, 'state': new, 'older': None}
    if states['keep'] == 0:
        return dict(states, keep=1)
    return {'keep': 0, 'life': states['life'], 'state': new, 'older': push(states['state'], states['older'])}


def starts(states):
    out = []
    while states:
        out.append(states['state'])
        states = states['older']
    return out


class PushOrder(unittest.TestCase):
    def test_merge_order_equals_taelin_push_for_20000_ticks(self):
        parts, states, bad = [], None, []
        for t in range(20001):
            states = push(t, states)
            want = starts(states)
            parts.append((0, t))
            while len(parts) > len(want):
                k = ct.most_due(parts, t + 1, lambda l, i: True)
                self.assertIsNotNone(k, t)
                parts[k:k + 2] = [(parts[k][0] + 1, parts[k][1] >> 1)]
            if [i << l for l, i in reversed(parts)] != want:
                bad.append(t)
        self.assertEqual(bad, [])

    def test_first_message_age_is_not_the_order(self):
        # the spec's counter-example: view 0+4, 4+4, 8+1, 9+1 at T=10 merges 8+1 and 9+1, not 0+4 and 4+4
        parts = [(2, 0), (2, 1), (0, 8), (0, 9)]
        k = ct.most_due(parts, 10, lambda l, i: True)
        self.assertEqual(parts[k], (0, 8))


def fake_home(tmp):
    home = Path(tmp) / 'home'
    proj = home / '.claude/projects/-x'
    proj.mkdir(parents=True)
    t0 = datetime(2026, 10, 7, 10, 0, 0).astimezone()

    def at(m):
        return (t0 + timedelta(minutes=m)).isoformat()

    def rec(kind, m, content, **kw):
        return json.dumps(dict({'type': kind, 'uuid': f'u{m}{kind}', 'timestamp': at(m), 'message': {'role': kind, 'content': content}}, **kw))
    claude = [
        rec('user', 0, 'merhaba, bunu yap'),
        rec('assistant', 1, [{'type': 'thinking', 'thinking': 'secret thoughts'}, {'type': 'text', 'text': 'OK, doing it.'}]),
        rec('assistant', 2, [{'type': 'tool_use', 'name': 'Bash', 'input': {}}]),
        rec('user', 3, [{'type': 'tool_result', 'content': 'x'}]),
        rec('user', 4, 'the quick brown fox for the lazy dog and the cat to the end'),  # English, never filtered
        rec('user', 5, 'meta', isMeta=True),
        rec('user', 6, 'side', isSidechain=True),
        rec('user', 7, '<system-reminder>x</system-reminder>'),
        rec('user', 8, '<command-name>/foo</command-name><command-args>bar baz</command-args>'),
        rec('user', 9, 'key sk-' + 'a' * 30 + ' ok'),
        json.dumps({'type': 'attachment', 'uuid': 'att1', 'timestamp': at(10), 'attachment': {'type': 'queued_command', 'prompt': 'mid-turn word', 'delivery_id': 'd1', 'timestamp': at(10), 'origin': {'kind': 'human'}}}),
        json.dumps({'type': 'attachment', 'uuid': 'att2', 'timestamp': at(11), 'attachment': {'type': 'queued_command', 'prompt': 'task done', 'delivery_id': 'd2', 'origin': {'kind': 'task-notification'}}}),
        rec('user', 12, 'P' * 1500),
        rec('user', 13, 'P' * 1500),  # exact repeat of a long paste
    ]
    (proj / 's1.jsonl').write_text('\n'.join(claude) + '\n', encoding='utf-8')
    codex = home / '.codex/sessions/2026/10/07'
    codex.mkdir(parents=True)

    def cx(m, role, typ, text, ts=None):
        return json.dumps({'timestamp': ts or at(m), 'type': 'response_item', 'payload': {'type': 'message', 'role': role, 'content': [{'type': typ, 'text': text}]}})
    (codex / 'rollout-a.jsonl').write_text('\n'.join([
        json.dumps({'timestamp': at(0), 'type': 'session_meta', 'payload': {'id': 'a'}}),
        cx(0.5, 'user', 'input_text', 'codex user'), cx(14, 'assistant', 'output_text', 'codex reply'),
        cx(0, 'developer', 'input_text', 'dev')]) + '\n', encoding='utf-8')
    (codex / 'rollout-sub.jsonl').write_text('\n'.join([
        json.dumps({'timestamp': at(0), 'type': 'session_meta', 'payload': {'id': 'b', 'thread_source': 'subagent'}}),
        cx(2.5, 'user', 'input_text', 'subagent brief')]) + '\n', encoding='utf-8')
    ag = home / '.gemini/antigravity/brain/sess/.system_generated/logs'
    ag.mkdir(parents=True)
    (ag / 'transcript.jsonl').write_text('\n'.join([
        json.dumps({'step_index': 1, 'source': 'USER_EXPLICIT', 'type': 'USER_INPUT', 'created_at': at(20), 'content': '<USER_REQUEST>ag user</USER_REQUEST>'}),
        json.dumps({'step_index': 2, 'source': 'MODEL', 'type': 'PLANNER_RESPONSE', 'created_at': at(21), 'content': 'ag reply', 'tool_calls': []}),
        json.dumps({'step_index': 3, 'source': 'MODEL', 'type': 'VIEW_FILE', 'created_at': at(22)})]) + '\n', encoding='utf-8')
    return home, proj / 's1.jsonl'


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-chatlog-')
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.state, self.vault = self.tmp / 'state', self.tmp / 'vault'
        self.state.mkdir()
        self.vault.mkdir()
        self.calls = []
        saved = {k: os.environ.pop(k, None) for k in ('MEMROVE_NO_LLM', 'MEMROVE_INVOKED_BY')}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v) for k, v in saved.items() if v])
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)
        os.environ['MEMROVE_DOC_REPOS'] = str(self.tmp / 'no-repos')   # search must not read the owner's docs
        self.addCleanup(os.environ.pop, 'MEMROVE_DOC_REPOS', None)

    def add(self, n, size=100, kind='user', start=0, date='2026-10-07T10:00:00+00:00'):
        chatlog.append(self.state, [{'kind': kind, 'text': f'm{start + k} ' + 'x' * size, 'date': date} for k in range(n)])

    def stub(self, fn=None):
        def runner(system, payload, vault=None):
            self.calls.append(payload)
            return (fn or self.echo)(payload), 0.0, None
        llm.RUNNER = runner

    @staticmethod
    def echo(payload):
        out = []
        for m in ct.re.finditer(r'<input key="\[\[(\d+)\]\]" op="(\w+)" covers="([^"]+)">', payload):
            out.append(f'[[{m[1]}]] summary of {m[3]}')
        return '\n'.join(out)


class LogTest(Base):
    def test_append_ids_flush_torn_tail_and_lock(self):
        self.add(3)
        self.assertEqual(chatlog.count(self.state), 3)
        path = next((chatlog.root(self.state) / 'main').glob('*.jsonl'))
        with open(path, 'ab') as f:
            f.write(b'{"i": 3, "kind": "user", "te')  # crash mid-write
        self.assertEqual(chatlog.count(self.state), 3)
        self.add(1)
        recs = list(chatlog.messages(self.state))
        self.assertEqual([r['i'] for r in recs], [0, 1, 2, 3])
        self.assertEqual(recs[3]['size'], len(('user: ' + recs[3]['text']).encode()))
        self.assertEqual(chatlog.get(self.state, 2)['text'][:2], 'm2')
        with chatlog.lock(self.state) as first:
            self.assertTrue(first)
            with chatlog.lock(self.state) as second:
                self.assertFalse(second)
        with chatlog.lock(self.state) as again:
            self.assertTrue(again)

    def test_split_text_keeps_every_byte_and_respects_limit(self):
        text = ('para one. ' * 300 + '\n\n') * 5 + 'ş' * 5000 + 'z' * 20000
        parts = chatlog.split_text(text, 2000)
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(len(p.encode()) <= 2000 for p in parts))
        self.assertGreater(len(parts), 10)
        self.assertEqual(chatlog.split_text('short'), ['short'])


class IngestTest(Base):
    def test_sources_order_filters_redaction_cursors(self):
        home, s1 = fake_home(self.tmp)
        stats = chatlog.ingest(self.state, home)
        recs = list(chatlog.messages(self.state))
        texts = [(r['src'], r['kind'], r['text'][:30]) for r in recs]
        self.assertEqual([r['date'] for r in recs], sorted(r['date'] for r in recs))
        self.assertIn(('claude', 'user', 'merhaba, bunu yap'), texts)
        self.assertIn(('claude', 'agent', 'OK, doing it.'), texts)
        self.assertIn(('claude', 'user', 'the quick brown fox for the la'), texts)  # English kept
        self.assertIn(('claude', 'user', '/foo bar baz'), texts)
        self.assertIn(('claude', 'user', 'mid-turn word'), texts)  # typed while the agent worked
        self.assertIn(('codex', 'user', 'codex user'), texts)
        self.assertIn(('codex', 'agent', 'codex reply'), texts)
        self.assertIn(('antigravity', 'user', 'ag user'), texts)
        self.assertIn(('antigravity', 'agent', 'ag reply'), texts)
        joined = ' '.join(r['text'] for r in recs)
        for gone in ('secret thoughts', 'subagent brief', 'dev', 'task done', 'side', 'meta', 'tool_result'):
            self.assertNotIn(gone, [r['text'] for r in recs], gone)
        self.assertNotIn('sk-aaaa', joined)
        self.assertIn('[REDACTED]', joined)
        self.assertEqual(stats['repeat_pastes'], 1)
        self.assertEqual(sum(1 for r in recs if r['text'].startswith('P' * 100)), 1)
        # incremental: nothing new on a second run, and a replaced-file cursor reset does not double the log
        n = chatlog.count(self.state)
        self.assertEqual(chatlog.ingest(self.state, home)['new'], 0)
        (self.state / 'chatlog/cursors.json').unlink()
        self.assertEqual(chatlog.ingest(self.state, home)['new'], 0)
        with open(s1, 'a') as f:
            f.write(json.dumps({'type': 'assistant', 'uuid': 'late', 'timestamp': datetime(2026, 10, 8).astimezone().isoformat(), 'message': {'content': [{'type': 'text', 'text': 'later reply'}]}}) + '\n')
            f.write('{"type": "user", "half')  # half-written line waits
        self.assertEqual(chatlog.ingest(self.state, home)['new'], 1)
        self.assertEqual(chatlog.count(self.state), n + 1)

    def test_since_and_long_text_split_into_contiguous_messages(self):
        home, s1 = fake_home(self.tmp)
        with open(s1, 'a') as f:
            f.write(json.dumps({'type': 'user', 'uuid': 'big', 'timestamp': datetime(2026, 10, 7, 12).astimezone().isoformat(), 'message': {'content': 'word ' * 5000}}) + '\n')
        chatlog.ingest(self.state, home, since='2026-10-07')
        big = [r for r in chatlog.messages(self.state) if r['text'].startswith('word')]
        self.assertGreaterEqual(len(big), 3)
        self.assertEqual(''.join(r['text'] for r in big), ('word ' * 5000).strip())
        self.assertEqual([r['i'] for r in big], list(range(big[0]['i'], big[0]['i'] + len(big))))
        shutil_home = self.tmp / 'h2'
        shutil_home.mkdir()
        self.assertEqual(chatlog.ingest(self.tmp / 's2', shutil_home)['new'], 0)

    def test_lean_import_keeps_user_and_final_answers_before_the_day(self):
        home, _ = fake_home(self.tmp)
        rx = next((home / '.codex/sessions/2026/10/07').glob('rollout-a.jsonl'))
        ts = datetime(2026, 10, 7, 11).astimezone().isoformat()
        with open(rx, 'a') as f:
            for ph, txt in (('commentary', 'codex narration'), ('final_answer', 'codex final')):
                f.write(json.dumps({'timestamp': ts, 'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'phase': ph,
                                    'content': [{'type': 'output_text', 'text': txt}]}}) + '\n')
        ag = next(home.glob('.gemini/antigravity/brain/*/.system_generated/logs/transcript.jsonl'))
        with open(ag, 'a') as f:
            f.write(json.dumps({'step_index': 9, 'source': 'MODEL', 'type': 'PLANNER_RESPONSE', 'created_at': ts, 'content': 'ag narration', 'tool_calls': [{'name': 'x'}]}) + '\n')
        stats = chatlog.ingest(self.state, home, lean_before='2026-10-08')
        texts = [(r['src'], r['kind'], r['text']) for r in chatlog.messages(self.state)]
        for gone in ('codex narration', 'ag narration', 'OK, doing it.'):  # narration: codex commentary, a Claude text before a tool call, an Antigravity step with tool calls
            self.assertNotIn(gone, [t[2] for t in texts])
        for kept in ('codex final', 'codex reply', 'ag reply', 'codex user', 'ag user', 'merhaba, bunu yap'):
            self.assertIn(kept, [t[2] for t in texts])
        self.assertEqual(stats['lean_dropped'], {'codex': 1, 'antigravity': 1, 'claude': 1})
        # from the day on everything stays; the first value is kept like the floor
        self.assertEqual(json.loads((self.state / 'chatlog/config.json').read_text())['lean_before'], '2026-10-08')

    def test_lean_import_off_by_default(self):
        home, _ = fake_home(self.tmp)
        chatlog.ingest(self.state, home)
        self.assertIn('OK, doing it.', [r['text'] for r in chatlog.messages(self.state)])

    def test_old_files_are_skipped_by_since(self):
        home, _ = fake_home(self.tmp)
        self.assertEqual(chatlog.ingest(self.state, home, since='2026-10-09')['new'], 0)

    def test_first_since_stays_the_floor(self):
        # a later ingest without --since (the mod, the hook worker) must not append older history after newer messages
        home, _ = fake_home(self.tmp)
        chatlog.ingest(self.state, home, since='2026-10-09')
        self.assertEqual(chatlog.ingest(self.state, home)['new'], 0)
        self.assertEqual(chatlog.ingest(self.state, home, since='2026-01-01')['new'], 0)

    def test_first_call_keeps_both_since_and_lean_before(self):
        home, _ = fake_home(self.tmp)
        chatlog.ingest(self.state, home, since='2026-01-01', lean_before='2026-10-04')
        conf = json.loads((chatlog.root(self.state) / 'config.json').read_text())
        self.assertEqual((conf.get('since'), conf.get('lean_before')), ('2026-01-01', '2026-10-04'))


class TreeTest(Base):
    def test_free_nodes_cascade_and_persistence(self):
        self.add(4, size=20)
        mem = ct.Memory(self.state)
        self.assertEqual(mem.sync(), 4)
        n = mem.tree.nodes
        self.assertTrue(n[(0, 0)].startswith('user: m0'))
        self.assertEqual(n[(1, 0)], n[(0, 0)] + '\n' + n[(0, 1)])  # two short lines are joined by a newline
        self.assertIn((2, 0), n if len(n[(1, 0)]) * 2 < 512 else [(2, 0)])
        mem.save()
        again = ct.Memory(self.state)
        self.assertEqual(again.tree.nodes, mem.tree.nodes)
        self.assertEqual(again.view.parts, mem.view.parts)

    def test_long_messages_wait_in_queue_and_batch_is_leaves_then_merges(self):
        self.add(50, size=700)
        mem = ct.Memory(self.state)
        mem.sync()
        batch = ct.next_batch(mem)
        self.assertEqual([t['op'] for t in batch], ['message'] * ct.LEAF_MAX)
        self.assertEqual([t['i'] for t in batch], list(range(ct.LEAF_MAX)))
        for t in batch:
            mem.tree.put(0, t['i'], 's' * 400)
        batch2 = ct.next_batch(mem)
        ops = [t['op'] for t in batch2]
        self.assertEqual(ops[:ct.LEAF_MAX], ['message'] * ct.LEAF_MAX)
        self.assertIn('merge', ops)  # merges of finished halves ride along with the next leaves
        self.assertEqual(sorted({t['n'] for t in batch2}), list(range(1, len(batch2) + 1)))


class ViewTest(Base):
    def full_tree(self, n, size=300):
        tree = ct.Tree(self.state)
        for j in range(n):
            tree.nodes[(0, j)] = f'user: {j} ' + 'a' * size
        for l in range(1, 12):
            for i in range(n >> l):
                tree.nodes[(l, i)] = f'L{l} {i} ' + 'b' * size
        return tree

    def test_sawtooth_grows_to_hi_then_drops_to_lo_and_only_ever_coarsens(self):
        tree = self.full_tree(2000)
        v = ct.View(128_000, 64_000)
        sizes, drops, seen_starts = [], 0, None
        for j in range(2000):
            v.append(tree, (0, j))
            before = len(v.parts)
            v.fit(tree, j + 1)
            drops += len(v.parts) < before
            sizes.append(v.total)
            self.assertLessEqual(v.total, 128_000 + 400)
            self.assertEqual(v.total, sum(len(tree.line(*p).encode()) + 1 for p in v.parts))
        self.assertGreaterEqual(drops, 2)
        self.assertGreater(max(sizes), 127_000)
        self.assertLessEqual(min(s for s in sizes[800:]), 64_800)
        cover = 0
        for l, i in v.parts:  # the view tiles [0, T) in order
            self.assertEqual(i << l, cover)
            cover += 1 << l
        self.assertEqual(cover, 2000)

    def test_drain_resumes_when_parents_get_built(self):
        tree = ct.Tree(self.state)
        v = ct.View(2000, 1000)
        for j in range(40):
            tree.nodes[(0, j)] = 'x' * 100
            v.append(tree, (0, j))
        v.fit(tree, 40)
        self.assertTrue(v.draining)
        self.assertEqual(len(v.parts), 40)  # no parent built: nothing to merge yet
        for l, n in ((1, 20), (2, 10), (3, 5)):
            for i in range(n):
                tree.nodes[(l, i)] = 'y' * 100
        v.fit(tree, 40)
        self.assertLessEqual(v.total, 1000)
        self.assertFalse(v.draining)

    def test_view_json_is_persisted_and_never_rebuilt(self):
        self.add(300, size=600)
        mem = ct.Memory(self.state)
        mem.sync()
        for t in ct.next_batch(mem):
            mem.tree.put(0, t['i'], 'sum ' + str(t['i']))
        mem.refit([(0, j) for j in range(ct.LEAF_MAX)])
        mem.save()
        saved = json.loads((chatlog.root(self.state) / 'view.json').read_text())
        self.assertEqual(saved['T'], 300)
        self.add(5, size=20, start=300)
        later = ct.Memory(self.state)
        self.assertEqual(later.view.parts, mem.view.parts)  # loaded as it was, not folded again from the log
        self.assertEqual(later.sync(), 5)
        self.assertEqual(later.view.parts[:len(mem.view.parts)], mem.view.parts)

    def test_consecutive_views_share_a_long_prefix(self):
        tree = self.full_tree(600)
        v = ct.View(128_000, 64_000)
        shares, prev = [], None
        for j in range(600):
            v.append(tree, (0, j))
            before = len(v.parts)
            v.fit(tree, j + 1)
            cur = '\n'.join(tree.line(*p) for p in v.parts).encode()
            if prev is not None and len(v.parts) >= before:  # append-only steps
                shares.append(ct._common(prev, cur) / len(prev))
            prev = cur
        self.assertGreater(min(shares), 0.99)


class ReadersTest(Base):
    def test_wake_pages_fit_and_cover_the_view_and_zoom_date(self):
        self.add(1200, size=700)
        self.stub()
        with chatlog.lock(self.state):
            chatlog.ingest(self.state, self.tmp / 'nohome')
            out = ct.compact(self.state, self.vault, max_calls=500, force=True)
        self.assertEqual(out['unbuilt_leaves'], 0)
        mem = ct.Memory(self.state)
        pg = ct.pages([mem.tree.line(*p) for p in mem.view.parts])
        texts = [ct.wake(self.state, k + 1) for k in range(len(pg))]
        self.assertTrue(all(ct.u16(t) <= 30_000 for t in texts))
        body = [x for t in texts for x in t.splitlines() if '|' in x and x[0].isdigit()]
        self.assertEqual(body, [mem.tree.line(*p) for p in mem.view.parts])
        self.assertIn(f'part 1/{len(pg)}', texts[0])
        with self.assertRaises(ValueError):
            ct.wake(self.state, len(pg) + 1)
        l, i = mem.view.parts[0]
        zoomed = ct.zoom(self.state, i << l, 1 << l).splitlines() if l else None
        if l:
            self.assertEqual(len(zoomed), 2)
            self.assertTrue(zoomed[0].startswith(f'{i << l}+{1 << (l - 1)}|'))
        whole = ct.zoom(self.state, 3, 1)
        self.assertTrue(whole.startswith('3+1|user (2026-10-07'))
        self.assertIn('m3 ' + 'x' * 700, whole)
        with self.assertRaises(ValueError):
            ct.zoom(self.state, 3, 2)
        with self.assertRaises(ValueError):
            ct.zoom(self.state, 0, 3)
        self.assertEqual(ct.date_of(self.state, 3), '2026-10-07 10:00:00 +0000')

    def test_zoom_pages_a_long_message(self):
        chatlog.append(self.state, [{'kind': 'user', 'text': ('line of text\n' * 4000), 'date': '2026-10-07T10:00:00+00:00'}])
        ct.Memory(self.state).sync()
        mem = ct.Memory(self.state)
        mem.sync()
        mem.save()
        first = ct.zoom(self.state, 0, 1)
        self.assertRegex(first, r'\(page 1/3: next `memrove zoom 0 1 --page 2`\)')
        self.assertLessEqual(ct.u16(first), 30_000)
        self.assertIn('page 3/3)', ct.zoom(self.state, 0, 1, 3))

    def test_instruction_is_short_and_only_when_a_chat_exists(self):
        self.assertEqual(ct.instruction(self.state), '')
        self.add(2, size=10)
        mem = ct.Memory(self.state)
        mem.sync()
        mem.save()
        text = ct.instruction(self.state)
        self.assertIn('memrove wake', text)
        self.assertIn('memrove zoom', text)
        self.assertIn('memrove search', text)
        self.assertLess(len(text), 2500)


class ParserTest(unittest.TestCase):
    def test_parse_cases(self):
        txt = ('Here you go:\n```\n[[1]] first line\n[[2]] second\n  wrapped part\n[[2]] duplicate ignored\n[[4]] 12+1|copied head kept out\n[[9]] unknown\n```\n'
               'thanks')
        got = ct.parse_batch(txt, {1, 2, 3, 4})
        self.assertEqual(got, {1: 'first line', 2: 'second wrapped part', 4: 'copied head kept out'})  # key 3 missing
        self.assertEqual(ct.parse_batch(None, {1}), {})
        self.assertEqual(ct.parse_batch('[[1]]   \n[[2]] x', {1, 2}), {2: 'x'})

    def test_too_long_cut_is_byte_exact_and_utf8_safe(self):
        cut = ct.cut_bytes('ş' * 400)
        self.assertLessEqual(len(cut.encode()), ct.TARGET)
        self.assertEqual(len(cut), ct.TARGET // 2)
        task = ct.build_task([{'n': 1, 'op': 'message', 'id': '5+1', 'input': 'user: hi', 'last': 'a' * 600}])
        self.assertIn('Too long: your line is 600 bytes, over the limit.', task)
        self.assertIn('a' * ct.TARGET + '| ← LIMIT', task)
        self.assertIn('-' * ct.TARGET, task)


class CompactorTest(Base):
    def run_compact(self, **kw):
        with chatlog.lock(self.state) as got:
            self.assertTrue(got)
            return ct.compact(self.state, self.vault, **({'force': True} | kw))

    def test_batches_build_everything_with_few_calls_and_separate_cap(self):
        self.add(300, size=700)
        llm_usage = self.state / 'llm-usage.json'
        llm_usage.write_text(json.dumps({'day': datetime.now().date().isoformat(), 'calls': 7}))
        self.stub()
        out = self.run_compact(max_calls=100)
        self.assertEqual(out['unbuilt_leaves'], 0)
        self.assertLess(out['calls'], 40)
        self.assertEqual(json.loads(llm_usage.read_text())['calls'], 7)  # the J-job budget is untouched
        self.assertEqual(json.loads((self.state / 'chatlog/usage.json').read_text())['calls'], out['calls'])
        mem = ct.Memory(self.state)
        self.assertTrue(all((0, j) in mem.tree.nodes for j in range(300)))
        self.assertTrue(any(l >= 4 for (l, _) in mem.tree.nodes))  # merges were built too
        log = [json.loads(x) for x in (self.state / 'llm-log.jsonl').read_text().splitlines()]
        self.assertTrue(all(e['job'] == 'chatlog' for e in log))
        # a second run has nothing left to do and makes no call
        self.calls.clear()
        self.assertEqual(self.run_compact()['calls'], 0)

    def test_over_limit_lines_go_back_with_the_cut_then_shortest_after_five_tries(self):
        self.add(1, size=900)
        seen = []

        def fn(payload):
            seen.append(payload)
            n = len(seen)
            return '[[1]] ' + 'w' * (700 - n * 10)
        self.stub(fn)
        out = self.run_compact(max_calls=20)
        self.assertEqual(out['calls'], ct.TRIES)
        self.assertNotIn('Too long', seen[0])
        self.assertIn('Too long: your line is 690 bytes', seen[1])
        text = ct.Memory(self.state).tree.nodes[(0, 0)]
        self.assertEqual(len(text), 700 - 5 * 10)  # the shortest of the five tries is kept
        self.assertEqual(ct.Memory(self.state).tree.pending, {})

    def test_wake_all_joins_every_part(self):
        self.add(300, size=700)
        self.stub()
        self.run_compact(max_calls=50)
        n = len(ct.pages([ct.Memory(self.state).tree.line(*p) for p in ct.Memory(self.state).view.parts]))
        text = ct.wake_all(self.state)
        self.assertEqual(text.count('<chat part='), n)
        self.assertTrue(text.rstrip().endswith('messages)'))

    def test_line_up_to_512_plus_slack_is_accepted_without_a_retry(self):
        self.add(1, size=900)
        n = ct.NODE + ct.SLACK
        self.stub(lambda payload: '[[1]] ' + 'w' * n)
        out = self.run_compact(max_calls=20)
        self.assertEqual(out['calls'], 1)
        self.assertEqual(len(ct.Memory(self.state).tree.nodes[(0, 0)]), n)
        self.assertEqual(ct.NODE, 512)   # the prompt still asks for 512

    def test_line_over_the_slack_still_retries(self):
        self.add(1, size=900)
        seen = []
        self.stub(lambda payload: seen.append(payload) or '[[1]] ' + 'w' * (ct.NODE + ct.SLACK + 1))
        self.run_compact(max_calls=2)
        self.assertEqual(len(seen), 2)
        self.assertIn('Too long: your line is 537 bytes', seen[1])

    def test_ingest_only_and_status_for_the_transport(self):
        self.add(3, size=700)
        self.assertNotIn('locked', ct.ingest_only(self.state, self.tmp / 'nohome'))
        st = ct.status(self.state)
        self.assertEqual((st['calls_today'], st['cap']), (0, 60))
        ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', ingest=False, force=True)
        self.assertEqual(ct.status(self.state)['calls_today'], 1)

    def test_cap_default_config_and_disabled_and_failures(self):
        self.assertEqual(ct.cap(self.state), 60)
        (self.state / 'llm-config.json').write_text(json.dumps({'daily_cap': 95, 'chatlog_cap': 3}))
        self.assertEqual(ct.cap(self.state), 3)
        self.add(200, size=700)
        self.stub()
        self.assertEqual(self.run_compact()['calls'], 3)
        self.assertEqual(self.run_compact()['calls'], 0)  # today's three calls are spent
        os.environ['MEMROVE_INVOKED_BY'] = 'memrove-scripts'  # the hook-recursion guard
        self.assertEqual(self.run_compact(max_calls=5)['stopped'], 'disabled')
        os.environ.pop('MEMROVE_INVOKED_BY')

        def broken(system, payload, vault=None):
            return None, 0.0, 'claude-timeout'
        llm.RUNNER = broken
        out = self.run_compact(max_calls=10)
        self.assertEqual((out['calls'], out['stopped']), (2, 'failing'))

    def test_refusal_halves_the_batch_marks_the_node_and_its_sibling_pieces_and_nothing_else(self):
        chatlog.append(self.state, [{'kind': 'user', 'text': 'ok %d ' % j + 'x' * 700, 'date': '2026-10-07T10:00:00+00:00', 'k': f'c:ok{j}'} for j in range(6)]
                       + [{'kind': 'user', 'text': 'POISON %d ' % j + 'x' * 700, 'date': '2026-10-07T10:00:00+00:00', 'k': 'c:big' + ('#%d' % j if j else '')} for j in range(4)]
                       + [{'kind': 'user', 'text': 'ok2 %d ' % j + 'x' * 700, 'date': '2026-10-07T10:00:00+00:00', 'k': f'c:fine{j}'} for j in range(6)])

        def runner(system, payload, vault=None):
            self.calls.append(payload)
            if 'POISON' in payload:
                return None, 0.0, "claude-exit-1: API Error: Haiku 5.5 can't help with this. Learn more: https://www.anthropic.com/legal/aup"
            return self.echo(payload), 0.0, None
        llm.RUNNER = runner
        out = self.run_compact(max_calls=40)
        self.assertEqual(out['unbuilt_leaves'], 0)
        nodes = ct.Memory(self.state).tree.nodes
        for j in range(6):
            self.assertTrue(nodes[(0, j)].startswith('summary of'))
        self.assertEqual([nodes[(0, j)] for j in range(6, 10)], [f'user: {ct.REFUSED}'] * 4)
        self.assertTrue(all(nodes[(0, j)].startswith('summary of') for j in range(10, 16)))
        self.assertEqual(out['refused'], 4)
        self.assertLess(out['calls'], 20)

    def test_refusals_of_four_different_texts_stop_the_run(self):
        self.add(8, size=700)

        def runner(system, payload, vault=None):
            return None, 0.0, "claude-exit-1: API Error: Haiku 5.5 can't help with this. https://www.anthropic.com/legal/aup"
        llm.RUNNER = runner
        out = self.run_compact(max_calls=60)
        self.assertEqual(out['stopped'], 'refusing')

    def test_prepare_apply_round_trip_cap_and_claim_expiry(self):
        self.add(40, size=700)
        (self.state / 'llm-config.json').write_text(json.dumps({'chatlog_cap': 3}))
        b = ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)
        self.assertEqual(set(b), {'batch_id', 'system', 'prompt', 'node_ids'})
        self.assertEqual(len(b['node_ids']), ct.LEAF_MAX)
        self.assertIn('Compaction batch', b['prompt'])
        # a second claim does not hand out the claimed nodes again
        b2 = ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)
        self.assertFalse(set(b['node_ids']) & set(b2['node_ids']))
        reply = '\n'.join(f'[[{n}]] line for {nid}' for n, nid in enumerate(b['node_ids'], 1)) + '\n[[3]] ' + 'z' * 600
        res = ct.apply(self.state, b['batch_id'], reply)
        self.assertEqual(res['applied'], ct.LEAF_MAX)   # key 3 answered twice: the first answer wins
        self.assertEqual(res['retry_ids'], [])
        self.assertFalse(res['done'])
        self.assertEqual(ct.apply(self.state, b['batch_id'], reply)['error'], 'unknown or expired batch')
        # over-long answer: kept pending with its retry id, and the next claim carries the Too long cut
        long_ = '\n'.join(f'[[{n}]] ' + 'q' * 600 for n, _ in enumerate(b2['node_ids'], 1))
        res2 = ct.apply(self.state, b2['batch_id'], long_)
        self.assertEqual(sorted(res2['retry_ids']), sorted(b2['node_ids']))
        b3 = ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)
        self.assertIn('Too long: your line is 600 bytes', b3['prompt'])
        # the cap counts every claim from every transport: three claims made, a fourth is refused
        self.assertEqual(ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True), {})
        self.assertEqual(json.loads((self.state / 'chatlog/usage.json').read_text())['calls'], 3)

    def test_unapplied_claim_expires_and_its_nodes_return(self):
        self.add(5, size=700)
        with chatlog.lock(self.state):
            claim = ct._prepare(self.state, ttl=-1)   # already expired
            self.assertEqual(len(claim['tasks']), 5)
            again = ct._prepare(self.state)
        self.assertEqual([t['id'] for t in again['tasks']], [t['id'] for t in claim['tasks']])
        self.assertFalse((self.state / 'chatlog/claims' / (claim['batch_id'] + '.json')).exists())
        # a live claim holds its nodes back; an expired one cannot be applied
        with chatlog.lock(self.state):
            self.assertIsNone(ct._prepare(self.state))
            stale = None
        gone = ct.apply(self.state, again['batch_id'], '[[1]] x')
        self.assertEqual(gone['applied'], 1)
        with chatlog.lock(self.state):
            dead = ct._prepare(self.state, ttl=-1)
        self.assertEqual(ct.apply(self.state, dead['batch_id'], '[[1]] x')['error'], 'unknown or expired batch')
        self.assertNotIn((0, dead['tasks'][0]['i']), ct.Memory(self.state).tree.nodes)

    def test_refused_batch_over_apply_is_halved_into_waiting_batches(self):
        self.add(4, size=700)
        b = ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)
        r = ct.apply(self.state, b['batch_id'], None, "claude-exit-1: API Error: Haiku 5.5 can't help with this. https://www.anthropic.com/legal/aup")
        self.assertEqual(r['applied'], 0)
        h1 = ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)
        h2 = ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)
        self.assertEqual((len(h1['node_ids']), len(h2['node_ids'])), (2, 2))

    def _prep(self):
        return ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', force=True)

    def test_silent_error_on_the_same_node_set_twice_is_a_refusal_and_marks_only_the_poison_node(self):
        self.add(4, size=700)
        poison = None
        for _ in range(40):
            b = self._prep()
            if not b:
                break
            if poison is None:
                poison = b['node_ids'][2]
            if poison in b['node_ids']:
                ct.apply(self.state, b['batch_id'], None, 'api-error 400 invalid_request')
            else:
                ct.apply(self.state, b['batch_id'], self.echo(b['prompt']))
        nodes = ct.Memory(self.state).tree.nodes
        self.assertEqual(nodes[(0, 2)], f'user: {ct.REFUSED}')
        self.assertTrue(all(nodes[(0, j)].startswith('summary of') for j in (0, 1, 3)))

    def test_one_silent_error_only_retries_the_same_set(self):
        self.add(4, size=700)
        b = self._prep()
        ct.apply(self.state, b['batch_id'], None, 'empty-reply')
        again = self._prep()
        self.assertEqual(again['node_ids'], b['node_ids'])

    def test_failures_of_different_node_sets_in_a_row_stop_halving_and_mark_nothing(self):
        self.add(4, size=700)
        for _ in range(12):  # an outage: every call fails
            b = self._prep()
            if not b:
                break
            ct.apply(self.state, b['batch_id'], None, 'api-error 529 overloaded')
        mem = ct.Memory(self.state)
        self.assertEqual(mem.tree.nodes, {})
        self.assertTrue(mem.tree.pending['_errs']['halt'])
        # the transport is back: an answered call clears the halt
        b = self._prep()
        ct.apply(self.state, b['batch_id'], self.echo(b['prompt']))
        self.assertFalse(ct.Memory(self.state).tree.pending['_errs']['halt'])

    def test_no_view_line_is_cut_text(self):
        self.add(40, size=2000)
        mem = ct.Memory(self.state)
        mem.sync()
        for p in mem.view.parts:
            line = mem.tree.line(*p)
            self.assertTrue(line.endswith(ct.PLACEHOLDER) or '2000' not in line)
        self.assertNotIn('x' * 400, '\n'.join(mem.tree.line(*p) for p in mem.view.parts))   # only the 300-byte head of a pending message shows


class GateTest(Base):
    """A model call only when it is worth it: 20 ready nodes, a full view with its merges ready, the oldest ready node 4 h old, or a force."""

    def now(self):
        return datetime.now().astimezone().isoformat(timespec='seconds')

    def prep(self, **kw):
        return ct.prepare(self.state, self.vault, home=self.tmp / 'nohome', **kw)

    def test_a_fill_20_ready_nodes(self):
        self.add(ct.MIN_FILL - 1, size=700, date=self.now())
        self.assertEqual(self.prep(), {})
        self.add(1, size=700, start=ct.MIN_FILL - 1, date=self.now())
        self.assertEqual(len(self.prep()['node_ids']), ct.MIN_FILL)
        self.assertEqual(json.loads((self.state / 'chatlog/usage.json').read_text())['calls'], 1)

    def test_a_short_messages_and_held_nodes_do_not_count(self):
        self.add(30, size=100, date=self.now())   # no model node at all
        self.assertEqual(self.prep(), {})
        self.add(ct.MIN_FILL, size=700, start=30, date=self.now())
        self.assertTrue(self.prep())
        self.assertEqual(self.prep(), {})   # the claimed 20 are held: nothing more is ready

    def test_b_full_view_with_ready_merges(self):
        self.add(6, size=700, date=self.now())
        b = self.prep(force=True)
        ct.apply(self.state, b['batch_id'], '\n'.join(f'[[{n}]] ' + 'q' * 300 for n in range(1, 7)))   # long lines: their merges need the model
        mem = ct.Memory(self.state)
        self.assertEqual(len(mem.tree.merges), 3)
        self.assertIsNone(ct.gate_reason(mem))   # 3 recent merges, the view far under VIEW_HI: not worth a call
        self.assertEqual(self.prep(), {})
        saved = ct.VIEW_HI, ct.VIEW_LO
        self.addCleanup(lambda: (setattr(ct, 'VIEW_HI', saved[0]), setattr(ct, 'VIEW_LO', saved[1])))
        ct.VIEW_HI, ct.VIEW_LO = 1500, 1000
        mem = ct.Memory(self.state)
        self.assertGreaterEqual(mem.view.total, ct.VIEW_HI)
        self.assertEqual(ct.gate_reason(mem), 'view')
        self.assertEqual(len(self.prep()['node_ids']), 3)
        # a full view whose merges are not ready (their halves are still unbuilt) is no reason either
        __import__('shutil').rmtree(self.state / 'chatlog')
        self.add(4, size=700, date=self.now())
        self.assertIsNone(ct.gate_reason(ct.Memory(self.state)))

    def test_c_oldest_ready_node_waited_max_wait(self):
        ago = lambda sec: (datetime.now().astimezone() - timedelta(seconds=sec)).isoformat(timespec='seconds')
        self.add(1, size=700, date=ago(ct.MAX_WAIT - 120))
        self.assertEqual(self.prep(), {})
        self.add(1, size=700, start=1, date=self.now())
        self.assertEqual(self.prep(), {})
        mem = ct.Memory(self.state)
        later = datetime.now().astimezone() + timedelta(seconds=300)
        self.assertEqual(ct.gate_reason(mem, now=later), 'wait')   # five minutes later the oldest has waited MAX_WAIT; the batch takes both
        self.assertEqual(len(self.prep(force=True)['node_ids']), 2)

    def test_d_force(self):
        self.add(1, size=700, date=self.now())
        self.assertEqual(self.prep(), {})
        self.assertEqual(len(self.prep(force=True)['node_ids']), 1)

    def test_compact_and_run_follow_the_gate(self):
        self.add(3, size=700, date=self.now())
        self.stub()
        with chatlog.lock(self.state):
            self.assertEqual(ct.compact(self.state, self.vault)['calls'], 0)
            self.assertEqual(ct.compact(self.state, self.vault, force=True)['calls'], 1)
        self.assertEqual(self.calls and len(self.calls), 1)

    def test_manual_run_forces_unless_no_force(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('memrove_cli', ROOT / 'scripts' / 'memrove_v3.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        self.assertFalse(cli.parser().parse_args(['--vault', '/tmp', 'chatlog', 'run']).no_force)
        self.assertTrue(cli.parser().parse_args(['--vault', '/tmp', 'chatlog', 'run', '--no-force']).no_force)

    def test_view_limits_from_config_and_an_existing_larger_view_shrinks_by_the_drain(self):
        self.assertEqual(ct.view_limits(self.state), (32_000, 16_000))
        cfg = self.state / 'llm-config.json'
        cfg.write_text(json.dumps({'chatlog_view_hi': 1_000_000, 'chatlog_view_lo': 900_000}))
        self.add(60, size=700, date=self.now())
        self.stub(lambda payload: '\n'.join(f'[[{m[1]}]] ' + 'q' * 400 for m in ct.re.finditer(r'<input key="\[\[(\d+)\]\]"', payload)))
        with chatlog.lock(self.state):
            ct.compact(self.state, self.vault, force=True)
        before = ct.Memory(self.state)
        self.assertGreater(before.view.total, 20_000)   # grown past the new limit, never drained
        nodes, calls = len(before.tree.nodes), json.loads((self.state / 'chatlog/usage.json').read_text())['calls']
        cfg.write_text(json.dumps({'chatlog_view_hi': 8_000, 'chatlog_view_lo': 4_000}))
        self.assertEqual(ct.view_limits(self.state), (8_000, 4_000))
        self.add(1, size=100, start=60, date=self.now())
        mem = ct.Memory(self.state)
        mem.sync()
        self.assertLessEqual(mem.view.total, 4_000 + 600)
        self.assertEqual(len(mem.tree.nodes), nodes + 1)   # the new message only: the shrink merges built parents, nothing is rebuilt
        self.assertEqual(json.loads((self.state / 'chatlog/usage.json').read_text())['calls'], calls)
        cfg.write_text(json.dumps({'chatlog_view_hi': 100, 'chatlog_view_lo': 500}))   # lo over hi: lo falls to half of hi
        self.assertEqual(ct.view_limits(self.state), (100, 50))

    def test_pending_long_message_shows_its_head(self):
        text = 'a' + 'é' * 400   # 801 bytes; byte 300 falls inside a 2-byte character
        chatlog.append(self.state, [{'kind': 'user', 'text': text, 'date': self.now()}, {'kind': 'agent', 'text': 'short reply', 'date': self.now()}])
        want = '0+1|user: a' + 'é' * 149 + '… ' + ct.PLACEHOLDER
        mem = ct.Memory(self.state)
        mem.sync()
        self.assertEqual(mem.tree.line(0, 0), want)
        self.assertEqual(mem.tree.line(0, 1), '1+1|agent: short reply')   # <= 512 B: verbatim, unchanged
        mem.save()
        self.assertEqual(ct.Memory(self.state).tree.line(0, 0), want)   # a fresh process rebuilds the head from the log
        self.assertEqual(ct.Memory(self.state).view.total, sum(len(ct.Memory(self.state).tree.line(*p).encode()) + 1 for p in mem.view.parts))
        b = self.prep(force=True)
        ct.apply(self.state, b['batch_id'], '[[1]] built line')
        self.assertEqual(ct.Memory(self.state).tree.line(0, 0), '0+1|built line')


class OptOutAndSearch(Base):
    def project(self, home, name, cwd_key='cwd'):
        proj = home / '.claude/projects' / name
        proj.mkdir(parents=True)
        return proj

    def test_ingest_skips_no_memory_projects_for_claude_and_codex(self):
        home = self.tmp / 'h'
        cwd_out, cwd_in = self.tmp / 'quiet', self.tmp / 'loud'
        for c in (cwd_out, cwd_in):
            (c / '.memrove').mkdir(parents=True)
        (cwd_out / '.memrove' / 'no-memory').write_text('')
        ts = '2026-10-07T10:00:00+00:00'

        def claude(name, cwd, word):
            f = self.project(home, name) / 's.jsonl'
            f.write_text(json.dumps({'type': 'user', 'uuid': name, 'timestamp': ts, 'cwd': str(cwd), 'message': {'role': 'user', 'content': word}}) + '\n')
        claude('a', cwd_out, 'eval transcript word')
        claude('b', cwd_in, 'real chat word')
        cx = home / '.codex/sessions/2026/10/07'
        cx.mkdir(parents=True)
        for n, cwd, word in (('out', cwd_out, 'codex eval word'), ('in', cwd_in, 'codex real word')):
            (cx / f'rollout-{n}.jsonl').write_text('\n'.join([
                json.dumps({'timestamp': ts, 'type': 'session_meta', 'payload': {'id': n, 'cwd': str(cwd)}}),
                json.dumps({'timestamp': ts, 'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': word}]}})]) + '\n')
        with chatlog.lock(self.state) as held:
            self.assertTrue(held)
            chatlog.ingest(self.state, home)
        texts = [m['text'] for m in chatlog.messages(self.state)]
        self.assertEqual(sorted(texts), ['codex real word', 'real chat word'])
        with chatlog.lock(self.state):  # the skipped files are read once, not re-read every turn
            self.assertEqual(chatlog.ingest(self.state, home)['new'], 0)

    def test_search_finds_turkish_forms_incrementally(self):
        import memrove_v3_chatsearch as cs
        chatlog.append(self.state, [{'kind': 'user', 'text': t, 'date': '2026-10-07T10:00:00+00:00'} for t in (
            'Bugün hava güzel', 'Kararlarımızı yazalım: IŞIK sistemi için postgres seçtik', 'başka bir konu')])
        out = cs.search(self.state, 'karar postgres')
        self.assertEqual(out.splitlines()[0].split('|')[:3], ['1', '2026-10-07 10:00', 'user'])
        self.assertIn('postgres', out)
        self.assertEqual(cs.search(self.state, 'ışık').split('|')[0], '1')  # dotless i folds
        self.assertEqual(cs.search(self.state, 'zzzz'), 'no message matches')
        chatlog.append(self.state, [{'kind': 'agent', 'text': 'new message postgres', 'date': '2026-10-07T11:00:00+00:00'}])
        self.assertEqual({l.split('|')[0] for l in cs.search(self.state, 'postgres').splitlines()}, {'1', '3'})

    def test_search_reindexes_a_replaced_log(self):
        import shutil
        import memrove_v3_chatsearch as cs
        chatlog.append(self.state, [{'kind': 'user', 'text': t, 'date': '2026-10-07T10:00:00+00:00'} for t in ('eski postgres', 'eski redis')])
        self.assertEqual(cs.search(self.state, 'postgres').split('|')[0], '0')
        shutil.rmtree(chatlog.root(self.state) / 'main')  # a longer log swapped in: same ids, other messages
        chatlog.append(self.state, [{'kind': 'user', 'text': t, 'date': '2026-10-01T10:00:00+00:00'} for t in ('yeni sqlite', 'yeni mongo', 'yeni postgres')])
        self.assertEqual(cs.search(self.state, 'postgres').split('|')[0], '2')
        self.assertEqual(cs.search(self.state, 'redis'), 'no message matches')

    def test_search_full_word_beats_prefix_and_compounds_meet(self):
        import memrove_v3_chatsearch as cs
        d = '2026-10-07T10:00:00+00:00'
        chatlog.append(self.state, [{'kind': 'user', 'text': t, 'date': d} for t in (
            'graph graph graph code review graph torch graph', 'graphs and graphite', 'we chose graphql for the api',
            'context-mode kuruldu', 'unrelated context here', 'mode only here')])
        ids = lambda q: [l.split('|')[0] for l in cs.search(self.state, q).splitlines()]
        self.assertEqual(ids('graphql')[0], '2')  # the full token ranks above the graph* prefix hits
        self.assertIn('0', ids('graphql'))        # prefix matching stays for inflection
        for q in ('context-mode', 'context mode', 'contextmode'):
            self.assertEqual(ids(q)[0], '3', q)

    def test_search_date_range(self):
        import memrove_v3_chatsearch as cs
        days = ('2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08')
        chatlog.append(self.state, [{'kind': 'user', 'text': f'postgres note {d}', 'date': f'{d}T10:00:00+00:00'} for d in days])
        ids = lambda out: [l.split('|')[0] for l in out.splitlines()]
        self.assertEqual(sorted(ids(cs.search(self.state, 'postgres', since='2026-10-07'))), ['2', '3'])
        self.assertEqual(sorted(ids(cs.search(self.state, 'postgres', until='2026-10-06'))), ['0', '1'])  # until is inclusive
        self.assertEqual(ids(cs.search(self.state, 'postgres', since='2026-10-06', until='2026-10-07')).__len__(), 2)
        self.assertEqual(ids(cs.search(self.state, 'postgres', until='2026-10-06T09:00:00')), ['0'])
        self.assertEqual(cs.search(self.state, 'postgres', since='2026-11-01'), 'no message matches')
        self.assertEqual(ids(cs.search(self.state, '', since='2026-10-06', until='2026-10-07')), ['2', '1'])  # newest first
        self.assertEqual(ids(cs.search(self.state, '', since='2026-10-06', until='2026-10-06')), ['1'])
        self.assertEqual(cs.search(self.state, ''), 'no message matches')
        with self.assertRaises(ValueError):
            cs.search(self.state, 'postgres', since='last week')

    def test_search_merges_saved_records_with_a_tag(self):
        import sqlite3
        import memrove_v3_chatsearch as cs
        chatlog.append(self.state, [{'kind': 'user', 'text': 'we talked about the zebra cache once', 'date': '2026-10-07T10:00:00+00:00'}])
        rec = lambda i, **kw: (i, json.dumps({'kind': 'decision', 'status': 'active', 'source': 'notes/observed/decision/x.md',
                                              'valid_from': '2026-10-05', 'text': 'zebra cache is dropped for redis', 'why': 'too slow', **kw}))
        db = sqlite3.connect(self.state / 'memory.sqlite3')
        db.execute('CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        db.executemany('INSERT INTO records VALUES (?,?)', [
            rec('obs-new', supersedes=['obs-old'], valid_from='2026-10-06', text='zebra cache is back, redis is out'),
            rec('obs-old'),
            rec('obs-late', valid_from='2026-10-09', text='zebra cache late word'),
            rec('obs-off', status='dropped_llm', text='zebra cache dropped record'),
            rec('q-auto', source='notes/captured/decision/y.md', text='zebra cache auto captured'),
            rec('obs-priv', visibility='private', text='zebra cache private'),
            rec('obs-gone', valid_to='2026-10-06', text='zebra cache expired')])
        db.commit()
        db.close()
        ids = lambda out: [l.split('|')[0] for l in out.splitlines()]
        out = cs.search(self.state, 'zebra cache', until='2026-10-08')
        self.assertEqual(sorted(ids(out)), ['0', 'rec:obs-new', 'rec:q-auto'])  # hidden: replaced, later than T, dropped, private, expired; cue-captured (q-) records are returned
        self.assertIn('|decision|zebra cache is back', out)
        self.assertEqual([i for i in ids(cs.search(self.state, 'zebra cache', until='2026-10-06T09:00')) if i.startswith('rec:')], ['rec:obs-old', 'rec:q-auto'])  # a datetime T excludes its own day; the successor is not yet valid
        self.assertEqual(sorted(i for i in ids(cs.search(self.state, 'zebra cache')) if i.startswith('rec:')), ['rec:obs-late', 'rec:obs-new'])


if __name__ == '__main__':
    unittest.main()
