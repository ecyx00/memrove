#!/usr/bin/env python3
"""Handoff card: a decision session gets a card, chit-chat none (no model call), the push stays inside its cap."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_card as card  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

SID = 'abcdef123456789'


def transcript(path, turns):
    with open(path, 'w', encoding='utf-8') as f:
        for t in turns:
            f.write(json.dumps({'type': 'user', 'message': {'content': t}}, ensure_ascii=False) + '\n')
            f.write(json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'ok'}]}}) + '\n')


class CardTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-card-')
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.vault, self.state = self.dir / 'vault', self.dir / 'state'
        self.vault.mkdir()
        self.state.mkdir()
        self.engine = SyncEngine(self.vault, self.state)
        self.calls = []
        llm.RUNNER = self.runner
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)

    def runner(self, system, payload, vault=None):
        self.calls.append(payload)
        return json.dumps({'did': ['Chose sqlite for the queue'], 'why': ['no server needed'], 'open': ['migration'], 'next': 'Write the migration.'}), 0.001, None

    def run_session(self, turns):
        tp = self.dir / 't.jsonl'
        transcript(tp, turns)
        ev = [{'event': 'SessionEnd', 'session': SID, 'transcript_path': str(tp), 'project': 'repo', 'day': '2026-10-18', 'at': 1}]
        return card.draft_all(self.engine, self.vault, self.state, ev)

    def records(self):
        with self.engine.store._connect() as db:
            return {json.loads(r[0])['id']: json.loads(r[0]) for r in db.execute('SELECT payload FROM records')}

    def test_decision_session_gets_a_card_rewritten_in_place(self):
        self.engine.note_create('notes/captured/d-1.md', 'sqlite kullanalim', dict(id='d-1', kind='decision', project='repo', session=SID, status='active'))
        self.run_session(['merhaba', 'karar verdim, kuyruk için sqlite kullanalım', 'tamam devam'])
        c = self.records()['c-abcdef123456']
        self.assertEqual((c['kind'], c['did'], c['project']), ('card', ['Chose sqlite for the queue'], 'repo'))
        with self.engine.store._connect() as db:
            self.assertIn(('c-abcdef123456', 'd-1', 'cites'), db.execute('SELECT src,dst,rel FROM links').fetchall())
        self.run_session(['merhaba', 'karar verdim, kuyruk için sqlite kullanalım', 'tamam devam', 'bir de test ekle'])   # grown transcript: same id, no second card
        self.assertEqual([i for i in self.records() if i.startswith('c-')], ['c-abcdef123456'])
        self.assertEqual(self.records()['c-abcdef123456']['revision'], 2)
        self.assertIn('- next: Write the migration.', card.block(self.state, 'repo'))

    def test_agent_written_card_is_not_overwritten_by_the_fallback(self):
        mine = {'project': 'repo', 'session': SID, 'origin': 'agent', 'origin_type': 'agent_inference', 'valid_from': '2026-10-18',
                'did': ['agent wrote this'], 'why': [], 'open': [], 'next': 'x'}
        card.upsert(self.engine, 'c-abcdef123456', mine)
        self.run_session(['merhaba', 'karar verdim, kuyruk için sqlite kullanalım', 'tamam devam'])
        self.assertEqual((self.calls, self.records()['c-abcdef123456']['did']), ([], ['agent wrote this']))

    def test_chit_chat_gets_no_card_and_no_call(self):
        self.run_session(['merhaba', 'nasılsın', 'saat kaç'])
        self.assertEqual((self.calls, [i for i in self.records() if i.startswith('c-')]), ([], []))

    def test_push_is_capped_with_truncation_line(self):
        meta = {'project': 'repo', 'session': SID, 'origin': 'llm', 'origin_type': 'agent_inference', 'valid_from': '2026-10-18',
                'did': ['did %d ' % i + 'x' * 150 for i in range(4)], 'why': ['y' * 150] * 3, 'open': ['z' * 150] * 3, 'next': 'n' * 150}
        card.upsert(self.engine, 'c-abcdef123456', meta)
        out = card.block(self.state, 'repo')
        self.assertLessEqual(len(out.encode()), card.CAP)
        self.assertEqual(out.splitlines()[-1], 'truncated, open c-abcdef123456')
        self.assertEqual(card.block(self.state, 'other'), '')


if __name__ == '__main__':
    unittest.main()
