#!/usr/bin/env python3
"""bench/lme.py on a 3-question fixture: scoring, turn -> session mapping, gate."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
sys.dont_write_bytecode = True
import lme  # noqa: E402


def q(qid, question, sessions, gold):
    """sessions: {sid: [(role, text)]}; gold: set of (sid, 1-based turn)."""
    return {'question_id': qid, 'question_type': 'single-session-user', 'question': question, 'haystack_session_ids': list(sessions),
            'haystack_dates': ['2023/01/01'] * len(sessions),
            'haystack_sessions': [[{'role': r, 'content': t, **({'has_answer': True} if (sid, n) in gold else {})} for n, (r, t) in enumerate(s, 1)]
                                  for sid, s in sessions.items()]}


FIX = [
    q('a1', 'which degree did I graduate with', {'s1': [('user', 'I like pizza'), ('assistant', 'ok')], 's2': [('user', 'I graduated with a degree in biology'), ('assistant', 'nice')], 's3': [('user', 'the weather is cold')]}, {('s2', 1)}),
    q('b2', 'what is my dog called', {'s1': [('user', 'my dog is called Rex'), ('assistant', 'good dog')], 's2': [('user', 'I like tea')]}, {('s1', 1)}),
    q('c3_abs', 'abstention question', {'s1': [('user', 'x', )]}, set()),
    q('d4', 'where do I park the car', {'s1': [('assistant', 'park the car at home')], 's2': [('user', 'nothing here')]}, set()),   # no has_answer user turn: dropped
    q('e5', 'what colour is the bike', {'s1': [('user', 'the bike is red'), ('assistant', 'red bike')], 's2': [('user', 'I own a car')], 's3': [('user', 'sunny day')]}, {('s1', 1)}),
]


class LmeBench(unittest.TestCase):
    def test_dirs_stay_under_work(self):
        self.assertEqual(lme.check_dirs('/tmp/lme/work', '/tmp/lme/work/out')[1], Path('/tmp/lme/work/out').resolve())
        for work, out in (('/tmp/lme/work', '/tmp/lme/out'), ('/tmp/lme/livecopy/x', '/tmp/lme/livecopy/x'), ('/tmp/lme/work', '/tmp/rev/o'), ('/tmp/lme/work/../livecopy', '/tmp/lme/work/o')):
            with self.assertRaises(SystemExit):
                lme.check_dirs(work, out)

    def test_filter_and_gold(self):
        s = lme.scored(FIX)
        self.assertEqual([x[0]['question_id'] for x in s], ['a1', 'b2', 'e5'])   # _abs and no-evidence questions dropped
        self.assertEqual(s[0][1:], ({'s2'}, {'s2_1'}))

    def test_turn_to_session_first_appearance(self):
        self.assertEqual(lme.sessions_of(['b_3', 'a_1', 'b_1', 'c_2', 'a_2']), ['b', 'a', 'c'])

    def test_recall(self):
        self.assertEqual(lme.recall(['x', 'g1', 'y', 'g2'], {'g1', 'g2'}, 2), (100.0, 0.0))
        self.assertEqual(lme.recall(['x', 'g1', 'y', 'g2'], {'g1', 'g2'}, 5), (100.0, 100.0))
        self.assertEqual(lme.recall(['x', 'y'], {'g1'}, 5), (0.0, 0.0))

    def test_bm25_order(self):
        self.assertEqual(lme.bm25_order([['a', 'b'], ['rare', 'b'], ['c']], ['rare']), [1, 2, 0])   # tie: later doc first

    def test_gate_on_fixture(self):
        rows = []
        for it, gs, gt in lme.scored(FIX):
            r = lme.arm_bm25_turn(it, None)
            rows.append((it['question_type'], r, gs, gt))
        m = lme.evaluate(rows)
        self.assertEqual(m['n'], 3)
        self.assertEqual(m['Sany@5'], 100.0)
        self.assertEqual(m['Tany@5'], 100.0)
        self.assertTrue(lme.gate(m, 100.0, 1.0))
        self.assertFalse(lme.gate(m))   # 100 is outside 86.2 +- 1.0


if __name__ == '__main__':
    unittest.main()
