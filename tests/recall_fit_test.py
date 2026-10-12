#!/usr/bin/env python3
"""Learn-then-Test fitting and the cascade runner of the recall eval (synthetic candidates, no model)."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_hybrid as hyb
import recall_eval as ev
import recall_fit as fit


def cand(i, cos, z=0.0, pcover=0.0, cos2=None, z2=0.0):
    return {'id': i, 'cos': cos, 'z': z, 'pcover': pcover, 'cos2': cos2, 'z2': z2}


def world(n=60, seed=1):
    """Prompt k has one relevant record (high cosine) and noise; every 3rd prompt has nothing relevant."""
    rnd = random.Random(seed)
    tops, rows = {}, {}
    for k in range(n):
        pid, rel = f'p{k}', k % 3 != 0
        cs = [cand(f'{pid}-r', 0.8 + rnd.random() * 0.1 if rel else 0.4 + rnd.random() * 0.2)]
        cs += [cand(f'{pid}-n{j}', 0.3 + rnd.random() * 0.3) for j in range(5)]
        tops[pid] = {'top': cs}
        rows[pid] = {'candidates': [{'record_id': c['id'], 'label': 'relevant' if c['id'].endswith('-r') and rel else 'not'} for c in cs]}
    return tops, (lambda pid: ({c['record_id']: c['label'] for c in rows[pid]['candidates']}, set())), list(tops)


class LttTest(unittest.TestCase):
    def test_clopper_pearson_upper(self):
        self.assertAlmostEqual(fit.cp_upper(0, 10, 0.1), 1 - 0.1 ** 0.1, places=4)
        self.assertEqual(fit.cp_upper(5, 5, 0.1), 1.0)
        self.assertGreater(fit.cp_upper(3, 10, 0.1), 0.3)

    def test_fixed_sequence_stops_at_first_failure(self):
        stats = [(0, 0), (10, 0), (20, 1), (30, 12), (40, 1)]  # the last one would pass but follows a failure
        self.assertEqual(fit.ltt_pick(stats, 0.3, 0.1), 2)
        self.assertIsNone(fit.ltt_pick([(0, 0), (5, 4)], 0.3, 0.1))

    def test_pick_matches_decide(self):
        top = [dict(cand(f'x{i}', 0.5 + i * 0.05), s1=0.5 + i * 0.05) for i in range(6)]
        for tau, delta in ((0.6, 0.1), (0.5, 0.3), (0.9, 0.1)):
            want = [c['id'] for c in hyb.decide(top, {'tau1': tau, 'delta1': delta}, 'x')[0]]
            got = fit.pick(fit.rank(top, lambda c: c['s1']), tau, delta)
            self.assertEqual(got, want)


class FitTest(unittest.TestCase):
    def test_fit_meets_alpha_out_of_sample_and_abstains_when_it_cannot(self):
        tops, judge, pids = world()
        a, b = pids[:30], pids[30:]
        f = fit.fit_single(tops, a, judge, 0.3)
        self.assertIsNotNone(f)
        acc, fal, ans, hit = fit.tally(fit.run_single(tops, b, f), judge)
        self.assertLessEqual(fal / acc, 0.3)
        self.assertGreater(hit, 0)
        self.assertIsNone(fit.fit_single(tops, a[:2], judge, 0.01))  # two prompts cannot certify anything
        self.assertEqual(fit.run_single(tops, b, None), {p: [] for p in b})

    def test_cascade_l2_rescues_only_the_band(self):
        tops, judge, pids = world()
        for pid in pids:  # L2 sees what L1 cannot: the relevant record gets a clearly higher cos2
            for c in tops[pid]['top']:
                c['cos2'] = 0.9 if c['id'].endswith('-r') and int(pid[1:]) % 3 else 0.3
        f = fit.fit_cascade(tops, pids, judge, 0.3)
        self.assertIsNotNone(f['l1'])
        runs = fit.run_cascade(tops, pids, f)
        acc, fal, ans, hit = fit.tally(runs, judge)
        self.assertGreater(acc, 0)
        self.assertLessEqual(fal / acc, 0.3)
        single = fit.tally(fit.run_single(tops, pids, f['l1']), judge)
        self.assertGreaterEqual(hit, single[3])  # never fewer hits than L1 alone

    def test_pool_covers_every_injectable_pair(self):
        tops, judge, pids = world(12)
        pool = fit.pool_pairs(tops, pids)
        for z, lw, d in fit.configs():
            ranked = {p: fit.rank(tops[p]['top'], fit.k1(z, lw)) for p in pids}
            for t in (0.0, 0.5, 1.0):
                for p, ids in {p: fit.pick(ranked[p], t, d) for p in pids}.items():
                    self.assertTrue(all((p, i) in pool for i in ids))


class EvalGlueTest(unittest.TestCase):
    def test_merge_extra_and_query_memo(self):
        rows = {'q1': {'candidates': [{'record_id': 'a', 'label': 'not'}]}}
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'x.jsonl'
            f.write_text(json.dumps({'prompt_id': 'q1', 'candidates': [{'record_id': 'a', 'label': 'relevant'}, {'record_id': 'b', 'label': 'partly', 'label_source': 'e31-pool'}]}) + '\n')
            ev.merge_extra(rows, f)
            self.assertEqual([(c['record_id'], c['label']) for c in rows['q1']['candidates']], [('a', 'not'), ('b', 'partly')])

            class Inner:
                name, calls = 'm', 0

                def embed_query(self, texts):
                    Inner.calls += len(texts)
                    return [[1.0, 0.0] for _ in texts]
            m = ev.MemoQuery(Inner(), Path(tmp) / 'q.json')
            m.embed_query(['a', 'b'])
            m2 = ev.MemoQuery(Inner(), Path(tmp) / 'q.json')
            m2.embed_query(['a', 'b', 'c'])
            self.assertEqual(Inner.calls, 3)  # the second run embedded only the new query
            self.assertEqual(m2.name, 'm')


if __name__ == '__main__':
    unittest.main()
