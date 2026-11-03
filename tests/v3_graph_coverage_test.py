#!/usr/bin/env python3
"""Link coverage: remember queues the record, backfill respects its caps and resumes, embeddings add pairs with no shared words."""
import array
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_graph as G  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

WORDS = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliet']


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memrove-cov-')
        self.addCleanup(self.tmp.cleanup)
        self.vault = Path(self.tmp.name) / 'vault'
        self.vault.mkdir()
        (self.vault / 'README.md').write_text('hello', encoding='utf-8')
        self.state = Path(self.tmp.name) / 'state'
        self.engine = SyncEngine(self.vault, self.state)
        for i, w in enumerate(WORDS):
            self.engine.store.ingest(dict(id=f'dec:{i}', kind='decision', text=f'{w} subject {w} detail {w} words', project=f'/work/p{i % 2}',
                                          source='README.md', session='abcdef1234567890', updated_at=f'2026-10-{i + 1:02d}'))

    def fake_runner(self):
        def fake(system, payload, vault=None):
            nodes = json.loads(payload)['nodes']
            return json.dumps({'nodes': [{'id': n['id'], 'edges': [{'dst': n['candidates'][0]['id'], 'type': 'relates', 'why': 'ayni konu'}] if n['candidates'] else [],
                                          'concept': None} for n in nodes]}), 0.01, None
        old, llm.RUNNER = llm.RUNNER, fake
        self.addCleanup(setattr, llm, 'RUNNER', old)

    def test_remember_queues_the_record(self):
        env = dict(os.environ, MEMROVE_NO_LLM='1')  # no worker is spawned; the queue line is the contract
        out = subprocess.run([sys.executable, str(ROOT / 'scripts/memrove_v3.py'), '--vault', str(self.vault), '--state', str(self.state), 'remember',
                              'Queue me for linking at write time', '--kind', 'decision', '--project', 'p', '--cwd', self.tmp.name],
                             capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(out.stdout.strip().splitlines()[0], G.pending(self.state))

    def test_queued_record_is_linked_before_older_unseen_ones(self):
        self.fake_runner()
        G.enqueue(self.state, 'dec:0')  # the oldest record would come last by day order
        res = G.link_new(self.engine, self.vault, self.state, limit=1, token_cap=10 ** 6, budget={'left': 5, 'calls': 0, 'cost': 0.0})
        self.assertEqual(res['nodes'], 1)
        with self.engine.store._connect() as db:
            self.assertEqual([r[0] for r in db.execute('SELECT id FROM graph_seen')], ['dec:0'])
        self.assertEqual(G.pending(self.state), [])

    def test_backfill_respects_cost_cap_resumes_and_undoes(self):
        self.fake_runner()

        def run(*a):
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                G.main(['backfill', '--tokens', '1000000', *a], self.vault, self.state, lambda ro: self.engine.store, lambda: self.engine)
            return json.loads(buf.getvalue())
        dry = run('--max-usd', '1', '--dry-run', '--job', 'bf1')
        self.assertEqual((dry['nodes'], dry['calls'], dry['batches']), (10, 0, 3))
        first = run('--max-usd', '0.02', '--job', 'bf1')
        self.assertEqual((first['calls'], first['nodes'], first['stopped']), (2, 8, 'cost-cap'))
        second = run('--max-usd', '0.02', '--job', 'bf1')  # resumes: only the two unseen nodes are left
        self.assertEqual((second['calls'], second['nodes']), (1, 2))
        self.assertEqual(run('--max-usd', '1', '--job', 'bf1')['nodes'], 0)  # idempotent
        with self.engine.store._connect() as db:
            self.assertTrue(G.load(db)[1] and all(e['prov']['job'] == 'bf1' for e in G.load(db)[1]))
        self.assertEqual(G.undo(self.engine, 'bf1')[0] > 0, True)
        with self.engine.store._connect() as db:
            self.assertEqual(G.load(db)[1], [])  # nodes the job linked are unseen again (a zero-edge node stays seen: nothing of it to undo)

    def test_embedding_finds_pair_without_shared_words(self):
        with self.engine.store._connect() as db:
            pool = G.load_nodes(db)
        pool = {i: dict(pool[i]) for i in ('dec:0', 'dec:7')}
        pool['dec:7'].update(title='zebra quartz', text='zebra quartz')  # shares no word with dec:0
        index, node = G.Index(pool), pool['dec:0']
        vec = lambda *x: array.array('f', x)  # noqa: E731
        vecs = {'dec:0': vec(1, 0), 'dec:7': vec(1, 0)}
        self.assertNotIn('dec:7', G.candidates(index, node))
        self.assertIn('dec:7', G.candidates(index, node, vecs=vecs))


if __name__ == '__main__':
    unittest.main()
