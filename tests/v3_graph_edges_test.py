#!/usr/bin/env python3
"""Link graph: edges carry provenance, the linker only links what it was shown, the view has no index star and no junk hubs."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3_graph as G  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_projections as P  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location('graph_stats', ROOT / 'scripts/graph_stats.py')
graph_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(graph_stats)


class GraphEdgesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memrove-e66-')
        self.addCleanup(self.tmp.cleanup)
        self.vault = Path(self.tmp.name) / 'vault'
        self.vault.mkdir()
        (self.vault / 'README.md').write_text('hello', encoding='utf-8')
        self.state = Path(self.tmp.name) / 'state'
        self.engine = SyncEngine(self.vault, self.state)
        base = dict(source='README.md', session='abcdef1234567890', updated_at='2026-10-01')
        for rid, project, text in [
            ('dec:1', '/work/Alpha App', 'Hybrid search uses BM25 and embeddings. Second sentence.'),
            ('dec:2', '/work/beta', 'Hybrid search fuses ranks with RRF in beta.'),
            ('dec:3', str(Path.home()), 'Home folder decision about nothing.'),
            ('dec:4', 'unknown', 'Unknown project decision.'),
        ]:
            self.engine.store.ingest(dict(base, id=rid, kind='decision', text=text, project=project))

    def db(self):
        return self.engine.store._connect()

    def test_edge_needs_provenance_reason_and_known_type(self):
        with self.db() as db:
            prov = G.provenance('agent:test', 'agent', session='s1')
            G.add_edge(db, 'dec:1', 'dec:2', 'relates', 'both are about hybrid search', prov)
            stored = G.load(db)[1][0]
            self.assertEqual((stored['src'], stored['dst'], stored['type']), ('dec:1', 'dec:2', 'relates'))
            self.assertEqual(stored['prov']['by'], 'agent:test')
            self.assertEqual(stored['prov']['basis'], 'agent')
            self.assertTrue(stored['prov']['at'])
            for bad in (dict(prov=None), dict(prov={'by': 'x'}), dict(why=''), dict(type='friends'), dict(dst='dec:1')):
                args = dict(src='dec:1', dst='dec:2', type='relates', why='w', prov=prov) | bad
                with self.assertRaises(ValueError, msg=bad):
                    G.add_edge(db, args['src'], args['dst'], args['type'], args['why'], args['prov'])

    def test_linker_refuses_links_to_non_candidates(self):
        with self.db() as db:
            nodes = G.load_nodes(db)
            pool = {i: n for i, n in nodes.items()}
            index = G.Index(pool)
            items = G._payload([pool['dec:1']], pool, index)
            shown = {c['id'] for c in items[0]['candidates']}
            self.assertIn('dec:2', shown)
            answer = {'nodes': [{'id': 'dec:1', 'edges': [
                {'dst': 'dec:2', 'type': 'relates', 'why': 'same subject'},
                {'dst': 'dec:4', 'type': 'relates', 'why': 'never shown'},
                {'dst': 'dec:2', 'type': 'part_of', 'why': 'part_of needs a concept'},
                {'dst': 'dec:2', 'type': 'bogus', 'why': 'unknown type'}],
                'concept': {'name': 'hybrid search', 'why': 'ranking by lexical plus dense signals'}}]}
            added, made, refused = G.apply_answer(db, items, answer)
            self.assertEqual((added, made, refused), (2, 1, 3))
            edges = G.load(db)[1]
            self.assertEqual({(e['dst'], e['type']) for e in edges}, {('dec:2', 'relates'), ('concept:hybrid-search', 'part_of')})
            self.assertTrue(all(e['prov']['basis'] == 'model' and e['prov']['job'] == 'j5' for e in edges))

    def test_linker_call_goes_through_runner_and_token_cap(self):
        calls = []

        def fake(system, payload, vault=None):
            calls.append(payload)
            ids = [n['id'] for n in json.loads(payload)['nodes']]
            return json.dumps({'nodes': [{'id': i, 'edges': [], 'concept': None} for i in ids]}), 0.01, None
        old, llm.RUNNER = llm.RUNNER, fake
        self.addCleanup(setattr, llm, 'RUNNER', old)
        res = G.link_new(self.engine, self.vault, self.state, limit=4, token_cap=10)
        self.assertEqual((res['calls'], res['stopped']), (0, 'token-cap'))
        res = G.link_new(self.engine, self.vault, self.state, limit=4, token_cap=10 ** 6, budget={'left': 5, 'calls': 0, 'cost': 0.0})
        self.assertEqual((res['calls'], res['nodes']), (1, 4))
        res = G.link_new(self.engine, self.vault, self.state, limit=4, token_cap=10 ** 6, budget={'left': 5, 'calls': 0, 'cost': 0.0})
        self.assertEqual(res['nodes'], 0)  # seen nodes are not sent again

    def test_card_links_contract(self):
        block = G.card_links({'groups': ['Harness'], 'project': '/work/panel'}, 'text')
        self.assertIn('[[memrove/concepts/harness|Harness]]', block)
        self.assertIn('[[memrove/projects/panel|panel]]', block)
        self.assertEqual(G.card_links({}, 'x'), '')
        self.assertNotIn('Proje', G.card_links({'proje': 'unknown'}, 'x'))

    def test_view_has_no_index_star_no_junk_and_readable_names(self):
        with self.db() as db:
            prov = G.provenance('agent:test', 'agent')
            cid = G.add_concept(db, 'hybrid search', 'BM25 plus embeddings', prov)
            G.add_edge(db, 'dec:1', cid, 'part_of', 'it is hybrid search', prov)
            G.add_edge(db, 'dec:2', cid, 'part_of', 'it is hybrid search too', prov)
        result = P.project_graph(self.engine, budget_s=30)
        self.assertEqual(result['conflicts'], [])
        files = {p.relative_to(self.vault).as_posix(): p.read_text(encoding='utf-8') for p in (self.vault / 'memrove').rglob('*.md')}
        self.assertTrue(any(re.fullmatch(r'memrove/decisions/hybrid-search-uses-bm25-and-embeddings-[0-9a-z]{4,8}\.md', f) for f in files), sorted(files))
        self.assertFalse([f for f in files if '0000-00-00' in f])
        self.assertIn('memrove/projects/genel.md', files)
        for junk in ('dev', 'unknown', 'claude', Path.home().name):
            self.assertNotIn(f'memrove/projects/{junk}.md', files)
        self.assertFalse([f for f, t in files.items() if f != 'memrove/index.md' and '[[memrove/index' in t])
        concept = files['memrove/concepts/hybrid-search.md']
        self.assertIn('## alpha-app', concept)
        self.assertIn('## beta', concept)
        dec = next(t for f, t in files.items() if f.startswith('memrove/decisions/hybrid-search-uses-bm25'))
        self.assertIn('# Hybrid search uses BM25 and embeddings', dec)
        self.assertIn('part_of: [[memrove/concepts/hybrid-search|hybrid search]] — it is hybrid search', dec)
        stats = graph_stats.stats(self.vault)
        self.assertEqual((stats['broken'], stats['links_to_index']), (0, 0), stats)

    def test_cli_queries_show_why_and_provenance(self):
        with self.db() as db:
            prov = G.provenance('agent:test', 'agent')
            cid = G.add_concept(db, 'hybrid search', 'BM25 plus embeddings', prov)
            G.add_edge(db, 'dec:1', cid, 'part_of', 'it is hybrid search', prov)
            G.add_edge(db, 'dec:2', cid, 'part_of', 'it is hybrid search too', prov)
            text = G.neighbors(db, None, 'dec:1', hops=2)
            self.assertIn('part_of: it is hybrid search (agent:test/agent)', text)
            self.assertIn('dec:2'[:0] + 'decision Hybrid search fuses ranks', text)
            self.assertIn('project:alpha-app', text)
            self.assertLessEqual(len(text.encode()), 1500)
            self.assertIn('alpha-app', G.related(db, None, 'hybrid search'))
            self.assertIn('concepts: hybrid search (1)', G.hub(db, None, 'beta'))

    def test_project_of_skips_home_folder_cwd(self):
        transcript = Path(self.tmp.name) / 't.jsonl'
        project = Path(self.tmp.name) / 'proj'
        (project / '.git').mkdir(parents=True)
        transcript.write_text(json.dumps({'cwd': str(Path.home())}) + '\n' + json.dumps({'cwd': str(project)}) + '\n', encoding='utf-8')
        self.assertEqual(llm._project_of(transcript), 'proj')
        transcript.write_text(json.dumps({'cwd': str(Path.home())}) + '\n', encoding='utf-8')
        self.assertEqual(llm._project_of(transcript), 'unknown')


if __name__ == '__main__':
    unittest.main()
