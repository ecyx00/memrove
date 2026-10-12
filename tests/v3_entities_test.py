#!/usr/bin/env python3
"""E11: entities from J3, merge/alias rules, J4 compile with citation filter, stale chain, projection, match()."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_docs as docs  # noqa: E402
import memrove_v3_entities as ent  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402
import memrove_v3_projections as P  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

A, B = 'doc:repo/docs/supabase.md', 'doc:repo/docs/auth.md'


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-ent-')
        self.addCleanup(tmp.cleanup)
        r = Path(tmp.name)
        self.vault, self.state, self.root = r / 'vault', r / 'state', r / 'repo'
        for p in (self.vault, self.state, self.root / 'docs'):
            p.mkdir(parents=True)
        (self.root / 'docs/supabase.md').write_text('# Supabase settings\n\nRLS is on for every table.\n', encoding='utf-8')
        (self.root / 'docs/auth.md').write_text('# Auth rules\n\nLogin goes through Supabase auth.\n', encoding='utf-8')
        self.engine = SyncEngine(self.vault, self.state)
        self.store = self.engine.store
        self.reply = {}
        llm.RUNNER = self.runner
        self.addCleanup(setattr, llm, 'RUNNER', llm._claude)
        for k in ('MEMROVE_NO_LLM', 'MEMROVE_INVOKED_BY'):
            os.environ.pop(k, None)
        ent._CACHE.clear()
        docs.scan(self.store, self.root)

    def runner(self, system, payload, vault=None):
        job = next(k for k, v in llm.SYSTEM.items() if v == system)
        r = self.reply.get(job)
        return (json.dumps(r) if r is not None else None), 0.0, (None if r is not None else 'boom')

    def tag(self, a, b=None):
        self.reply['j3'] = {'items': [{'id': A, 'kind': 'reference', 'topics': [], 'entities': a}, {'id': B, 'kind': 'rule', 'topics': [], 'entities': b or []}]}
        llm.tag_docs(self.engine, self.vault, self.state)

    def ents(self):
        with self.store._connect() as db:
            return {e['id']: e for e in ent.all_entities(db)}

    def compile(self, lines):
        self.reply['j4'] = {'sections': {'what': lines}}
        return ent.compile_entities(self.store, self.vault, self.state, self.root)


class EntityTest(Base):
    def test_j3_entities_create_edges(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase', 'sb'], 'kind': 'service'}])
        e = self.ents()['entity:supabase']
        self.assertEqual((e['kind'], e['sources'], e['stale']), ('entity', [A], True))
        self.assertEqual(e['aliases'], ['supabase'])  # 'sb' is shorter than 3
        with self.store._connect() as db:
            self.assertEqual(db.execute("SELECT dst FROM links WHERE src=? AND rel='about_entity'", (A,)).fetchall(), [('entity:supabase',)])

    def test_merge_by_name_and_alias(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supa base'], 'kind': 'service'}],
                 [{'name': 'Supa Base', 'aliases': [], 'kind': 'service'}])
        self.assertEqual(list(self.ents()), ['entity:supabase'])
        self.assertEqual(self.ents()['entity:supabase']['sources'], sorted([A, B]))

    def test_merge_by_shared_alias(self):
        self.tag([{'name': 'Supabase', 'aliases': ['sbase'], 'kind': 'service'}], [{'name': 'Supabase Cloud', 'aliases': ['sbase'], 'kind': 'service'}])
        self.assertEqual(len(self.ents()), 1)

    def test_ambiguous_alias_not_matched(self):
        self.tag([{'name': 'Supabase', 'aliases': ['pgx'], 'kind': 'service'}], [{'name': 'Postgres', 'aliases': ['pgx'], 'kind': 'tool'}])
        self.assertEqual(len(self.ents()), 2)
        self.assertEqual(ent.match(self.store, 'pgx config'), [])
        self.assertEqual([m[0] for m in ent.match(self.store, 'postgres and supabase')], ['entity:postgres', 'entity:supabase'])

    def test_stoplist_negatives(self):
        words = 'app api web data test user admin server ayar proje sistem'.split()
        self.tag([{'name': 'Thing', 'aliases': words, 'kind': 'tool'}])
        self.assertEqual(self.ents()['entity:thing']['aliases'], [])
        self.assertEqual(ent.match(self.store, ' '.join(words)), [])

    def test_j4_drops_uncited_and_foreign_lines(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.assertEqual(self.compile([f'RLS enabled. [{A}]', 'Unsourced line.', f'Other document. [{B}]', 'Made up. [doc:x/y.md]']), 1)
        e = self.ents()['entity:supabase']
        self.assertEqual([p['t'] for p in e['profile']], [f'RLS enabled. [{A}]'])
        self.assertEqual((e['origin_type'], e['model_version'], e['stale']), ('derived', 'haiku', False))

    def test_stale_after_edit_and_cleared_after_recompile(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.compile([f'RLS enabled. [{A}]'])
        self.assertEqual(ent.refresh_stale(self.store), 0)
        f = self.root / 'docs/supabase.md'
        f.write_text('# Supabase settings\n\nRLS is OFF now.\n', encoding='utf-8')
        os.utime(f, (1, 1))
        docs.scan(self.store, self.root)
        self.assertEqual(ent.refresh_stale(self.store), 1)
        self.assertTrue(self.ents()['entity:supabase']['stale'])
        self.compile([f'RLS disabled. [{A}]'])
        e = self.ents()['entity:supabase']
        self.assertFalse(e['stale'])
        self.assertEqual(ent.refresh_stale(self.store), 0)

    def test_removed_source_marks_stale(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}], [{'name': 'Supabase', 'aliases': [], 'kind': 'service'}])
        self.compile([f'RLS enabled. [{A}]'])
        (self.root / 'docs/auth.md').unlink()
        docs.scan(self.store, self.root)
        self.assertEqual(ent.refresh_stale(self.store), 1)
        self.assertEqual(self.ents()['entity:supabase']['sources'], [A])

    def test_projection_and_user_note_wins(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.compile([f'RLS enabled. [{A}]'])
        P.project_graph(self.engine, budget_s=30)
        note = self.vault / 'memrove/entities/supabase.md'
        text = note.read_text(encoding='utf-8')
        self.assertIn('"generated": true', text)
        self.assertIn('[[memrove/docs/repo/docs-supabase]]', text)
        self.assertIn('[[memrove/projects/repo]]', text)
        # a user-authored note at another entity's path is never overwritten
        ent.record_doc_entities(self.store, B, [{'name': 'Stripe', 'aliases': ['stripe'], 'kind': 'service'}])
        (self.vault / 'memrove/entities').mkdir(exist_ok=True)
        mine = self.vault / 'memrove/entities/stripe.md'
        mine.write_text('my own stripe notes\n', encoding='utf-8')
        res = P.project_graph(self.engine, budget_s=30)
        self.assertEqual(mine.read_text(encoding='utf-8'), 'my own stripe notes\n')
        self.assertTrue(any(c['source'].endswith('entities/stripe.md') for c in res['conflicts']))

    def test_match_supabase_rls(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}])
        self.compile([f'RLS enabled. [{A}]'])
        m = ent.match(self.store, 'Supabase RLS ayarı')
        self.assertEqual(m, [('entity:supabase', 'supabase', f'RLS enabled. [{A}]')])
        self.assertEqual(ent.match(self.store, 'supabasexxxxxxx'), [])
        self.assertEqual(ent.match(self.store, 'SUPABASE'), [m[0][:2] + (m[0][2],)])

    def test_turkish_suffix_tail(self):
        self.tag([{'name': 'Supabase', 'aliases': ['supabase'], 'kind': 'service'}],
                 [{'name': 'Vercel', 'aliases': [], 'kind': 'service'}, {'name': 'Git', 'aliases': [], 'kind': 'tool'}, {'name': 'Vite', 'aliases': [], 'kind': 'tool'}])
        ids = lambda t: [m[0] for m in ent.match(self.store, t)]
        self.assertEqual(ids('supabaseden veri'), ['entity:supabase'])
        self.assertEqual(ids('vercelin ayari'), ['entity:vercel'])
        self.assertEqual(ids('supabasexxxxxxx'), [])  # tail longer than 5
        self.assertEqual(ids('gitlab'), [])  # 3-char alias: strict boundary
        self.assertEqual(ids('vitede'), [])  # 4-char alias: strict boundary
        self.assertEqual(ids('vite ve git'), ['entity:vite', 'entity:git'])
        self.tag([{'name': 'Thing', 'aliases': 'app api web data test user admin server ayar proje sistem'.split(), 'kind': 'tool'}])
        self.assertEqual(ids('apple apps webs datas testing users'), [])

    def test_match_speed(self):
        import random, time
        rnd = random.Random(1)
        with self.store._connect() as db:
            db.execute(ent.SCHEMA)
            for i in range(50):
                names = [f'plat{i}x{j}zz' for j in range(4)]
                ent._save(db, {'id': f'entity:p{i}', 'kind': 'entity', 'name': names[0], 'aliases': names[1:], 'entity_kind': 'tool', 'sources': [A], 'stale': False,
                               'profile': [{'s': 'what', 't': f'line {k} [{A}]'} for k in range(5)]})
        words = [f'kelime{k}' for k in range(300)] + [f'plat{i}x1zz' for i in range(0, 50, 10)]
        ts = []
        for _ in range(1000):
            rnd.shuffle(words)
            t = time.perf_counter()
            ent.match(self.store, ' '.join(words[:300]))
            ts.append((time.perf_counter() - t) * 1e6)
        ts.sort()
        print('match p95 us=%d' % ts[949])
        self.assertLess(ts[949], 150000)


if __name__ == '__main__':
    unittest.main()
