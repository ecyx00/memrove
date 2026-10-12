#!/usr/bin/env python3
"""Hybrid (model-based) recall: stage 1 candidates, abstain, content cards, reranker stage + fallback, vector cache, static embedder, logs."""
import os as _os
_os.environ.setdefault("MEMROVE_NO_LLM", "1")
import array
import json
import math
import os
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
sys.dont_write_bytecode = True
import memrove_v3_embed as emb
import memrove_v3_hybrid as hyb
import memrove_v3_recall as recall
from v3_recall_test import Base, LESSON, PROMPT

AXES = {'pay': 0, 'invoice': 0, 'billing': 0, 'cat': 1, 'garden': 2, 'soil': 2}


class FakeEmbedder:
    """Keyword -> basis vector; a record and a prompt that name different words of one axis are still near each other."""
    name, dim = 'fake-3', 3

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0]
            for w, i in AXES.items():
                if w in t.lower():
                    v[i] += 1.0
            out.append(emb._unit(v if any(v) else [0.0, 0.0, 0.01]))
        return out


class HybridBase(Base):
    MODE = None

    def setUp(self):
        super().setUp()
        for p in (patch.object(emb, 'get_embedder', lambda spec=None: FakeEmbedder()),
                  patch.dict(os.environ, {'MEMROVE_RECALL_MODE': 'hybrid', 'DECIDE_HOME': str(self.root / 'dlog')})):
            p.start()
            self.addCleanup(p.stop)
        self.log = lambda: [json.loads(l) for l in (self.state / 'recall-log.jsonl').read_text().splitlines()]


class Stage1Test(HybridBase):
    def test_dense_proposes_a_record_with_no_shared_word_and_card_has_content(self):
        self.add('d1', kind='decision', project='other', title='Reconciliation rule', text='Always reconcile every invoice at month end, never earlier',
                 origin_type='user_explicit', updated_at='2026-10-01T00:00:00Z')
        out = self.recall('how should we pay the customers every month, thinking about billing')
        self.assertIn('[d1] decision', out)
        self.assertIn('Always reconcile every invoice at month end', out)  # the content, not a title pointer
        self.assertLessEqual(recall.tok(out), 300)

    def test_unrelated_prompt_abstains_and_logs_it(self):
        self.add('d1', kind='decision', project='other', title='Reconciliation rule', text='Always reconcile every invoice at month end',
                 origin_type='user_explicit')
        self.assertEqual(self.recall('tell me about the garden soil in spring please now'), '')
        row = self.log()[-1]
        self.assertEqual(row['passed'], [])
        self.assertTrue(row['counts']['abstain'])
        self.assertEqual(row['candidates'][0]['reason'], 'below_s1')
        self.assertIn('s1', row['candidates'][0])

    def test_no_keyword_gate_a_single_shared_word_can_pass(self):
        self.add('l1', **dict(LESSON, title='Stripe', topics=[], approach='dedupe', why='billing retries are safe', text='Stripe webhook retries billing'))
        self.assertIn('[l1]', self.recall('we need billing for the stripe flow'))

    def test_inferred_lesson_still_needs_admission(self):
        self.add('l1', **dict(LESSON, origin_type='agent_inference', verified_by='llm', seen_sessions=['a']))
        self.assertEqual(self.recall(PROMPT), '')

    def test_cwd_project_is_recalled_but_not_twice_or_when_sessionstart_showed_it(self):
        self.add('l1', **dict(LESSON, project='here'))
        self.assertIn('[l1]', self.recall(PROMPT))
        self.assertEqual(self.recall(PROMPT), '')  # same session: already injected
        with self.store._connect() as db:  # SessionStart records bare ids in the same table
            db.execute("INSERT INTO injected VALUES('s3', 0, 'l1')")
        self.assertEqual(self.recall(PROMPT, session='s3'), '')
        self.assertIn('[l1]', self.recall(PROMPT, session='s4'))

    def test_short_prompt_uses_previous_assistant_turn(self):
        self.add('l1', **LESSON)
        tr = self.root / 't.jsonl'
        tr.write_text('\n'.join(json.dumps(m) for m in [
            {'type': 'user', 'message': {'role': 'user', 'content': 'x'}},
            {'type': 'assistant', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'older'}]}},
            {'type': 'assistant', 'message': {'role': 'assistant', 'content': [{'type': 'tool_use', 'name': 'x'}, {'type': 'text', 'text': 'Next: ' + PROMPT}]}},
        ]) + '\n')
        self.assertEqual(recall.recall(self.store, 'evet devam', str(self.repo), 's1', str(tr)).count('[l1]'), 1)
        self.assertTrue(self.log()[-1]['counts']['ctx'])
        recall.recall(self.store, 'evet devam', str(self.repo), 's2')
        self.assertFalse(self.log()[-1]['counts']['ctx'])  # no transcript: the prompt alone
        self.assertTrue(recall.prev_assistant_tail(tr).startswith('Next:'))
        self.assertEqual(recall.prev_assistant_tail(self.root / 'missing'), '')

    def test_tail_read_is_bounded(self):
        tr = self.root / 'big.jsonl'
        line = json.dumps({'type': 'assistant', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'z' * 5000}]}})
        tr.write_text(line + '\n' + ('{"type":"user","message":{"role":"user","content":"' + 'u' * 100 + '"}}\n') * 600)
        self.assertEqual(recall.prev_assistant_tail(tr), '')  # the only assistant turn is outside the last 32 KB

    def test_decide_log_carries_scores_and_abstain(self):
        self.add('d1', kind='decision', project='other', title='x', text='pay invoice billing rule', origin_type='user_explicit')
        self.recall('how do we pay invoice billing here')
        self.recall('garden soil advice for the spring', session='s2')
        rows = [json.loads(l) for l in (self.root / 'dlog/log/decisions.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['use'] == 'memrove.recall']
        self.assertEqual([r['state']['abstain'] for r in rows] if isinstance(rows[0]['state'], dict) else
                         [json.loads(r['state'])['abstain'] for r in rows], [False, True])
        opt = rows[0]['options'][0]
        self.assertEqual(opt['gate'], 'model')
        self.assertIn('s1', opt)


class Stage2Test(HybridBase):
    def setUp(self):
        super().setUp()
        for i in range(4):
            self.add(f'r{i}', kind='decision', project='other', title=f'T{i}', text=f'pay invoice billing rule number {i}', origin_type='user_explicit')
        self.cfg = patch.dict(os.environ, {'MEMROVE_RERANK': '1'})
        self.cfg.start()
        self.addCleanup(self.cfg.stop)
        self.prompt = 'how do we pay the invoice billing'

    def test_reranker_scores_pick_and_margin_trim(self):
        def fake(req, timeout=1.2):
            self.assertEqual(req['op'], 'rerank')
            return {'ok': True, 'scores': [0.95, 0.9, 0.65, 0.2][:len(req['docs'])]}
        with patch.object(emb, 'daemon_call', fake):
            out = self.recall(self.prompt)
        row = self.log()[-1]
        self.assertEqual(row['counts']['stage2'], 'on')
        self.assertEqual(row['counts']['score_key'], 's2')
        self.assertEqual(len(row['passed']), 2)  # 0.65 is more than delta2=0.25 below the best, 0.2 is below tau2
        self.assertEqual(out.count('decision'), 2)

    def test_all_below_tau2_abstains(self):
        with patch.object(emb, 'daemon_call', lambda req, timeout=1.2: {'ok': True, 'scores': [0.1] * len(req['docs'])}):
            self.assertEqual(self.recall(self.prompt), '')
        self.assertTrue(self.log()[-1]['counts']['abstain'])

    def test_daemon_down_falls_back_to_stage1_and_spawns(self):
        with patch.object(emb, 'daemon_call', lambda req, timeout=1.2: None), patch.object(emb, 'spawn_daemon') as spawn:
            out = self.recall(self.prompt)
        spawn.assert_called_once()
        self.assertEqual(self.log()[-1]['counts']['stage2'], 'fallback')
        self.assertEqual(self.log()[-1]['counts']['score_key'], 's1')
        self.assertIn('decision', out)

    def test_rerank_off_by_default(self):
        self.cfg.stop()
        with patch.object(emb, 'daemon_call', side_effect=AssertionError('daemon must not be called')):
            self.recall(self.prompt)
        self.cfg.start()


class VectorCacheTest(HybridBase):
    def test_content_hash_dedupes_and_refresh_is_incremental(self):
        self.add('a', kind='decision', project='p1', title='t', text='pay invoice billing rule', origin_type='user_explicit')
        self.add('b', kind='decision', project='p2', title='t', text='pay invoice billing rule', origin_type='user_explicit')
        self.add('c', kind='decision', project='p3', title='t', text='garden soil rule', origin_type='user_explicit')
        first = emb.refresh(self.state, 'ngram')
        self.assertEqual(first, {'new': 2, 'total': 3})  # a and b share one content hash
        self.assertEqual(emb.refresh(self.state, 'ngram'), {'new': 0, 'total': 3})
        with sqlite3.connect(self.state / 'vectors.sqlite3') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM vec').fetchone()[0], 2)
        self.assertEqual(set(emb.VecStore(self.state).load('fake-3')), {'a', 'b', 'c'})

    def test_refresh_builds_l2_vectors_only_when_cascade_is_on(self):
        self.add('r0', kind='decision', project='p', title='T', text='pay invoice billing', origin_type='user_explicit')
        with patch.object(emb, 'ensure_daemon', return_value=True) as up, patch.object(emb, 'DaemonEmbedder', FakeG2):
            self.assertNotIn('l2', emb.refresh(self.state, 'ngram'))
            up.assert_not_called()
            (self.state / 'recall-config.json').write_text(json.dumps({'cascade': {'band': 'always'}}))
            self.assertEqual(emb.refresh(self.state, 'ngram')['l2']['new'], 1)
            self.assertEqual(len(emb.VecStore(self.state).load(FakeG2.name)), 1)

    def test_refresh_fails_open(self):
        self.assertIn('error', emb.refresh(self.root / 'missing-state'))


class StaticEmbedderTest(unittest.TestCase):
    def make(self, tmp):
        d = Path(tmp)
        pieces = [('<pad>', 0.0), ('<unk>', 0.0), ('▁', -2.0), ('▁ab', -1.0), ('▁a', -3.0), ('b', -3.0), ('▁cd', -1.0), ('c', -4.0), ('d', -4.0)]
        db = sqlite3.connect(d / 'vocab.sqlite')
        db.execute('CREATE TABLE vocab(piece TEXT PRIMARY KEY, id INTEGER, score REAL) WITHOUT ROWID')
        db.executemany('INSERT INTO vocab VALUES(?,?,?)', [(p, i, s) for i, (p, s) in enumerate(pieces)])
        db.commit()
        db.close()
        vecs = [[0, 0], [0, 0], [0, 0], [1, 0], [0.5, 0], [0, 0.5], [0, 1], [0, 0.2], [0, 0.2]]
        (d / 'emb.f16').write_bytes(b''.join(struct.pack('<2e', *v) for v in vecs))
        (d / 'meta.json').write_text(json.dumps({'dim': 2, 'unk': 1}))
        return d

    def test_viterbi_tokenizer_and_mean_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = emb.StaticEmbedder(self.make(tmp))
            self.assertEqual(e.tokenize('ab cd'), [3, 6])  # best-scoring segmentation, not the greedy-char one
            self.assertEqual(e.tokenize('ab ??'), [3, 2])  # unknown characters are skipped (the lone word marker stays)
            va, vb = e.embed(['ab']), e.embed(['cd'])
            self.assertAlmostEqual(va[0][0], 1.0, places=3)
            self.assertAlmostEqual(vb[0][1], 1.0, places=3)
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in e.embed(['ab cd'])[0])), 1.0, places=3)
            self.assertTrue(emb.StaticEmbedder.installed(Path(tmp)))

    def test_get_embedder_auto_falls_back_to_ngram(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MEMROVE_MODELS': tmp}):
            self.assertEqual(emb.get_embedder('auto').name, 'ngram-v1')
            self.assertFalse(emb.StaticEmbedder.installed())


class ConfigTest(unittest.TestCase):
    def test_file_then_env_and_per_embedder_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'recall-config.json').write_text(json.dumps({'tau1': {'ngram-v1': 0.9}, 'rerank': True}))
            with patch.dict(os.environ, {'MEMROVE_RECALL_MODE': 'gate'}):
                cfg = hyb.config(tmp)
            self.assertEqual((cfg['mode'], cfg['rerank']), ('gate', True))
            self.assertEqual(hyb.threshold(cfg, 'tau1', 'ngram-v1'), 0.9)
            self.assertEqual(hyb.threshold(cfg, 'tau1', 'other'), hyb.DEFAULTS['tau1'])
        self.assertFalse(hyb.DEFAULTS['rerank'])  # shipped off until measured


class DaemonClientTest(unittest.TestCase):
    def test_no_socket_is_none_and_spawn_is_throttled(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MEMROVE_MODELD_SOCK': str(Path(tmp) / 's')}):
            self.assertIsNone(emb.daemon_call({'op': 'ping'}, 0.2))
            with patch.object(emb.subprocess, 'Popen') as popen:
                emb.spawn_daemon()
                emb.spawn_daemon()
            self.assertEqual(popen.call_count, 1)

    def test_spawn_passes_idle_and_zero_means_resident(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MEMROVE_MODELD_SOCK': str(Path(tmp) / 's')}):
            with patch.object(emb.subprocess, 'Popen') as popen:
                emb.spawn_daemon(idle=0)
            argv = popen.call_args[0][0]
            self.assertEqual(argv[argv.index('--idle') + 1], '0')
        self.assertEqual(hyb.DEFAULTS['daemon_idle'], 600)


class FakeG2(FakeEmbedder):
    """Stands in for the EG2 daemon embedder: same name, so the cascade finds its vectors."""
    name = emb.DaemonEmbedder.name


class EmbedFormatTest(HybridBase):
    def test_eg2_documents_carry_title_uncapped_and_a_versioned_key(self):
        sent = []

        def fake(req, timeout=1.2):
            sent.append(req)
            return {'ok': True, 'vectors': [[1.0] + [0.0] * 767] * len(req['texts'])}
        long = 'pay invoice ' + 'x' * 1500
        rec = {'kind': 'decision', 'title': 'Billing rule', 'text': long}
        e = emb.DaemonEmbedder()
        with patch.object(emb, 'daemon_call', fake):
            e.embed(*[[long]], titles=['Billing rule'])
        self.assertEqual(sent[0]['titles'], ['Billing rule'])
        self.assertGreater(len(sent[0]['texts'][0]), emb.EMBED_CHARS)  # no 700-char cap for EG2
        self.assertTrue(e.name.endswith('-' + emb.DaemonEmbedder.FORMAT))
        title, text = emb.doc_input(rec, e)
        self.assertEqual(title, 'Billing rule')
        self.assertNotIn('Billing rule', text)
        self.assertEqual(len(emb.embed_text(rec)), emb.EMBED_CHARS)  # static keeps the cap

        class Other(emb.DaemonEmbedder):
            fmt = 'f3'
        self.assertNotEqual(emb.vec_key(e, title, text), emb.vec_key(Other(), title, text))  # a format change cannot reuse stale vectors
        self.assertEqual(emb.vec_key(FakeEmbedder(), None, 'abc'), emb._chash('abc'))  # static / ngram keys are unchanged


class ScoreTransformTest(HybridBase):
    def test_zscore_per_embedder_lex_w_and_delta(self):
        for i in range(4):
            self.add(f'r{i}', kind='decision', project='other', title=f'T{i}', text=f'pay invoice billing rule number {i}', origin_type='user_explicit')
        self.add('g', kind='decision', project='other', title='G', text='garden soil', origin_type='user_explicit')
        items = recall._load(recall.StateStore(self.state), None, {})
        cfg = dict(hyb.DEFAULTS, lex_w={'fake-3': 1.0}, znorm={'fake-3': True})
        top, _ = hyb.stage1('how do we pay the invoice billing', items, FakeEmbedder(), emb.VecStore(self.state), cfg)
        best = max(top, key=lambda c: c['s1'])
        self.assertAlmostEqual(best['s1'], best['z'] + 1.0 * best['pcover'], places=6)
        zs = hyb.zscores({c['id']: c['cos'] for c in top if c['cos'] is not None})
        self.assertAlmostEqual(sum(zs.values()), 0.0, places=6)
        plain, _ = hyb.stage1('how do we pay the invoice billing', items, FakeEmbedder(), emb.VecStore(self.state), dict(hyb.DEFAULTS))
        self.assertAlmostEqual(max(c['s1'] for c in plain), max(c['cos'] + 0.5 * c['pcover'] for c in plain), places=6)
        self.assertEqual(hyb.threshold(dict(hyb.DEFAULTS, delta1={'fake-3': 0.07}), 'delta1', 'fake-3'), 0.07)


class CascadeTest(HybridBase):
    PROMPT = 'how do we pay the invoice billing'

    def setUp(self):
        super().setUp()
        for i in range(3):
            self.add(f'r{i}', kind='decision', project='other', title=f'T{i}', text=f'pay invoice billing rule number {i}', origin_type='user_explicit')
        recs = [(rid, json.loads(p)) for p, _f, rid in recall._load(recall.StateStore(self.state), None, {})]
        emb.VecStore(self.state).refresh(recs, FakeG2())
        self.calls = []

    def conf(self, **kw):
        cas = {'band': [0.0, 99.0], 'budget': 0.25}
        (self.state / 'recall-config.json').write_text(json.dumps(dict({'cascade': cas, 'tau2': {FakeG2.name: 99.0}}, **kw)))

    def daemon(self, warm=True):
        def fake(req, timeout=1.2):
            self.calls.append(req['op'])
            if req['op'] == 'ping':
                return {'ok': True, 'warm': warm}
            return {'ok': True, 'vectors': [list(v) for v in FakeEmbedder().embed(req['texts'])]}
        return patch.object(emb, 'daemon_call', fake)

    def test_off_by_default_never_touches_the_daemon(self):
        with patch.object(emb, 'daemon_call', side_effect=AssertionError('no daemon')), patch.object(emb, 'spawn_daemon', side_effect=AssertionError('no spawn')):
            self.assertIn('decision', self.recall(self.PROMPT))

    def test_in_band_and_warm_l2_decides(self):
        self.conf()
        with self.daemon(), patch.object(emb, 'spawn_daemon', side_effect=AssertionError('never spawn')):
            self.assertEqual(self.recall(self.PROMPT), '')  # L1 alone would inject; L2's tau2 says abstain
        c = self.log()[-1]['counts']
        self.assertEqual((c['cascade'], c['score_key']), ('l2', 's2'))
        self.assertEqual(self.calls, ['ping', 'embed'])

    def test_cold_daemon_or_miss_keeps_the_l1_answer(self):
        self.conf()
        with self.daemon(warm=False):
            self.assertIn('decision', self.recall(self.PROMPT))
        self.assertEqual((self.log()[-1]['counts']['cascade'], self.calls), ('cold', ['ping']))  # no embed request: a cold model is never woken
        with patch.object(emb, 'daemon_call', lambda req, timeout=1.2: None), patch.object(emb, 'spawn_daemon') as spawn:
            self.assertIn('decision', self.recall(self.PROMPT, session='s2'))
        self.assertEqual(self.log()[-1]['counts']['cascade'], 'cold')
        spawn.assert_called_once()  # nothing listening: warm it for the next prompt, never wait for it

    def test_always_band_and_max_items_one(self):
        self.conf(cascade={'band': 'always', 'budget': 0.25}, tau2={FakeG2.name: -99.0}, delta2=99.0)
        with self.daemon():
            self.recall(self.PROMPT)
        self.assertGreater(len(self.log()[-1]['passed']), 1)  # default max_items 3
        self.conf(cascade={'band': 'always', 'budget': 0.25}, tau2={FakeG2.name: -99.0}, delta2=99.0, max_items=1)
        with self.daemon():
            self.recall(self.PROMPT, session='s2')
        c = self.log()[-1]
        self.assertEqual((c['counts']['cascade'], len(c['passed'])), ('l2', 1))

    def test_outside_the_band_l1_decides_without_a_call(self):
        self.conf(cascade={'band': [50.0, 99.0]})
        with patch.object(emb, 'daemon_call', side_effect=AssertionError('no daemon')):
            self.assertIn('decision', self.recall(self.PROMPT))
        self.assertEqual(self.log()[-1]['counts']['cascade'], 'l1')


if __name__ == '__main__':
    unittest.main()
