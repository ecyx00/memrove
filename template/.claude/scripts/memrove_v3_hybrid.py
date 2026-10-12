"""Model-based message-time recall: stage 1 (BM25 + dense, RRF top-20) -> optional stage 2 (reranker daemon) -> calibrated
threshold. Replaces the keyword gate of memrove_v3_recall (still available as MEMROVE_RECALL_MODE=gate and as the eval baseline).

Interface (stable, so the local `decide` model can plug in later without touching the rest):
    scorer.score(prompt_ctx, record) -> float        (batch form: scorer.score_many(prompt_ctx, [record, ...]))
    plus a calibrated threshold per scorer; an empty selection ("abstain") is a first-class result and is logged as such.

Stage 1 score  s1 = cosine(query, record) + lex_w * (idf-weighted share of the prompt's terms the record contains).
               With `znorm` the cosine is replaced by its per-query z-score over every eligible record (removes the per-query level shift);
               `lex_w`, `znorm`, `delta*` and `tau*` may each be a number or {embedder name: value}: the scales differ per embedder.
Cascade (cfg `cascade`, off by default): L1 = the configured embedder; when its best s1 lands inside `band` ("always" = every prompt) and the model daemon is already warm,
               L2 (EmbeddingGemma 2) re-scores the L1 candidates within `budget` seconds. The hook never spawns or waits for a cold model:
               a cold daemon, a miss or missing vectors leave the L1 answer.
Stage 2 score  s2 = reranker P(yes) in [0, 1]. Selection keeps up to MAX_ITEMS candidates with score >= tau and within `delta` of the best.
Everything fails open: no vectors / no daemon / slow daemon -> the stage below answers; nothing answers -> abstain."""
import json
import math
import os
import time
from pathlib import Path

RRF_K, TOP_K, LEX_W, NO_COS = 60, 20, 0.5, 0.25
MAX_ITEMS = 3
DEFAULTS = {"mode": "hybrid", "embedder": "auto", "rerank": False, "rerank_k": 10, "tau1": 0.5, "delta1": 0.2, "tau2": 0.5, "delta2": 0.25,
            "deadline": 1.2, "daemon_python": None, "daemon_idle": 600, "fill_budget": 0.3, "lex_w": LEX_W, "znorm": False, "cascade": False, "max_items": MAX_ITEMS}
ENV = {"MEMROVE_RECALL_MODE": ("mode", str), "MEMROVE_EMBEDDER": ("embedder", str), "MEMROVE_RERANK": ("rerank", lambda v: v == "1"),
       "MEMROVE_RECALL_TAU1": ("tau1", float), "MEMROVE_RECALL_TAU2": ("tau2", float)}


def config(state):
    """Defaults < <state>/recall-config.json < environment. tau1 may be a number or {embedder name: number} (the eval runner writes the latter)."""
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads((Path(state) / "recall-config.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    for key, (name, cast) in ENV.items():
        if os.environ.get(key):
            try:
                cfg[name] = cast(os.environ[key])
            except ValueError:
                pass
    return cfg


def option(cfg, name, embedder_name=None):
    """cfg[name], or its per-embedder entry when it is a dict (falls back to the default)."""
    v = cfg.get(name, DEFAULTS[name])
    return v.get(embedder_name, DEFAULTS[name]) if isinstance(v, dict) else v


def threshold(cfg, name, embedder_name=None):
    return float(option(cfg, name, embedder_name))


def s1_of(c, lex_w, znorm):
    """Stage-1 score of a candidate from its stored parts (the eval re-scores with other lex_w / znorm without re-embedding)."""
    base = c.get("z") if znorm else c.get("cos")
    if base is None:  # no vector yet: a neutral value, so the lexical half alone can still clear a threshold
        base = 0.0 if znorm else NO_COS
    return base + lex_w * c["pcover"]


def zscores(cos):
    """{id: cosine} -> {id: per-query z-score over all of them} (std floored: a flat query gets z = 0 everywhere)."""
    if not cos:
        return {}
    vals = list(cos.values())
    mean = sum(vals) / len(vals)
    std = max(1e-6, math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))
    return {rid: (c - mean) / std for rid, c in cos.items()}


# --- scorers ---------------------------------------------------------------------------------------------------------
class Stage1Scorer:
    """Dense cosine + lexical coverage of one record against the prompt (the batch path in `stage1` is the production one)."""

    def __init__(self, embedder, vectors=None):
        self.embedder, self.vectors = embedder, vectors or {}

    def score_many(self, ctx, records):
        from memrove_v3_embed import cosine, embed_text
        from memrove_v3_recall import terms
        q = getattr(self.embedder, "embed_query", self.embedder.embed)([ctx])[0]
        want = set(terms(ctx))
        out = []
        for r in records:
            v = self.vectors.get(r.get("id")) or self.embedder.embed([embed_text(r)])[0]
            have = set(terms(embed_text(r)))
            out.append(cosine(q, v) + LEX_W * (len(want & have) / max(1, len(want))))
        return out

    def score(self, ctx, record):
        return self.score_many(ctx, [record])[0]


class RerankScorer:
    """Stage-2 slot: a reranker behind the model daemon (none is configured or shipped; the daemon answers ok=false, so callers fall back). score_many -> None-free list, or raises OSError when the daemon does not answer in time."""

    def __init__(self, deadline=1.2):
        self.deadline = deadline

    def score_many(self, ctx, records):
        from memrove_v3_embed import daemon_call, embed_text
        reply = daemon_call({"op": "rerank", "query": ctx[:1500], "docs": [embed_text(r) for r in records]}, self.deadline)
        if not reply or len(reply.get("scores", [])) != len(records):
            raise OSError("rerank daemon unavailable")
        return [float(x) for x in reply["scores"]]

    def score(self, ctx, record):
        return self.score_many(ctx, [record])[0]


# --- stage 1 ---------------------------------------------------------------------------------------------------------
def stage1(prompt, items, embedder, vstore, cfg, dropped=None):
    """items: _load() rows. -> (candidates best-first by RRF, top TOP_K, info). Each candidate: dict(r, id, bm, pcover, cos, rrf, s1, user, hit).
    BM25 only proposes (no coverage / min-term gate); the dense side proposes records with no shared word at all."""
    from memrove_v3_embed import cosine, clip_query, embed_text
    from memrove_v3_recall import _rank, terms, lesson_admission, corroboration
    import json as _json
    info = {"dense": "off"}
    ranked = _rank(items, terms(prompt), gate=False, dropped=dropped, admit=True)
    cand = {s["r"]["id"]: dict(s, id=s["r"]["id"], cos=None) for s in ranked}
    payloads = {rid: payload for payload, _f, rid in items}
    cos = {}
    try:
        qv = getattr(embedder, "embed_query", embedder.embed)([clip_query(prompt)])[0]
        vecs = vstore.load(embedder.name)
        missing = [rid for rid in payloads if rid not in vecs]
        if missing:  # fresh store / just-synced records: embed on the fly, within a time budget; sync-time refresh normally got there first
            t0 = time.monotonic()
            recs = [(rid, _json.loads(payloads[rid])) for rid in missing]
            try:
                vstore.refresh(recs, embedder, prune=False, budget_s=cfg["fill_budget"])
                vecs = vstore.load(embedder.name)
            except Exception:
                pass
            info["fill_ms"] = round((time.monotonic() - t0) * 1000, 1)
        cos = {rid: cosine(qv, v) for rid, v in vecs.items() if rid in payloads}
        info["dense"] = "on" if cos else "no_vectors"
        info["vectors"], info["missing"] = len(cos), len(payloads) - len(cos)
    except Exception as exc:
        info["dense"] = "error:" + type(exc).__name__
    z = zscores(cos)
    for rid, c in cos.items():
        if rid in cand:
            cand[rid]["cos"], cand[rid]["z"] = c, z[rid]
    # dense proposals: top-K by cosine that BM25 did not already surface
    for rid, c in sorted(cos.items(), key=lambda kv: -kv[1])[:TOP_K]:
        if rid in cand:
            continue
        r = _json.loads(payloads[rid])
        if r.get("kind") == "lesson" and lesson_admission(r, corroboration([r])) != "admit":
            if dropped is not None:
                dropped.append({"id": rid, "reason": "lesson_not_admitted"})
            continue
        cand[rid] = {"r": r, "id": rid, "bm": 0.0, "cover": 0.0, "pcover": 0.0, "hit": set(), "cos": c, "z": z[rid], "score": 0.0,
                     "user": r.get("origin_type") == "user_explicit" or (not r.get("origin_type") and r.get("verified_by") == "user_quote")}
    by_bm = sorted((c for c in cand.values() if c["bm"] > 0), key=lambda c: -c["bm"])
    by_cos = sorted((c for c in cand.values() if c["cos"] is not None), key=lambda c: -c["cos"])
    for c in cand.values():
        c["rrf"] = 0.0
    for lst in (by_bm, by_cos):
        for rank, c in enumerate(lst):
            c["rrf"] += 1.0 / (RRF_K + rank + 1)
    top = sorted(cand.values(), key=lambda c: -c["rrf"])[:TOP_K]
    lex_w, znorm = float(option(cfg, "lex_w", embedder.name)), bool(option(cfg, "znorm", embedder.name))
    for c in top:
        c["s1"] = s1_of(c, lex_w, znorm)
    info["proposed"] = len(cand)
    return top, info


# --- stage 2 + decision ------------------------------------------------------------------------------------------------
def stage2(prompt, top, cfg):
    """Rerank the best `rerank_k` by s1 (sets c['s2']). -> info dict; a missed deadline leaves s2 unset and the caller uses stage 1."""
    from memrove_v3_embed import spawn_daemon
    t0 = time.monotonic()
    head = sorted(top, key=lambda c: -c["s1"])[:cfg["rerank_k"]]
    try:
        scores = RerankScorer(cfg["deadline"]).score_many(prompt, [c["r"] for c in head])
    except OSError:
        spawn_daemon(cfg.get("daemon_python"), cfg.get("daemon_idle", 600))  # cold daemon: start it for the next prompt, answer this one from stage 1; daemon_idle 0 = resident (cascade L2)
        return {"stage2": "fallback", "ms2": round((time.monotonic() - t0) * 1000, 1)}
    for c, s in zip(head, scores):
        c["s2"] = s
    return {"stage2": "on", "ms2": round((time.monotonic() - t0) * 1000, 1)}


def decide(top, cfg, embedder_name):
    """-> (selected best-first, why-not dict by id). Uses s2 when every one of the reranked candidates has it, else s1."""
    use2 = any("s2" in c for c in top)
    key, tau, delta = ("s2", threshold(cfg, "tau2", embedder_name), threshold(cfg, "delta2", embedder_name)) if use2 else ("s1", threshold(cfg, "tau1", embedder_name), threshold(cfg, "delta1", embedder_name))
    pool = sorted((c for c in top if key in c), key=lambda c: -c[key])
    best = pool[0][key] if pool else 0.0
    chosen, why = [], {}
    for c in pool:
        if c[key] < tau:
            why[c["id"]] = f"below_{key}"
        elif c[key] < best - delta:
            why[c["id"]] = "margin"
        elif len(chosen) >= int(cfg.get("max_items") or MAX_ITEMS):
            why[c["id"]] = "budget"
        else:
            chosen.append(c)
    return chosen, why, key


# --- cascade (L2) ---------------------------------------------------------------------------------------------------------
def _cas(cfg):
    return cfg["cascade"] if isinstance(cfg["cascade"], dict) else {}


def in_band(top, cfg, l1_name):
    """L1 is uncertain when its best s1 falls inside cfg['cascade']['band'] = [lo, hi) (default: tau1 +- 0.1)."""
    if not top:
        return False
    if _cas(cfg).get("band") == "always":
        return True
    tau = threshold(cfg, "tau1", l1_name)
    lo, hi = _cas(cfg).get("band") or (tau - 0.1, tau + 0.1)
    return lo <= max(c["s1"] for c in top) < hi


def l2_rescore(top, qv, vecs, ids, cfg, l2_name):
    """Pure part of L2: score each candidate with the L2 vectors (sets cos2, z2, s2 on those that have one). -> number of candidates scored."""
    from memrove_v3_embed import cosine
    cos = {rid: cosine(qv, v) for rid, v in vecs.items() if rid in ids}
    z = zscores(cos)
    lex_w, znorm = float(option(cfg, "lex_w", l2_name)), bool(option(cfg, "znorm", l2_name))
    n = 0
    for c in top:
        if c["id"] in cos:
            c["cos2"], c["z2"] = cos[c["id"]], z[c["id"]]
            c["s2"] = (c["z2"] if znorm else c["cos2"]) + lex_w * c["pcover"]
            n += 1
    return n


def cascade_l2(prompt, top, cfg, vstore, ids):
    """Hook-side L2. Never spawns and never waits for a cold model: the daemon must answer a ping saying the model is loaded,
    and the query embedding must come back within cascade.budget seconds. -> (info, used) ; used=False leaves the L1 answer."""
    from memrove_v3_embed import DaemonEmbedder, clip_query, daemon_call
    t0 = time.monotonic()
    budget = float(_cas(cfg).get("budget", 0.25))
    ms = lambda: round((time.monotonic() - t0) * 1000, 1)
    ping = daemon_call({"op": "ping"}, min(budget, 0.1))
    if not ping or not ping.get("warm"):
        if not ping:  # nothing listening: start it for the next prompt (throttled to one spawn a minute); this prompt keeps the L1 answer
            from memrove_v3_embed import spawn_daemon
            spawn_daemon(cfg.get("daemon_python"), cfg.get("daemon_idle", 600))
        return {"cascade": "cold", "ms2": ms()}, None
    e2 = DaemonEmbedder(timeout=max(0.05, budget - (time.monotonic() - t0)))
    try:
        qv = e2.embed_query([clip_query(prompt)])[0]
    except (OSError, KeyError, ValueError):
        return {"cascade": "miss", "ms2": ms()}, None
    if not l2_rescore(top, qv, vstore.load(e2.name), ids, cfg, e2.name):
        return {"cascade": "no_vectors", "ms2": ms()}, None
    return {"cascade": "l2", "ms2": ms()}, e2.name


def select(store, prompt, here, done, cfg, stats, dropped):
    """The hybrid replacement of the keyword gate. -> dict(scored, log, abstain, info); scored = chosen candidates in injection order,
    already minus those injected earlier this session."""
    from memrove_v3_embed import VecStore, get_embedder, NgramEmbedder
    from memrove_v3_recall import _load
    t0 = time.monotonic()
    items = _load(store, None, stats)  # all projects: the cwd project's records are valuable; SessionStart overlap is removed via `done` below
    info = {"mode": "hybrid"}
    try:
        embedder = get_embedder(cfg["embedder"])
    except Exception:
        embedder = NgramEmbedder()
    info["embedder"] = embedder.name
    top, i1 = stage1(prompt, items, embedder, VecStore(store.state_dir), cfg, dropped)
    info.update(i1)
    seen = lambda i: "recall:" + i in done or i in done  # a bare id was put there by SessionStart (RULES, THREADS, LAST SESSION, decisions)
    top_fresh = [c for c in top if not seen(c["id"])]
    dropped += [{"id": c["id"], "reason": "already_injected"} for c in top if seen(c["id"])]
    if cfg["rerank"] and top_fresh:
        # the hook has a 2 s timeout: stage 2 gets what is left of a 1.5 s total, never more than cfg deadline
        info.update(stage2(prompt, top_fresh, dict(cfg, deadline=min(cfg["deadline"], max(0.2, 1.5 - (time.monotonic() - t0))))))
    chosen, why, key = decide(top_fresh, cfg, embedder.name)
    if cfg["cascade"] and not cfg["rerank"] and top_fresh:
        try:
            if in_band(top_fresh, cfg, embedder.name):
                i2, l2_name = cascade_l2(prompt, top_fresh, cfg, VecStore(store.state_dir), {rid for _p, _f, rid in items})
                info.update(i2)
                if l2_name:
                    chosen, why, key = decide(top_fresh, cfg, l2_name)
            else:
                info["cascade"] = "l1"
        except Exception as exc:  # fail open: the L1 answer stands
            info["cascade"] = "error:" + type(exc).__name__
    info["score_key"] = key
    log = [{"id": c["id"], "score": round(c.get(key, c["s1"]), 4), "s1": round(c["s1"], 4), "s2": None if c.get("s2") is None else round(c["s2"], 4),
            "cos": None if c["cos"] is None else round(c["cos"], 4), "bm": round(c["bm"], 3), "pcover": round(c["pcover"], 3),
            "reason": why.get(c["id"], "selected")} for c in top_fresh]
    return {"scored": chosen, "log": log, "abstain": not chosen, "info": info}
