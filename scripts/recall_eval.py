#!/usr/bin/env python3
"""Recall eval: the same gold prompts through the legacy keyword gate and the model-based variants, one table.

    PYTHONPATH=template/.claude/scripts nice lockf /tmp/heavy-bench.lock python3 scripts/recall_eval.py \
        --gold ~/.local/share/memrove/evals/recall_gold.jsonl --records /tmp/mem.sqlite3 [--hook-latency 25] [--out table.md]

Gold row: prompt, ts, session, cwd_project, candidates[{record_id, label relevant|partly|not, maybe_origin}], nothing_relevant.
Records: a memory.sqlite3 snapshot (table records(id, payload)). Every prompt is run against ALL records eligible at its time (the gold's
time rule), not just its labelled pool; an injected record with no label counts as not relevant (so precision is a lower bound).
Thresholds are calibrated on the first day's prompts (--calib-day) and the table reports the later days as the held-out slice.
Variants: gate, gate-prod (also excludes the cwd project), s1 per embedder (s1-ngram, s1-static, s1-gemma = EmbeddingGemma 2 daemon); s2-<embedder> exists for a future reranker (none is configured).
Metrics: precision@injected (strict; lenient = partly counts half), recall of relevant pairs, abstain accuracy on nothing-relevant prompts,
injected per prompt, pipeline latency p50/p95, and (with --hook-latency N) real UserPromptSubmit hook subprocess p50/p95 and peak RSS."""
import argparse
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_embed as emb
import memrove_v3_hybrid as hyb
import memrove_v3_recall as rc
import recall_fit as fit

MAX_ITEMS = 3
USE_CTX = True
FALSE_COST = 0.25  # calibration objective: +1 relevant, +0.5 partly, -FALSE_COST per other injected record
LOAD_MAX = 8.0


def q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))] if vals else float("nan")


def load_records(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out = {}
    for rid, payload in db.execute("SELECT id, payload FROM records"):
        r = json.loads(payload)
        if r.get("status") not in ("active", None) or r.get("kind") not in rc.KINDS:
            continue
        out[rid] = {"payload": payload, "r": r, "folded": rc.fold(json.dumps(r, ensure_ascii=False) if "\\u" in payload else payload)}
    return out


def eligible(rec, gold_row):
    """The gold builder's time rule: nothing that may postdate the prompt, nothing from this very session-day, no origin record."""
    r, day, prompt = rec["r"], gold_row["ts"][:10], gold_row["prompt"]
    vf = r.get("valid_from")
    vf = str(vf)[:10] if vf else None
    sess = {r.get("session")} | set(r.get("seen_sessions") or [])
    if vf and vf > day:
        return False
    if vf == day and gold_row.get("session") in sess:
        return False
    text = " ".join(str(r.get(k) or "") for k in ("text", "rule", "why", "situation", "approach")).strip()
    m = re.match(r"# Session note (\d{4}-\d\d-\d\d)", text)
    if m and m.group(1) >= day:
        return False
    if vf is None:
        ds = re.findall(r"20\d\d-\d\d-\d\d", text[:600])
        if ds and max(ds) >= day:
            return False
    first = rc.fold(re.split(r"[.\n]", text)[0]).strip()
    return not (len(first) >= 15 and first in rc.fold(prompt))


def same_project(gold_row, r):
    p, c = r.get("project") or "", gold_row.get("cwd_project") or ""
    return bool(p) and (c == p or c.endswith("-" + p))


def labels_of(row):
    lab, skip = {}, set()
    for c in row["candidates"]:
        if c.get("maybe_origin") and c["label"] == "relevant":
            skip.add(c["record_id"])
        else:
            lab[c["record_id"]] = c["label"]
    return lab, skip


def metrics(runs, rows_by_id, slice_ids):
    inj = rel = part = unl = n_inj = n_rel_total = n_nothing = n_abs = n_nothing2 = n_abs2 = 0
    for pid in slice_ids:
        row, ids = rows_by_id[pid], runs[pid]["ids"]
        lab, skip = labels_of(row)
        ids = [i for i in ids if i not in skip]
        n_inj += len(ids)
        for i in ids:
            l = lab.get(i)
            rel += l == "relevant"
            part += l == "partly"
            unl += l is None
        n_rel_total += sum(1 for v in lab.values() if v == "relevant")
        if row["nothing_relevant"]:
            n_nothing += 1
            n_abs += not ids
        if row["nothing_relevant_or_partly"]:
            n_nothing2 += 1
            n_abs2 += not ids
    hit = sum(1 for pid in slice_ids for i in runs[pid]["ids"] if labels_of(rows_by_id[pid])[0].get(i) == "relevant")
    return {"n": len(slice_ids), "inj_per_prompt": n_inj / max(1, len(slice_ids)),
            "p_strict": rel / n_inj if n_inj else float("nan"), "p_lenient": (rel + 0.5 * part) / n_inj if n_inj else float("nan"),
            "recall": hit / max(1, n_rel_total), "abstain": n_abs / max(1, n_nothing), "abstain_incl_partly": n_abs2 / max(1, n_nothing2),
            "unlabeled_inj": unl, "n_inj": n_inj, "lat": [runs[p]["ms"] for p in slice_ids]}


def utility(runs, rows_by_id, slice_ids):
    u = 0.0
    for pid in slice_ids:
        lab, skip = labels_of(rows_by_id[pid])
        for i in runs[pid]["ids"]:
            if i in skip:
                continue
            u += {"relevant": 1.0, "partly": 0.5}.get(lab.get(i), -FALSE_COST)
    return u


# --- variants --------------------------------------------------------------------------------------------------------
def run_gate(rows, recs, exclude_here):
    runs = {}
    for row in rows:
        t0 = time.perf_counter()
        items = [(v["payload"], v["folded"], rid) for rid, v in recs.items() if eligible(v, row) and not (exclude_here and same_project(row, v["r"]))]
        terms = rc.terms(row["prompt"])
        ids = []
        if len(set(terms)) >= rc.MIN_WORDS and items:
            ids = [s["r"]["id"] for s in rc._rank(items, terms, gate=True)[:MAX_ITEMS]]
        runs[row["prompt_id"]] = {"ids": ids, "ms": (time.perf_counter() - t0) * 1000}
    return runs


def stage_tops(rows, recs, embedder, state, rerank, rerank_k, daemon_python):
    """Stage 1 (and 2) once per prompt -> {pid: {"top": [...], "ms": float}}; thresholds are applied afterwards."""
    emb.VecStore(state).refresh([(rid, v["r"]) for rid, v in recs.items()], embedder)
    cfg = dict(hyb.DEFAULTS, rerank_k=rerank_k, daemon_python=daemon_python)
    out = {}
    for row in rows:
        if len(out) % 25 == 24:
            cool(LOAD_MAX)  # short pauses between batches keep the fanless Air cool
            time.sleep(1)
        t0 = time.perf_counter()
        items = [(v["payload"], v["folded"], rid) for rid, v in recs.items() if eligible(v, row)]
        query = row["prompt"] + ("\n" + row["prev_assistant_tail"] if USE_CTX and row.get("prev_assistant_tail") else "")
        top, _ = hyb.stage1(query, items, embedder, emb.VecStore(state), cfg)
        ok = None
        if rerank:
            ok = hyb.stage2(row["prompt"], top, cfg)["stage2"] == "on"
        out[row["prompt_id"]] = {"top": top, "ms": (time.perf_counter() - t0) * 1000, "s2_ok": ok, "ids": {rid for _p, _f, rid in items}, "query": query}
    return out


def apply(tops, cfg, name):
    return {pid: {"ids": [c["id"] for c in hyb.decide(t["top"], cfg, name)[0]], "ms": t["ms"]} for pid, t in tops.items()}


def calibrate(tops, rows_by_id, calib_ids, rerank):
    key = "s2" if rerank else "s1"
    vals = sorted({round(c[key], 3) for pid in calib_ids for c in tops[pid]["top"] if key in c})
    grid = vals[:: max(1, len(vals) // 150)] + [vals[-1] + 1e-3] if vals else [1.0]
    best = None
    for tau in grid:
        for delta in (0.05, 0.1, 0.2, 0.3, 1.0):
            cfg = {"tau1": tau, "delta1": delta, "tau2": tau, "delta2": delta}
            u = utility(apply({p: tops[p] for p in calib_ids}, cfg, "x"), rows_by_id, calib_ids)
            if best is None or u > best[0] + 1e-9:
                best = (u, tau, delta)
    return best


def daemon_up(python):
    if emb.daemon_call({"op": "ping"}, 1.0):
        return True
    emb.spawn_daemon(python, idle=1800)
    for _ in range(60):
        time.sleep(1)
        if emb.daemon_call({"op": "ping"}, 1.0):
            return True
    return False


def rss_of_daemon():
    r = emb.daemon_call({"op": "ping"}, 1.0)
    if not r:
        return None
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(r["pid"])], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out else None


def hook_latency(records_db, cfg_file, env_extra, prompts, n, transcript=False):
    """Real UserPromptSubmit hook subprocesses against a copy of the snapshot as a state dir. -> (p50 ms, p95 ms, peak RSS MB, empty share)."""
    with tempfile.TemporaryDirectory() as tmp:
        state, vault, repo = Path(tmp, "state"), Path(tmp, "vault"), Path(tmp, "repo")
        for d in (state, vault, repo / ".git"):
            d.mkdir(parents=True)
        shutil.copy(records_db, state / "memory.sqlite3")
        for name in ("vectors.sqlite3",):
            if cfg_file.get("_vectors"):
                shutil.copy(cfg_file["_vectors"] / name, state / name)
        (state / "recall-config.json").write_text(json.dumps({k: v for k, v in cfg_file.items() if not k.startswith("_")}))
        ms, rss, empty = [], 0.0, 0
        script = ROOT / "template/.claude/scripts/memrove_v3_hook.py"
        for i, p in enumerate(prompts[:n]):
            body = {"hook_event_name": "UserPromptSubmit", "prompt": p, "session_id": f"e{i}", "cwd": str(repo)}
            if transcript:  # ~250 KB transcript: the hook must read only the last 32 KB
                tr = Path(tmp, "transcript.jsonl")
                if not tr.exists():
                    turn = {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Stripe webhook retries: dedupe by event id. " * 12}]}}
                    tr.write_text((json.dumps(turn) + "\n") * 500, encoding="utf-8")
                body["transcript_path"] = str(tr)
            payload = json.dumps(body)
            t0 = time.perf_counter()
            r = subprocess.run(["/usr/bin/time", "-l", sys.executable, str(script), "--vault", str(vault), "--state", str(state), "--harness", "claude", "--global"],
                               input=payload, text=True, capture_output=True, cwd=repo, env=dict(os.environ, HOME=tmp, MEMROVE_V3_NO_SPAWN="1", MEMROVE_NO_DECIDE_LOG="1", MEMROVE_MODELS=str(emb.models_dir()), MEMROVE_MODELD_SOCK=emb.sock_path(), **env_extra))
            ms.append((time.perf_counter() - t0) * 1000)
            m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
            rss = max(rss, int(m.group(1)) / 1e6) if m else rss
            empty += r.stdout.strip() in ("", "{}")
        return q(ms, 0.5), q(ms, 0.95), rss, empty / max(1, len(ms))


def cool(limit):
    """Fanless laptop: wait while the 1-minute load average is above `limit`."""
    while os.getloadavg()[0] > limit:
        print(f"load {os.getloadavg()[0]:.1f} > {limit}, pausing", file=sys.stderr)
        time.sleep(15)


def embed_cost(embedder, recs, n=20):
    """ms per record (document side) and per query for one embedder, warm."""
    texts = [emb.embed_text(v["r"]) for v in list(recs.values())[:n]]
    # Synthetic Turkish prompts (stripe webhook retry how-to; token refresh error at login) for the Turkish recall path.
    queries = ["stripe webhook retry nasıl yapılır", "kullanıcı girişinde token yenileme hatası"] * (n // 2)
    embedder.embed(texts[:2])
    t0 = time.perf_counter()
    for t in texts:
        embedder.embed([t])
    doc = (time.perf_counter() - t0) / len(texts) * 1000
    qe = getattr(embedder, "embed_query", embedder.embed)
    t0 = time.perf_counter()
    for t in queries:
        qe([t])
    return doc, (time.perf_counter() - t0) / len(queries) * 1000


def fmt(x, d=2):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


class MemoQuery:
    """Embedder wrapper that caches (and persists) query vectors; documents pass straight through, so the vector store does the doc caching."""

    def __init__(self, inner, path):
        self.inner, self.path = inner, Path(path)
        try:
            self.cache = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.cache = {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def embed(self, texts, **kw):
        return self.inner.embed(texts, **kw)

    def embed_query(self, texts):
        import array
        miss = [t for t in texts if t not in self.cache]
        if miss:
            for t, v in zip(miss, self.inner.embed_query(miss)):
                self.cache[t] = list(v)
            self.path.write_text(json.dumps(self.cache))
        return [array.array("f", self.cache[t]) for t in texts]


def merge_extra(rows_by_id, path):
    """Extra labels (same candidate layout as the gold, label_source e31-pool) are appended to their prompt's candidates."""
    n = 0
    for line in open(path, encoding="utf-8"):
        if line.strip():
            x = json.loads(line)
            have = {c["record_id"] for c in rows_by_id[x["prompt_id"]]["candidates"]}
            new = [c for c in x["candidates"] if c["record_id"] not in have]
            rows_by_id[x["prompt_id"]]["candidates"] += new
            n += len(new)
    print(f"merged {n} extra labels from {path}", file=sys.stderr)


def write_cascade_config(a, tops, pids, l2_name, l1_name):
    """tau2 = the score of the k-th best L2 top-1 over `pids` (k = coverage * n), the same key as recall_fit.k2(False, 0.5): the cross-fit rule, fitted on all prompts."""
    key = fit.k2(False, hyb.DEFAULTS["lex_w"])
    best = sorted((r[0][0] for r in (fit.rank(tops[p]["top"], key) for p in pids) if r), reverse=True)
    tau2 = round(best[max(0, int(round(a.coverage * len(pids))) - 1)], 4)
    cfg = {"mode": "hybrid", "embedder": "static", "rerank": False, "tau1": {l1_name: 0.607}, "delta1": 0.05, "daemon_python": a.daemon_python,
           "daemon_idle": 0, "max_items": 1, "cascade": {"band": "always", "budget": 0.25}, "tau2": {l2_name: tau2}, "delta2": {l2_name: hyb.DEFAULTS["delta2"]}}
    Path(a.write_cascade_config).write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote {a.write_cascade_config}: tau2={tau2} (top-{int(round(a.coverage * len(pids)))} of {len(pids)} prompts)", file=sys.stderr)


def fair_main(a, rows, rows_by_id, recs, calib, held):
    """Static vs EG2 vs cascade on one fair label pool; cross-fit only; thresholds by Learn-then-Test. Needs the model daemon (EG2)."""
    if not daemon_up(a.daemon_python):
        sys.exit("model daemon did not start")
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="recall-fair-"))
    for d in ("static", "gemma"):
        (work / d).mkdir(parents=True, exist_ok=True)
    g = MemoQuery(emb.DaemonEmbedder(timeout=5), work / "gemma_queries.json")
    st = emb.StaticEmbedder()
    allp = [r["prompt_id"] for r in rows]
    tops_s = stage_tops(rows, recs, st, work / "static", False, a.rerank_k, a.daemon_python)
    tops_g = stage_tops(rows, recs, g, work / "gemma", False, a.rerank_k, a.daemon_python)
    vg = emb.VecStore(work / "gemma").load(g.name)
    for pid in allp:  # L2 scores for the L1 candidates, once; the fit picks z / lex_w later
        t = tops_s[pid]
        hyb.l2_rescore(t["top"], g.embed_query([emb.clip_query(t["query"])])[0], vg, t["ids"], hyb.DEFAULTS, g.name)
    if a.write_cascade_config:
        write_cascade_config(a, tops_s, allp, g.name, st.name)
        stop_daemon()
        return
    pools = {"static": fit.pool_pairs(tops_s, allp), "eg2": fit.pool_pairs(tops_g, allp), "cascade": fit.pool_pairs(tops_s, allp, with_l2=True)}
    labelled = {(r["prompt_id"], c["record_id"]) for r in rows for c in r["candidates"]}
    missing = {k: sorted(v - labelled) for k, v in pools.items()}
    lines = ["| arm | pairs any setting could inject | unlabelled |", "|---|---|---|"]
    lines += [f"| {k} | {len(pools[k])} | {len(missing[k])} |" for k in pools]
    if a.emit_pool:
        todo = {}
        for k, pairs in missing.items():
            for pid, rid in pairs:
                todo.setdefault(pid, {}).setdefault(rid, []).append(k)
        with open(a.emit_pool, "w", encoding="utf-8") as f:
            for pid in sorted(todo):
                row = rows_by_id[pid]
                pairs = []
                for rid, arms in todo[pid].items():
                    r = recs[rid]["r"]
                    pairs.append({"record_id": rid, "kind": r.get("kind"), "record_project": r.get("project"), "valid_from": r.get("valid_from"),
                                  "sources": arms, "title": r.get("title"), "body": rc._body(r)[:900]})
                f.write(json.dumps({"prompt_id": pid, "ts": row["ts"], "prompt": row["prompt"], "prev_assistant_tail": row.get("prev_assistant_tail", ""),
                                    "cwd_project": row.get("cwd_project"), "pairs": pairs}, ensure_ascii=False) + "\n")
        print("\n".join(lines))
        print(f"wrote {sum(len(v) for v in todo.values())} unlabelled pairs over {len(todo)} prompts to {a.emit_pool}")
        stop_daemon()
        return
    judge = lambda pid: labels_of(rows_by_id[pid])
    arms = {"static": (tops_s, fit.fit_single, fit.run_single), "eg2": (tops_g, fit.fit_single, fit.run_single), "cascade": (tops_s, fit.fit_cascade, fit.run_cascade)}
    head = ("| arm | alpha | direction | n | fitted | inj/prompt | precision (strict) | precision (lenient) | recall | coverage | abstain (nothing relevant) | unlabelled inj |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = []
    for alpha in [float(x) for x in a.alphas.split(",")]:
        for arm, (tops, fitter, runner) in arms.items():
            f_a, f_b = fitter(tops, calib, judge, alpha), fitter(tops, held, judge, alpha)  # fit on one day, test on the other
            ra, rb = runner(tops, held, f_a), runner(tops, calib, f_b)
            for name, ids, runs, fp in (("fit Oct 4 -> test Oct 5-6", held, ra, f_a), ("fit Oct 5-6 -> test Oct 4", calib, rb, f_b), ("pooled cross-fit", allp, {**ra, **rb}, None)):
                m = metrics({p: {"ids": v, "ms": 0.0} for p, v in runs.items()}, rows_by_id, ids)
                cov = sum(1 for p in ids if runs[p]) / max(1, len(ids))
                desc = "" if fp is None and name.startswith("pooled") else ("no valid setting: abstains" if not fp or (arm == "cascade" and not fp["l1"] and not fp["l2"]) else json.dumps(fp, default=str)[:120])
                out.append(f"| {arm} | {alpha} | {name} | {m['n']} | {desc} | {fmt(m['inj_per_prompt'])} | {fmt(m['p_strict'])} | {fmt(m['p_lenient'])} | {fmt(m['recall'])} | "
                           f"{fmt(cov)} | {fmt(m['abstain'])} | {m['unlabeled_inj']}/{m['n_inj']} |")
    text = "\n".join(lines + ["", head] + out)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    stop_daemon()


def stop_daemon():
    """The eval started the daemon (idle 1800 s); do not leave EG2 resident."""
    import signal
    r = emb.daemon_call({"op": "ping"}, 1.0)
    if r:
        os.kill(r["pid"], signal.SIGTERM)
        time.sleep(0.5)
        Path(emb.sock_path()).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path.home() / ".local/share/memrove/evals/recall_gold.jsonl"))
    ap.add_argument("--records", default="/tmp/mem.sqlite3")
    ap.add_argument("--calib-day", default="2026-10-04")
    ap.add_argument("--variants", default="gate,gate-prod,s1-ngram,s1-static,s1-gemma")
    ap.add_argument("--rerank-k", type=int, default=10)
    ap.add_argument("--daemon-python", default=str(Path.home() / ".local/share/memrove/venv/bin/python"))
    ap.add_argument("--hook-latency", type=int, default=0, help="N real hook runs per variant (0 = skip)")
    ap.add_argument("--write-config", help="write the calibrated thresholds of this variant (e.g. s1-static) into a recall-config.json path")
    ap.add_argument("--load-max", type=float, default=8.0, help="pause while the 1-min load average is above this")
    ap.add_argument("--false-cost", type=float, default=0.25)
    ap.add_argument("--no-ctx", action="store_true", help="leave the previous assistant turn out of the query")
    ap.add_argument("--ctx-latency", action="store_true", help="also time the hook with short prompts, with and without a transcript")
    ap.add_argument("--fair", action="store_true", help="cascade eval: static / EG2 / cascade arms, label-pool check, Learn-then-Test cross-fit tables (loads EG2: heavy)")
    ap.add_argument("--emit-pool", help="with --fair: write the unlabelled pairs any arm could inject to this jsonl for labelling, then stop")
    ap.add_argument("--gold-extra", help="with --fair: jsonl of extra labels (label_source e31-pool) merged into the gold")
    ap.add_argument("--work", help="with --fair: persistent work dir (vectors + query vectors), so a re-run embeds nothing twice")
    ap.add_argument("--write-cascade-config", help="with --fair: write the decided default (cascade L2 on every prompt, 1 item, tau2 at --coverage of the prompts) to this recall-config.json path, then stop")
    ap.add_argument("--coverage", type=float, default=0.1)
    ap.add_argument("--alphas", default="0.3,0.5")
    ap.add_argument("--out")
    a = ap.parse_args()
    global LOAD_MAX, USE_CTX, FALSE_COST
    FALSE_COST = a.false_cost
    USE_CTX = not a.no_ctx
    LOAD_MAX = a.load_max
    if not Path(a.gold).is_file():
        sys.exit(f"gold set not found: {a.gold}")
    rows = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    rows_by_id = {r["prompt_id"]: r for r in rows}
    recs = load_records(a.records)
    if a.fair and a.gold_extra:
        merge_extra(rows_by_id, a.gold_extra)
    calib = [r["prompt_id"] for r in rows if r["ts"][:10] <= a.calib_day]
    held = [r["prompt_id"] for r in rows if r["ts"][:10] > a.calib_day]
    reach = sum(1 for r in rows for c in r["candidates"] if c["label"] == "relevant" and c["record_id"] in recs)
    total = sum(1 for r in rows for c in r["candidates"] if c["label"] == "relevant")
    print(f"gold {len(rows)} prompts ({len(calib)} calibration, {len(held)} held-out); records in recall kinds: {len(recs)}; relevant pairs reachable {reach}/{total}", file=sys.stderr)
    if a.fair:
        return fair_main(a, rows, rows_by_id, recs, calib, held)
    table, notes, cost = [], [], {}
    work = Path(tempfile.mkdtemp(prefix="recall-eval-"))
    tmp_vec = {}
    for v in a.variants.split(","):
        cool(a.load_max)
        t0 = time.time()
        daemon_rss, cfgfile, env = None, {"mode": "hybrid"}, {}
        if v.startswith("gate"):
            runs, label, tau = run_gate(rows, recs, v == "gate-prod"), v, ""
            slices = {"all": runs, "cross-fit": runs, "held-out": runs}
            cfgfile, env = {"mode": "gate"}, {"MEMROVE_RECALL_MODE": "gate"}
        else:
            kind, emb_name = v.split("-")
            if emb_name == "gemma":
                if not daemon_up(a.daemon_python):
                    notes.append(f"{v}: skipped, model daemon did not start")
                    continue
                embedder = emb.DaemonEmbedder(timeout=5)
            elif emb_name == "static":
                if not emb.StaticEmbedder.installed():
                    notes.append(f"{v}: skipped, static model not installed (python -c 'import memrove_v3_embed as e; e.install_static()')")
                    continue
                embedder = emb.StaticEmbedder()
            else:
                embedder = emb.NgramEmbedder()
            rerank = kind == "s2"
            if rerank and not daemon_up(a.daemon_python):
                notes.append(f"{v}: skipped, model daemon did not start")
                continue
            state = work / v
            state.mkdir()
            if not rerank:
                cost[v] = embed_cost(embedder, recs)
            try:
                tops = stage_tops(rows, recs, embedder, state, rerank, a.rerank_k, a.daemon_python)
            except OSError as exc:
                notes.append(f"{v}: skipped ({exc})")
                continue
            if rerank:
                notes.append(f"{v}: reranker answered within the deadline for {sum(1 for t in tops.values() if t['s2_ok'])}/{len(tops)} prompts (rest fell back to stage 1), rerank_k={a.rerank_k}")
            allp = [r["prompt_id"] for r in rows]
            u, tau_v, delta = calibrate(tops, rows_by_id, allp, rerank)  # shipped setting: fit on every gold prompt
            cfg = {"tau1": tau_v, "delta1": delta, "tau2": tau_v, "delta2": delta}
            fit = lambda ids: (lambda b: {"tau1": b[1], "delta1": b[2], "tau2": b[1], "delta2": b[2]})(calibrate(tops, rows_by_id, ids, rerank))
            cfg_a, cfg_b = fit(calib), fit(held)  # honest numbers: fit on one day, apply to the other
            ra, rb_ = apply({p: tops[p] for p in held}, cfg_a, embedder.name), apply({p: tops[p] for p in calib}, cfg_b, embedder.name)
            runs = apply(tops, cfg, embedder.name)
            slices = {"all": runs, "cross-fit": {**ra, **rb_}, "held-out": ra}
            tau = f" tau={tau_v:.3f} delta={delta}"
            label = v
            if rerank and any("s2" not in c for t in tops.values() for c in t["top"][:1]):
                pass
            cfgfile = {"mode": "hybrid", "embedder": {"gemma": "daemon-gemma"}.get(emb_name, emb_name), "rerank": rerank, "rerank_k": a.rerank_k,
                       ("tau2" if rerank else "tau1"): tau_v, ("delta2" if rerank else "delta1"): delta, "daemon_python": a.daemon_python, "_vectors": state}
            env = {"MEMROVE_EMBEDDER": cfgfile["embedder"]}
            if a.write_config and a.write_config.split(":")[0] == v:
                Path(a.write_config.split(":")[1]).write_text(json.dumps({k: (({embedder.name: val}) if k == "tau1" else val) for k, val in cfgfile.items() if not k.startswith("_")}, indent=1))
            if emb_name == "gemma" or rerank:
                daemon_rss = rss_of_daemon()
        hook = None
        if a.hook_latency:
            hook = hook_latency(a.records, cfgfile, env, [r["prompt"] for r in rows if r["ts"][:10] > a.calib_day], a.hook_latency)
            if a.ctx_latency:
                short = [r["prompt"][:40] for r in rows if r["ts"][:10] > a.calib_day]
                for tr in (False, True):
                    h2 = hook_latency(a.records, cfgfile, env, short, a.hook_latency, transcript=tr)
                    notes.append(f"{v} hook, 40-char prompts, {'with' if tr else 'without'} 250 KB transcript: p50/p95 {h2[0]:.0f}/{h2[1]:.0f} ms")
            if daemon_rss is None and (cfgfile.get("rerank") or cfgfile.get("embedder") == "daemon"):
                daemon_rss = rss_of_daemon()
        for name, ids in (("all", [r["prompt_id"] for r in rows]), ("cross-fit", [r["prompt_id"] for r in rows]), ("held-out", held)):
            m = metrics(slices[name], rows_by_id, ids)
            table.append((label + tau, {"all": "all (fit on all, in-sample)", "cross-fit": "cross-fit (fit on the other day)", "held-out": "held-out Oct 5-6 (fit Oct 4)"}[name], m, hook if name == "all" else None, daemon_rss))
        print(f"{v} done in {time.time() - t0:.0f}s", file=sys.stderr)
    head = ("| variant | slice | n | inj/prompt | P@inj strict | P@inj lenient | recall | abstain (strict) | abstain (incl partly) | unlabeled inj | pipeline ms p50/p95 | hook ms p50/p95 | hook peak RSS MB | daemon RSS MB |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for label, name, m, hook, drss in table:
        h = f"{hook[0]:.0f}/{hook[1]:.0f}" if hook else ""
        lines.append(f"| {label} | {name} | {m['n']} | {fmt(m['inj_per_prompt'])} | {fmt(m['p_strict'])} | {fmt(m['p_lenient'])} | {fmt(m['recall'])} | {fmt(m['abstain'])} | "
                     f"{fmt(m['abstain_incl_partly'])} | {m['unlabeled_inj']}/{m['n_inj']} | {q(m['lat'], .5):.0f}/{q(m['lat'], .95):.0f} | {h} | {fmt(hook[2], 0) if hook else ''} | {fmt(drss, 0) if drss else ''} |")
    lines += ["", "| embedder variant | embed ms per record (doc) | embed ms per query |", "|---|---|---|"]
    lines += [f"| {k} | {d:.1f} | {qq:.1f} |" for k, (d, qq) in cost.items()]
    text = "\n".join(lines + [""] + [f"- {n}" for n in notes])
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
