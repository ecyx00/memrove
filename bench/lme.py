#!/usr/bin/env python3
"""Tier-0 retrieval bench on LongMemEval-S, no LLM. Protocol copied from the 2026-10-08 phase A run (docs/state/2026-10-08-longmemeval-phaseA.md).

  nice python3 bench/lme.py --data /tmp/lme/data/longmemeval_s_cleaned.json --arms bm25-turn,bm25-session,memrove-chatlog

Arms: bm25-turn, bm25-session (baselines), memrove-chatlog (memrove_v3_chatlog.append + memrove_v3_chatsearch.merged, the `memrove search` path).
The dataset is untrusted data: run with `python3 -I`-style hygiene (this script only json-loads it). Engine code is imported from template/.claude/scripts,
never from ~/.claude. All state lives under --work (default /tmp/lme/work); HOME is pointed there and asserted.
"""
import argparse
import json
import math
import os
import re
import resource
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = (86.2, 1.0)   # bm25-turn S any@5, +- tolerance
TYPES = ("knowledge-update", "multi-session", "single-session-assistant", "single-session-preference", "single-session-user", "temporal-reasoning")
KS = (5, 10)
TOP = 50


def scored(items):
    """Phase A filter: no `_abs` question, at least one has_answer USER turn. -> [(item, gold_sessions, gold_turns)]; turn id = <sid>_<1-based turn index>."""
    out = []
    for it in items:
        if it["question_id"].endswith("_abs"):
            continue
        gs, gt = [], []
        for sid, sess in zip(it["haystack_session_ids"], it["haystack_sessions"]):
            hit = [f"{sid}_{n}" for n, t in enumerate(sess, 1) if t["role"] == "user" and t.get("has_answer")]
            if hit:
                gs.append(sid)
                gt += hit
        if gt:
            out.append((it, set(gs), set(gt)))
    return out


def bm25_order(docs, query, k1=1.5, b=0.75, eps=0.25):
    """rank_bm25.BM25Okapi re-implemented; docs and query are token lists, tokens from str.split(' '). -> doc indexes best first, ties: later doc first."""
    sc = bm25_scores(docs, query, k1, b, eps)
    return sorted(range(len(docs)), key=lambda i: (-sc[i], -i))


def bm25_scores(docs, query, k1=1.5, b=0.75, eps=0.25):
    n = len(docs)
    freqs, df, total = [], {}, 0
    for d in docs:
        f = {}
        for w in d:
            f[w] = f.get(w, 0) + 1
        freqs.append(f)
        total += len(d)
        for w in f:
            df[w] = df.get(w, 0) + 1
    avg = total / n
    idf, neg, s = {}, [], 0.0
    for w, c in df.items():
        idf[w] = math.log(n - c + 0.5) - math.log(c + 0.5)
        s += idf[w]
        if idf[w] < 0:
            neg.append(w)
    floor = eps * s / len(idf)
    for w in neg:
        idf[w] = floor
    sc = [0.0] * n
    for q in query:
        if q not in idf:
            continue
        for i, f in enumerate(freqs):
            if q in f:
                sc[i] += idf[q] * f[q] * (k1 + 1) / (f[q] + k1 * (1 - b + b * len(docs[i]) / avg))
    return sc


def sessions_of(turn_ids):
    """Turn ranking -> session ranking by first appearance."""
    seen, out = set(), []
    for t in turn_ids:
        s = t.rsplit("_", 1)[0]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall(ranked, gold, k):
    top = set(ranked[:k])
    return (100.0 if top & gold else 0.0), (100.0 if gold <= top else 0.0)   # any, all


def user_turns(it):
    return [(f"{sid}_{n}", t["content"]) for sid, sess in zip(it["haystack_session_ids"], it["haystack_sessions"]) for n, t in enumerate(sess, 1) if t["role"] == "user"]


def arm_bm25_turn(it, work):
    turns = user_turns(it)
    order = bm25_order([t[1].split(" ") for t in turns], it["question"].split(" "))
    ids = [turns[i][0] for i in order]
    return {"turns": ids, "sessions": sessions_of(ids), "texts": [turns[i][1] for i in order[:5]]}


def arm_bm25_session(it, work):
    sess = [(sid, " ".join(t["content"] for t in s if t["role"] == "user")) for sid, s in zip(it["haystack_session_ids"], it["haystack_sessions"])]
    order = bm25_order([s[1].split(" ") for s in sess], it["question"].split(" "))
    return {"turns": None, "sessions": [sess[i][0] for i in order], "texts": [sess[i][1] for i in order[:5]]}


def arm_memrove_chatlog(it, work):
    import memrove_v3_chatlog as chatlog
    import memrove_v3_chatsearch as cs
    allt, items = [], []
    for sid, date, sess in zip(it["haystack_session_ids"], it["haystack_dates"], it["haystack_sessions"]):
        for n, t in enumerate(sess, 1):
            allt.append(f"{sid}_{n}")
            items.append({"kind": "user" if t["role"] == "user" else "agent", "text": t["content"], "date": date})
    with tempfile.TemporaryDirectory(dir=work, prefix="state-") as state:
        assert str(Path(state).resolve()).startswith(str(Path(work).resolve())), state
        chatlog.append(state, items, today="2023-01-01")
        stored = chatlog.count(state)
        rows = cs.merged(state, it["question"], top=TOP, ref=qdate(it))
    msgs = [r for r in rows if not str(r[0]).startswith("rec:")]
    ids = [allt[r[0]] for r in msgs]
    return {"turns": ids, "sessions": sessions_of(ids), "texts": [r[3] for r in msgs[:5]],
            "extra": {"stored": stored, "expected": len(items), "turkish_route": bool(cs.is_turkish(it["question"])), "n_rows": len(rows)}}


def qdate(it):
    """The question's own date (LongMemEval question_date, 'YYYY/MM/DD (Dow) HH:MM') is the ref date for time expressions in the question."""
    from datetime import date
    y, m, d = it["question_date"].split(" ")[0].split("/")
    return date(int(y), int(m), int(d))


def mem0_open(d, collection="bench"):
    """Mem0 in raw mode: local HF embedder, local qdrant under d, history db under d. The llm block only satisfies the constructor; add(infer=False) never calls it."""
    os.environ["MEM0_TELEMETRY"] = "False"
    from mem0 import Memory
    return Memory.from_config({
        "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
        "vector_store": {"provider": "qdrant", "config": {"path": f"{d}/q", "collection_name": collection, "embedding_model_dims": 384, "on_disk": True}},
        "llm": {"provider": "openai", "config": {"api_key": "unused-never-called"}},
        "history_db_path": f"{d}/h.db"})


def arm_mem0_raw(it, work):
    """Same corpus as memrove-chatlog (every turn, both roles), ranked by Mem0's vector search."""
    allt = [f"{sid}_{n}" for sid, sess in zip(it["haystack_session_ids"], it["haystack_sessions"]) for n, _ in enumerate(sess, 1)]
    texts = [t["content"] for sess in it["haystack_sessions"] for t in sess]
    with tempfile.TemporaryDirectory(dir=work, prefix="mem0-") as d:
        m = mem0_open(d)
        for i, x in enumerate(texts):
            m.add(x, user_id="u", infer=False, metadata={"i": i})
        rows = m.search(it["question"], filters={"user_id": "u"}, limit=TOP)["results"]
        del m
    ids = [allt[r["metadata"]["i"]] for r in rows]
    return {"turns": ids, "sessions": sessions_of(ids), "texts": [r["memory"] for r in rows[:5]]}


OTHER = [""]   # the next question, set by the caller: a prompt unrelated to this haystack


def arm_memrove_autorecall(it, work):
    """The hook's chat lane on the haystack in a temp state: candidates with strength; packing happens per threshold in autorecall_report."""
    import memrove_v3_chatlog as chatlog
    import memrove_v3_chatsearch as cs
    allt, items = [], []
    for sid, date, sess in zip(it["haystack_session_ids"], it["haystack_dates"], it["haystack_sessions"]):
        for n, t in enumerate(sess, 1):
            allt.append(f"{sid}_{n}")
            items.append({"kind": "user" if t["role"] == "user" else "agent", "text": t["content"], "date": date})
    with tempfile.TemporaryDirectory(dir=work, prefix="state-") as state:
        assert str(Path(state).resolve()).startswith(str(Path(work).resolve())), state
        chatlog.append(state, items, today="2023-01-01")
        cs.LANE_RARE_DF = int(os.environ.get("BENCH_RARE_DF", cs.LANE_RARE_DF))
        tr = Path(work) / "empty-transcript.jsonl"   # the hook always has a transcript; an empty one excludes nothing. "recent" = no-transcript fallback
        tr.touch()
        tr = None if os.environ.get("BENCH_IN_CONTEXT") == "recent" else str(tr)
        cands = cs.lane_candidates(state, it["question"], tr, ref=qdate(it))
        foreign = cs.lane_candidates(state, OTHER[0], tr)
    return {"turns": [allt[c[1]] for c in cands], "sessions": sessions_of([allt[c[1]] for c in cands]), "texts": [],
            "cands": [(c[0], allt[c[1]], c[1], c[2], c[3], c[4]) for c in cands],
            "foreign_q": OTHER[0], "foreign": [(c[0], c[1], c[2], c[3], c[4]) for c in foreign]}


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def autorecall_report(items, results, count, thresholds, chosen, max_lines=None, width=None, min_rare=None):
    """items: scored [(it, gs, gt)], results: arm output per item. -> printed table over thresholds (the chosen one marked *)."""
    import memrove_v3_chatsearch as cs
    print(f"\nmax_lines={max_lines or cs.LANE_LINES} width={width or cs.LANE_SNIP}")
    print("| min strength | evidence-in-context@300 % | answer-in-context % (<=5 words, n) | chat-lane tok med | p90 | % prompts 0 tok | off-topic prompts injected % |\n|---|---|---|---|---|---|---|")
    for th in thresholds:
        ev = ans = nans = 0
        toks = []
        for (it, gs, gt), r in zip(items, results):
            ha = {f"{sid}_{n}" for sid, sess in zip(it["haystack_session_ids"], it["haystack_sessions"]) for n, t in enumerate(sess, 1) if t.get("has_answer")}
            cands = [(c[0], c[2], c[3], c[4], c[5]) for c in r["cands"]]
            tid = {c[2]: c[1] for c in r["cands"]}
            chatter = len({w for w in cs.WORD.findall(cs.fold(it["question"])) if len(w) > 3}) < cs.LANE_MIN_WORDS   # lane() skips these prompts
            lines = [] if chatter else cs.lane_pack(cands, it["question"], min_rare=min_rare, min_strength=th, budget=cs.LANE_TOKENS - cs._tokens(cs.LANE_HEADER + "\n"), max_lines=max_lines or cs.LANE_LINES, width=width or cs.LANE_SNIP)
            body = "\n".join(l for _, l in lines)
            toks.append(count(cs.LANE_HEADER + "\n" + body) if lines else 0)   # real tokens of what packing (chars/3 estimate) let through
            ev += any(tid[i] in ha for i, _ in lines)
            a = norm(it["answer"])
            if a and len(a.split()) <= 5:
                nans += 1
                ans += a in norm(body)
        n = len(items)
        fp = sum(bool(cs.lane_pack(r["foreign"], r["foreign_q"], min_rare=min_rare, min_strength=th, max_lines=max_lines or cs.LANE_LINES)) for r in results)
        mark = " *" if th == chosen else ""
        print(f"| {th}{mark} | {100 * ev / n:.1f} | {100 * ans / nans:.1f} (n={nans}) | {statistics.median(toks):.0f} | {pct(toks, 0.9)} | {100 * sum(t == 0 for t in toks) / n:.1f} | {100 * fp / n:.1f} |")


ARMS = {"bm25-turn": arm_bm25_turn, "bm25-session": arm_bm25_session, "memrove-chatlog": arm_memrove_chatlog, "memrove-autorecall": arm_memrove_autorecall, "mem0-raw": arm_mem0_raw}


def make_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (lambda s: len(enc.encode(s, disallowed_special=()))), "o200k_base tokens"
    except Exception:
        return (lambda s: len(s) / 4), "chars/4"


def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, math.ceil(p * len(v)) - 1)]


def evaluate(rows):
    """rows: [(qtype, result, gold_sessions, gold_turns)] -> metrics dict: S/T x any/all x k, plus n."""
    m = {"n": len(rows)}
    for lvl, key, gi in (("S", "sessions", 2), ("T", "turns", 3)):
        for k in KS:
            vals = [recall(r[1][key], r[gi], k) for r in rows if r[1][key] is not None]
            m[f"{lvl}any@{k}"] = sum(v[0] for v in vals) / len(vals) if vals else None
            m[f"{lvl}all@{k}"] = sum(v[1] for v in vals) / len(vals) if vals else None
    return m


def gate(metrics, target=GATE[0], tol=GATE[1]):
    return abs(metrics["Sany@5"] - target) <= tol


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


WORK_ROOT = Path("/tmp/lme/work")


def check_dirs(work, out):
    """-> (work, out) resolved. Everything the bench writes lives under /tmp/lme/work (a sibling such as /tmp/lme/livecopy is refused)."""
    work, out = Path(work).resolve(), Path(out).resolve()
    root = WORK_ROOT.resolve()
    for name, p in (("--work", work), ("--out", out)):
        if p != root and root not in p.parents:
            raise SystemExit(f"{name} must be under {root}: {p}")
    return work, out


def setup(work):
    """Per process (main and every pool worker): HOME under the work dir, no LLM, no vault, engine imported from the repo. Returns the dir to use."""
    work = Path(work)
    home = work / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)          # never the owner's HOME
    os.environ["MEMROVE_NO_LLM"] = "1"
    os.environ.pop("MEMROVE_VAULT", None)
    assert Path(os.path.expanduser("~")) == home
    check_dirs(work, work)
    sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
    sys.dont_write_bytecode = True
    return work


WORKER = [None]


def init_worker(work):
    WORKER[0] = setup(Path(work) / f"w{os.getpid()}")   # own temp state per worker


def run_one(arm, it, other):
    OTHER[0] = other
    return ARMS[arm](it, WORKER[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", default="bm25-turn,bm25-session,memrove-chatlog")
    ap.add_argument("--work", default="/tmp/lme/work")
    ap.add_argument("--out", default=None, help="raw rankings per arm (jsonl); default <work>/out, must be under /tmp/lme/work")
    ap.add_argument("--thresholds", default="0,2,4,6,8,10,12,14,16,20", help="autorecall: min-strength sweep")
    ap.add_argument("--grid", default="3x160,4x120,4x100,5x100,5x80,6x80", help="autorecall: max_lines x snippet width")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 3) - 2), help="worker processes (default CPU count - 2)")
    ap.add_argument("--rare-df", type=int, help="autorecall: a word is rare in at most this many messages (default: engine's LANE_RARE_DF)")
    ap.add_argument("--min-rare", default="", help="autorecall: rare words a hit must hold, comma list to sweep (default: engine's LANE_MIN_RARE)")
    ap.add_argument("--in-context", choices=("transcript", "recent"), default="transcript", help="autorecall: empty transcript (nothing excluded) or the 40-newest fallback")
    ap.add_argument("--limit", type=int, default=0, help="first N scored questions (debug)")
    a = ap.parse_args(argv)
    work, out = check_dirs(a.work, a.out or Path(a.work) / "out")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    setup(work)
    os.environ["BENCH_IN_CONTEXT"] = "recent" if a.in_context == "recent" else ""
    if a.rare_df is not None:
        os.environ["BENCH_RARE_DF"] = str(a.rare_df)
    with open(a.data, encoding="utf-8") as f:
        items = scored(json.load(f))
    if a.limit:
        items = items[:a.limit]
    count, unit = make_counter()
    print(f"questions scored: {len(items)}; token unit: {unit}", file=sys.stderr)
    table, bytype, tokcol, extras = {}, {}, {}, {}
    t0 = time.time()
    pool = ProcessPoolExecutor(a.jobs, initializer=init_worker, initargs=(str(work),))
    for arm in a.arms.split(","):
        rows, toks = [], []
        res = list(pool.map(run_one, [arm] * len(items), [x[0] for x in items], [items[(k + 1) % len(items)][0]["question"] for k in range(len(items))], chunksize=4))
        if arm == "memrove-autorecall":
            with open(out / f"{arm}.jsonl", "w") as fo:
                for (it, _, _), r in zip(items, res):
                    fo.write(json.dumps({"qid": it["question_id"], "turns": r["turns"][:TOP], "cands": [(round(c[0], 6), c[1]) for c in r["cands"][:TOP]]}) + "\n")
            import memrove_v3_chatsearch as cs
            ths = [float(x) for x in a.thresholds.split(",")]
            for g in a.grid.split(","):
                ml, w = g.split("x")
                for mr in a.min_rare.split(",") if a.min_rare else [None]:
                    print(f"\nmin_rare={cs.LANE_MIN_RARE if mr is None else mr} rare_df={cs.LANE_RARE_DF if a.rare_df is None else a.rare_df} in_context={a.in_context}")
                    autorecall_report(items, res, count, sorted(set(ths + [cs.LANE_MIN])), cs.LANE_MIN, int(ml), int(w), None if mr is None else int(mr))
            continue
        with open(out / f"{arm}.jsonl", "w") as fo:
            for n, ((it, gs, gt), r) in enumerate(zip(items, res)):
                rows.append((it["question_type"], r, gs, gt))
                toks.append(sum(count(t) for t in r["texts"]))
                fo.write(json.dumps({"qid": it["question_id"], "sessions": r["sessions"][:TOP], "turns": (r["turns"] or [])[:TOP], "extra": r.get("extra")}) + "\n")
                if n % 50 == 0:
                    print(f"{arm} {n}/{len(items)}", file=sys.stderr)
        table[arm] = evaluate(rows)
        bytype[arm] = {t: evaluate([r for r in rows if r[0] == t]) for t in TYPES}
        tokcol[arm] = (statistics.median(toks), pct(toks, 0.9))
        if arm == "memrove-chatlog":
            ex = [r[1]["extra"] for r in rows]
            ev_sess = sum(len(r[2]) for r in rows)
            ev_sess_hit = sum(len(r[2] & set(r[1]["sessions"])) for r in rows)
            ev_turn = sum(len(r[3]) for r in rows)
            ev_turn_hit = sum(len(r[3] & set(r[1]["turns"])) for r in rows)
            extras[arm] = {"questions": len(rows), "stored_all": sum(e["stored"] == e["expected"] for e in ex),
                           "turkish_route": sum(e["turkish_route"] for e in ex), "evidence_turns": ev_turn,
                           "evidence_turns_in_top50": ev_turn_hit,
                           "evidence_sessions": ev_sess, "evidence_sessions_with_hit_top50": ev_sess_hit,
                           "mean_rows": sum(e["n_rows"] for e in ex) / len(ex)}
    pool.shutdown()
    kb = 1 if sys.platform != "darwin" else 1024   # ru_maxrss: KiB on Linux, bytes on macOS
    print(f"wall {time.time() - t0:.0f}s, jobs {a.jobs}, peak RSS main {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / kb / 1024:.0f} MiB, "
          f"largest worker {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / kb / 1024:.0f} MiB", file=sys.stderr)
    if not table:
        return 0
    print("| arm | n | S any@5 | S all@5 | S any@10 | S all@10 | T any@5 | T all@5 | T any@10 | T all@10 | tok med | tok p90 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, m in table.items():
        print(f"| {arm} | {m['n']} | " + " | ".join(fmt(m[f'{l}{w}@{k}']) for l in "ST" for k in KS for w in ("any", "all")) + f" | {tokcol[arm][0]:.0f} | {tokcol[arm][1]:.0f} |")
    print(f"\ntoken unit: {unit}\n")
    print("| arm | " + " | ".join(f"{t} ({bytype[next(iter(bytype))][t]['n']})" for t in TYPES) + " |\n|---|" + "---|" * len(TYPES))
    for arm, bt in bytype.items():
        print(f"| {arm} | " + " | ".join("/".join(f"{bt[t][x]:.0f}" for x in ("Sany@5", "Sany@10", "Sall@5", "Sall@10")) if bt[t]["n"] else "-" for t in TYPES) + " |")
    for arm, e in extras.items():
        print(f"\n{arm} accounting: {json.dumps(e)}")
    if "bm25-turn" in table and not a.limit:
        ok = gate(table["bm25-turn"])
        print(f"\nGATE bm25-turn S any@5 = {table['bm25-turn']['Sany@5']:.1f} vs {GATE[0]} +- {GATE[1]}: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
