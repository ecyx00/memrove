#!/usr/bin/env python3
"""Abstention signals for memrove recall ("I do not know"), no LLM, no network.

  python3 bench/abstain.py [--locomo P] [--lme P] [--refresh]

Arm = memrove-chatlog, top 10 hits per question (cached in /tmp/lme/abstain-*.json; --refresh rebuilds). Each signal is higher = more confident an answer exists;
the policy abstains when signal < threshold. Threshold = 10th percentile of the signal over ANSWERABLE questions of LoCoMo conversations 0-4 (so <=10% of
answerable abstain on the fit set), then applied unchanged to LoCoMo conversations 5-9 and to LME-S (unanswerable = question_id ends in _abs; answerable = bench/lme.scored()).
Columns: abs-unans = % unanswerable abstained, abs-ans = % answerable abstained, R@5 = recall@5 over answerable with an abstained question counted 0
(LoCoMo: mean fraction of evidence turns in top 5; LME: session any@5), R@5 base = same without abstention, AUC = P(answerable signal > unanswerable signal).
Pass = abs-unans >= 50 and abs-ans <= 10 on both held-out sets. 'speaker' is LoCoMo-only (real memrove chat has no speaker names): reported, never chosen.
"""
import argparse
import json
import re
import statistics
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lme  # noqa: E402
import mem  # noqa: E402

K = 10
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being am do does did done have has had having i me my we our you your he she it its they them their his her this that these those
what when where who whom whose which why how whether will would can could shall should may might must not no yes so than then there here about into over after before during also just any some all
ever much many more most other tell said say says""".split())


def words(t):
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"\w+", t.lower())]


def terms(q):
    return {w for w in words(q) if w not in STOP}


def entities(q):
    """Capitalised words that are not the first word, plus numbers; lower-cased and stemmed like terms()."""
    toks = re.findall(r"\w+", q)[1:]
    return {words(t)[0] for t in toks if (t[0].isupper() or t.isdigit()) and t.lower() not in STOP}


# ---- collection (needs the memrove engine; the only slow part) ----
def search(state, q, ref=None):
    import memrove_v3_chatsearch as cs
    return cs._scored(state, q, K, None, None, None, ref)   # [(bm25 score, (idx, date, kind, text))], best first, lower = better


def collect_one(job):
    """job = (texts, sessions, speakers, question, ref, work) -> per-query record with hits and document frequencies of the query terms."""
    import memrove_v3_chatlog as chatlog
    texts, kinds, dates, q, ref, work = job
    lme.setup(work)
    with tempfile.TemporaryDirectory(dir=work, prefix="state-") as st:
        chatlog.append(st, [{"kind": k, "text": t, "date": d} for t, k, d in zip(texts, kinds, dates)], today="2023-01-01")
        rows = search(st, q, ref)
    qt = terms(q) | entities(q)
    toks = [set(words(t)) for t in texts]
    return {"hits": [{"i": r[0], "s": -sc, "t": r[3]} for sc, r in rows], "df": {w: sum(w in d for d in toks) for w in qt}, "n": len(texts)}


def collect_locomo(path, work):
    jobs, meta = [], []
    for ci, conv in enumerate(json.load(open(path))):
        ts = mem.turns(conv["conversation"])
        for q in conv["qa"]:
            g = mem.gold(q)
            if q["category"] != 5 and not g:
                continue
            jobs.append(([t[2] for t in ts], ["user" if t[1] == ts[0][1] else "agent" for t in ts], [""] * len(ts), q["question"], None, work))
            meta.append({"conv": ci, "unans": q["category"] == 5, "gold": sorted(g), "ids": [t[0] for t in ts], "spk": [t[1] for t in ts], "q": q["question"]})
    return jobs, meta


def collect_lme(path, work):
    jobs, meta = [], []
    scored = {it["question_id"]: (gs, gt) for it, gs, gt in lme.scored(json.load(open(path)))}
    for it in json.load(open(path)):
        qid = it["question_id"]
        if not (qid.endswith("_abs") or qid in scored):
            continue
        ids, texts, kinds, dates = [], [], [], []
        for sid, date, sess in zip(it["haystack_session_ids"], it["haystack_dates"], it["haystack_sessions"]):
            for n, t in enumerate(sess, 1):
                ids.append(f"{sid}_{n}")
                texts.append(t["content"])
                kinds.append("user" if t["role"] == "user" else "agent")
                dates.append(date)
        jobs.append((texts, kinds, dates, it["question"], lme.qdate(it), work))
        meta.append({"conv": -1, "unans": qid.endswith("_abs"), "gold": sorted(scored[qid][0]) if qid in scored else [], "ids": ids, "spk": None, "q": it["question"]})
    return jobs, meta


def load(kind, path, cache, refresh):
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.load(open(cache))
    work = str(lme.setup("/tmp/lme/work/abs"))
    jobs, meta = (collect_locomo if kind == "locomo" else collect_lme)(path, work)
    with ProcessPoolExecutor(4) as ex:
        recs = list(ex.map(collect_one, jobs, chunksize=4))
    for r, m in zip(recs, meta):
        m.update(r)
    json.dump(meta, open(cache, "w"))
    return meta


# ---- signals ----
def sess_of(kind, tid):
    return tid.split(":")[0] if kind == "locomo" else tid.rsplit("_", 1)[0]


def signals(kind, r):
    h = r["hits"]
    qt, ent = terms(r["q"]), entities(r["q"])
    if not h or not qt:
        return {k: float("-inf") for k in SIGNALS}
    sc = [x["s"] for x in h]
    htok = [set(words(x["t"])) for x in h]
    idf = {w: max(0.0, __import__("math").log((r["n"] - d + 0.5) / (d + 0.5) + 1)) for w, d in r["df"].items()}
    wsum = sum(idf.get(w, 0) for w in qt) or 1.0
    cov = lambda toks, ws: sum(idf.get(w, 0) for w in ws if w in toks) / (sum(idf.get(w, 0) for w in ws) or 1.0)
    u3 = set().union(*htok[:3])
    uk = set().union(*htok)
    ids = [r["ids"][x["i"]] for x in h]
    modal = statistics.mode([sess_of(kind, i) for i in ids])
    out = {
        "top1": sc[0],
        "top1/terms": sc[0] / len(qt),
        "gap12": sc[0] - sc[1] if len(sc) > 1 else sc[0],
        "gap12/top1": (sc[0] - sc[1]) / sc[0] if len(sc) > 1 and sc[0] else 1.0,
        "cov-top1": len(qt & htok[0]) / len(qt),
        "cov-top10": len(qt & uk) / len(qt),
        "idfcov-top1": cov(htok[0], qt),
        "idfcov-top3": cov(u3, qt),
        "idfcov-top10": cov(uk, qt),
        "entcov-top10": len(ent & uk) / len(ent) if ent else len(qt & uk) / len(qt),
        "session-agree": sum(sess_of(kind, i) == modal for i in ids) / len(ids),
        "n-hits": len(h) / K,
    }
    named = {x for x in set(r["spk"] or []) if x.lower() in r["q"].lower()}   # LoCoMo only, not a candidate: share of top-10 hits spoken by the speaker the question names (1.0 if none named)
    out["speaker(loco)"] = sum(r["spk"][x["i"]] in named for x in h) / len(h) if named else (1.0 if r["spk"] else float("nan"))
    top = max(qt, key=lambda w: idf.get(w, 0))
    out["idfmass-top1"] = sum(idf.get(w, 0) for w in qt & htok[0])
    out["rare-top3"] = float(top in u3)
    return out


SIGNALS = ["top1", "top1/terms", "gap12", "gap12/top1", "cov-top1", "cov-top10", "idfcov-top1", "idfcov-top3", "idfcov-top10", "entcov-top10", "session-agree", "n-hits", "idfmass-top1", "rare-top3", "speaker(loco)"]


def recall5(kind, r):
    ids = [r["ids"][x["i"]] for x in r["hits"][:5]]
    if kind == "locomo":
        g = set(r["gold"])
        return len(g & set(ids)) / len(g)
    return float(bool(set(r["gold"]) & {sess_of(kind, i) for i in ids}))


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    if any(x != x for x in pos + neg):
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def evaluate(kind, recs, sig, thr):
    """-> (abs-unans %, abs-ans %, R@5 after abstention, R@5 base, AUC)."""
    S = [signals(kind, r)[sig] for r in recs]
    un = [s for s, r in zip(S, recs) if r["unans"]]
    an = [(s, recall5(kind, r)) for s, r in zip(S, recs) if not r["unans"]]
    f = lambda xs, t: 100 * sum(x < t for x in xs) / len(xs) if xs else float("nan")
    return (f(un, thr), f([a[0] for a in an], thr), 100 * sum(0 if s < thr else v for s, v in an) / len(an),
            100 * sum(v for _, v in an) / len(an), auc([a[0] for a in an], un))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--locomo", default="/tmp/lme/locomo/locomo10.json")
    ap.add_argument("--lme", default="/tmp/lme/data/longmemeval_s_cleaned.json")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    lo = load("locomo", a.locomo, "/tmp/lme/abstain-locomo.json", a.refresh)
    lm = load("lme", a.lme, "/tmp/lme/abstain-lme.json", a.refresh)
    fit = [r for r in lo if r["conv"] < 5]
    held = [r for r in lo if r["conv"] >= 5]
    cnt = lambda rs: f"{sum(r['unans'] for r in rs)} unans / {sum(not r['unans'] for r in rs)} ans"
    print(f"fit LoCoMo conv 0-4: {cnt(fit)}; held-out LoCoMo conv 5-9: {cnt(held)}; LME-S: {cnt(lm)}")
    print(f"{'signal':<15}{'thr':>8} | {'LoCoMo held-out':^34} | {'LME-S':^34} | pass")
    print(f"{'':<23} | {'abs-un':>6}{'abs-an':>7}{'R@5':>7}{'base':>6}{'AUC':>6} | {'abs-un':>6}{'abs-an':>7}{'R@5':>7}{'base':>6}{'AUC':>6} |")
    for sig in SIGNALS:
        thr = pctl([signals("locomo", r)[sig] for r in fit if not r["unans"]], 0.10)
        if thr != thr:   # nan signal (speaker on a set without speakers is handled per row; fit set has them)
            continue
        h, m = evaluate("locomo", held, sig, thr), evaluate("lme", lm, sig, thr)
        ok = all(x[0] >= 50 and x[1] <= 10 for x in (h, m))
        row = lambda x: f"{x[0]:6.1f}{x[1]:7.1f}{x[2]:7.1f}{x[3]:6.1f}{x[4]:6.2f}"
        print(f"{sig:<15}{thr:8.3f} | {row(h)} | {row(m)} | {'PASS' if ok else 'fail'}")
    # trade-off curve: each set's own threshold, so the achievable ceiling is visible (not a transferable result)
    print("\nceiling per set (threshold re-fit on that set's answerable questions): abstain-% of unanswerable at answerable-abstain 5 / 10 / 20 / 30 %")
    for name, kind, rs in (("LoCoMo held-out", "locomo", held), ("LME-S", "lme", lm)):
        for sig in SIGNALS[:-1]:
            an = [signals(kind, r)[sig] for r in rs if not r["unans"]]
            cells = []
            for p in (0.05, 0.10, 0.20, 0.30):
                cells.append(f"{evaluate(kind, rs, sig, pctl(an, p))[0]:5.1f}")
            print(f"{name:<16}{sig:<15}" + " ".join(cells))


if __name__ == "__main__":
    main()
