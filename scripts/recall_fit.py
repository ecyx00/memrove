"""Threshold fitting for the recall eval: Learn-then-Test, fixed sequence (Angelopoulos et al. 2021), no embedding, no I/O.

A parameter setting is "valid" when the one-sided Clopper-Pearson upper bound of its false-inject rate (injected records that are not
labelled relevant) is <= alpha at confidence 1 - delta. Settings are tried from the most conservative threshold down; the sequence stops
at the first failure and the last valid one (most coverage) is kept, so no further correction for the threshold sweep is needed.
Other choices (z-score on/off, lex_w, delta, cascade band) are searched by Bonferroni: delta / number of settings.

Tops are `memrove_v3_hybrid.stage1` candidates (dict with id, cos, z, pcover and, for the cascade, cos2 / z2 from `l2_rescore`).
`judge(pid) -> (labels {record id: label}, skip set)`; an unlabelled injected record counts as a false inject (the fair eval makes that count 0)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "template/.claude/scripts"))
import memrove_v3_hybrid as hyb

INF = float("inf")
LEX = {False: (0.25, 0.5, 1.0), True: (0.5, 1.0, 2.0)}  # lex_w grid by znorm (z-scores are about 10x wider than cosines)
DELTA = {False: (0.05, 0.1, 0.3), True: (0.5, 1.0, 3.0)}
BAND_Q = (0.5, 0.7, 0.9)  # cascade: L2 sees the prompts whose L1 best score lies between this quantile and the L1 threshold
MAX_ITEMS = 3


def configs():
    return [(z, lw, d) for z in (False, True) for lw in LEX[z] for d in DELTA[z]]


def cp_upper(k, n, delta):
    """One-sided Clopper-Pearson upper bound for k errors in n trials: the p with P(Binom(n, p) <= k) = delta."""
    if n == 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > delta else (lo, mid)
    return hi


def ltt_pick(stats, alpha, delta):
    """stats: [(n_accepted, n_false, ...)] from most conservative to most liberal. -> index of the last setting of the unbroken run
    whose upper bound is <= alpha, or None. The run starts at the first setting that accepts at least n_min items, the smallest count
    for which zero errors could pass: that start depends on the scores only, never on the labels, so the fixed-sequence guarantee holds."""
    n_min = next((n for n in range(1, 10000) if cp_upper(0, n, delta) <= alpha), None)
    last = None
    for j, st in enumerate(stats):
        if last is None and (n_min is None or st[0] < n_min):
            continue
        if cp_upper(st[1], st[0], delta) > alpha:
            break
        last = j
    return last


# --- decisions without mutation -------------------------------------------------------------------------------------------
def k1(z, lw):
    return lambda c: hyb.s1_of(c, lw, z)


def k2(z, lw):
    return lambda c: None if c.get("cos2") is None else (c["z2"] if z else c["cos2"]) + lw * c["pcover"]


def rank(top, key):
    return sorted(((key(c), c["id"]) for c in top if key(c) is not None), key=lambda x: -x[0])


def pick(ranked, tau, delta):
    """Same rule as memrove_v3_hybrid.decide: >= tau, within delta of the best, at most MAX_ITEMS."""
    if not ranked:
        return []
    best = ranked[0][0]
    return [rid for s, rid in ranked if s >= tau and s >= best - delta][:MAX_ITEMS]


def tally(runs, judge):
    """runs {pid: [ids]} -> (accepted, false, answered prompts, relevant hits)."""
    acc = fal = ans = hit = 0
    for pid, ids in runs.items():
        lab, skip = judge(pid)
        ids = [i for i in ids if i not in skip]
        acc += len(ids)
        ans += bool(ids)
        h = sum(lab.get(i) == "relevant" for i in ids)
        hit += h
        fal += len(ids) - h
    return acc, fal, ans, hit


def tau_grid(rankeds, limit=120):
    vals = sorted({round(s, 3) for r in rankeds.values() for s, _ in r}, reverse=True)
    step = max(1, len(vals) // limit)
    return vals[::step] + ([vals[-1]] if vals else [])


def _best(old, new):
    return new if old is None or (new["answered"], new["hits"]) > (old["answered"], old["hits"]) else old


def fit_single(tops, pids, judge, alpha, delta=0.1):
    """One arm (static or EG2 alone). -> dict(z, lw, delta, tau, answered, hits, accepted) or None when no setting is valid."""
    best, cfgs = None, configs()
    for z, lw, d in cfgs:
        key = k1(z, lw)
        ranked = {p: rank(tops[p]["top"], key) for p in pids}
        taus = tau_grid(ranked)
        stats = [tally({p: pick(ranked[p], t, d) for p in pids}, judge) for t in taus]
        j = ltt_pick(stats, alpha, delta / len(cfgs))
        if j is not None:
            best = _best(best, {"z": z, "lw": lw, "delta": d, "tau": taus[j], "accepted": stats[j][0], "answered": stats[j][2], "hits": stats[j][3]})
    return best


def best_l1(tops, pids, p):
    """Best L1 score per prompt under setting p (a dict from fit_single, or the neutral default)."""
    key = k1(p["z"], p["lw"])
    return {pid: max([key(c) for c in tops[pid]["top"]] or [-INF]) for pid in pids}


DEFAULT_L1 = {"z": False, "lw": 0.5, "delta": 0.1, "tau": INF}


def fit_cascade(tops, pids, judge, alpha, delta=0.1, l1=None):
    """L1 keeps its own fitted threshold (hi); the prompts whose best L1 score lies in [lo, hi) go to L2, fitted only on those prompts.
    -> dict(l1, lo, q, l2) ; l2 None = no valid L2 setting (the cascade then equals L1)."""
    l1 = l1 or fit_single(tops, pids, judge, alpha, delta)
    p1 = l1 or DEFAULT_L1
    hi = p1["tau"]
    best1 = best_l1(tops, pids, p1)
    order = sorted(best1.values())
    out = {"l1": l1, "lo": hi, "q": None, "l2": None}
    cfgs = configs()
    for q in BAND_Q:
        lo = order[int(q * (len(order) - 1))] if order else INF
        reach = [p for p in pids if lo <= best1[p] < hi]
        if not reach:
            continue
        for z, lw, d in cfgs:
            key = k2(z, lw)
            ranked = {p: rank(tops[p]["top"], key) for p in reach}
            taus = tau_grid(ranked)
            stats = [tally({p: pick(ranked[p], t, d) for p in reach}, judge) for t in taus]
            j = ltt_pick(stats, alpha, delta / (len(cfgs) * len(BAND_Q)))
            if j is not None:
                new = {"z": z, "lw": lw, "delta": d, "tau": taus[j], "answered": stats[j][2], "hits": stats[j][3], "accepted": stats[j][0]}
                if out["l2"] is None or (new["answered"], new["hits"]) > (out["l2"]["answered"], out["l2"]["hits"]):
                    out.update(lo=lo, q=q, l2=new)
    return out


# --- applying a fit -------------------------------------------------------------------------------------------------------
def run_single(tops, pids, p):
    if p is None:
        return {pid: [] for pid in pids}
    key = k1(p["z"], p["lw"])
    return {pid: pick(rank(tops[pid]["top"], key), p["tau"], p["delta"]) for pid in pids}


def run_cascade(tops, pids, fit):
    p1 = fit["l1"] or DEFAULT_L1
    key1, hi = k1(p1["z"], p1["lw"]), p1["tau"]
    out, l2 = {}, fit["l2"]
    for pid in pids:
        r1 = rank(tops[pid]["top"], key1)
        b = r1[0][0] if r1 else -INF
        if b >= hi:
            out[pid] = pick(r1, hi, p1["delta"])
        elif l2 and fit["lo"] <= b:
            out[pid] = pick(rank(tops[pid]["top"], k2(l2["z"], l2["lw"])), l2["tau"], l2["delta"])
        else:
            out[pid] = []
    return out


def pool_pairs(tops, pids, with_l2=False, n=MAX_ITEMS):
    """Every (pid, id) that ANY setting of the grid could inject: the top-n of each ranking in the grid. The label pool must contain all of them."""
    pairs = set()
    for z in (False, True):
        for lw in LEX[z]:
            for key in (k1(z, lw),) + ((k2(z, lw),) if with_l2 else ()):
                for pid in pids:
                    pairs.update((pid, rid) for _s, rid in rank(tops[pid]["top"], key)[:n])
    return pairs
