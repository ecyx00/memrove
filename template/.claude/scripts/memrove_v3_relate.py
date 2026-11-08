"""Temporal relation edges between records (design 2026-11-07 sections 3, 6, 8).

relate(db, pairs, decider="jev") -> one result per pair. Order of gates:
  1. same-facet overlap gate (deterministic: token Jaccard, shared numbers / paths / entities). A pair that fails is never sent anywhere.
  2. a decider (jev | haiku | none, or any callable) names one relation or abstains. Error, no key, remote not allowed -> abstain, never a crash.
  3. contradicts / supports / derived_from are written as graph_edges (prov by=jev, valid_from = the newer record's date).
     supersedes is only PROPOSED (status "proposed"): it never closes a record or writes an edge on its own.
Records are dicts {id, text, valid_from}. Direction: src = newer record, dst = older one (newer date by code, not by the model).
The key is read from JEV_API_KEY only. Remote deciders receive text only when the caller passes remote_ok=True.
"""
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

import memrove_v3_graph as graph

RELATIONS = ("supersedes", "contradicts", "supports", "derived_from")
WRITABLE = ("contradicts", "supports", "derived_from")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_TRUE = 0.6  # noul score at or above this counts as yes (same cut as run_jev_verify_v2)
STOP = set("the a an and or of to in on for with is are be this that it as at by from not do does we you i our your should must was were has have had "
           "bir ve ile için bu şu da de mi mı ne ama gibi olarak olan".split())
NUM = re.compile(r"\b\d+(?:[.,]\d+)?%?")
PATH = re.compile(r"[\w.~-]*[/\\][\w./\\-]+|\b\w+\.(?:py|js|ts|md|json|sh|swift|sqlite3?|toml|yaml|yml)\b")
ENT = re.compile(r"`([^`]+)`|\b([A-Z][a-z]+[A-Z]\w*|[A-Z]{2,}\w*)\b")  # `code`, CamelCase, ACRONYM


def _tokens(text):
    return {t for t in re.findall(r"\w{3,}", text.lower()) if t not in STOP}


def _special(text):
    ents = {a or b for a, b in ENT.findall(text)}
    return {x.lower() for x in NUM.findall(text)} | {x.lower() for x in PATH.findall(text)} | {e.lower() for e in ents}


def overlap(a, b):
    """-> (jaccard, number of shared numbers/paths/entities)."""
    ta, tb = _tokens(a), _tokens(b)
    return (len(ta & tb) / len(ta | tb) if ta | tb else 0.0), len(_special(a) & _special(b))


def gate(a, b, min_jaccard=0.1, min_shared=2):
    j, s = overlap(a, b)
    return j >= min_jaccard or s >= min_shared


# ---------------------------------------------------------------- deciders: (old, new) -> {"relation": str|None, "why": str, "scores": dict}
QUESTIONS = {
    "supersedes": "Does NEWER replace OLDER, so that OLDER is no longer the current rule or decision on the same point?",
    "contradicts": "Do OLDER and NEWER assert incompatible things about the same subject, so both cannot be true at once?",
    "supports": "Does NEWER confirm, back or give evidence for what OLDER says, without replacing it?",
    "derived_from": "Is NEWER built from OLDER, a consequence, summary or refinement that takes OLDER as its source?",
}


def _noul(rel):
    return {"type": "noul", "instructions": QUESTIONS[rel],
            "criteria": {"true": f"The relation '{rel}' clearly holds from NEWER to OLDER.", "false": f"It does not clearly hold."}}


def _abstain(why):
    return {"relation": None, "why": why, "scores": {}}


def _score_ok(s):
    """A Jev score counts only as a finite number in [0, 1]; bool, NaN, inf and out-of-range values are dropped."""
    return isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and 0.0 <= s <= 1.0


def jev_decide(old, new, post=None):
    """One systemone call, four noul questions. Highest score, if >= JEV_TRUE -> that relation; else abstain."""
    key = (os.environ.get("JEV_API_KEY") or "").strip()
    if not key:
        return _abstain("no JEV_API_KEY")
    body = {"model": JEV_MODEL, "questions": {r: _noul(r) for r in RELATIONS},
            "state": {"query": "how does the NEWER statement relate to the OLDER one",
                      "pages": [{"url": "older", "title": "OLDER", "text": old["text"][:1500]},
                                {"url": "newer", "title": "NEWER", "text": new["text"][:1500]}]}}
    try:
        data = (post or _post)(body, key)
        scores = {r: (data.get("answers", {}).get(r, {}) or {}).get("noul") for r in RELATIONS}
        scores = {r: float(s) for r, s in scores.items() if _score_ok(s)}
    except urllib.error.HTTPError as e:  # 402/429/5xx: the status code only
        return _abstain(f"jev error: HTTP {e.code}")
    except Exception as e:  # timeout, bad JSON, bad header: the class name only, never str(e) (it can hold the key)
        return _abstain(f"jev error: {type(e).__name__}")
    # argmax, not "exactly one yes": the relations nest (supports ~ derived_from, contradicts ~ supersedes), so two scores
    # are often >= JEV_TRUE together. On 36 synthetic pairs: exactly-one 0.17-0.19 accuracy, argmax 1.00 (2026-11-08).
    top = max(scores, key=scores.get) if scores else None
    if top is None or scores[top] < JEV_TRUE:
        return {"relation": None, "why": "jev abstained", "scores": scores}
    return {"relation": top, "why": f"jev {top} {scores[top]:.2f}", "scores": scores}


def _post(body, key, tries=2):
    req = urllib.request.Request(JEV_URL, json.dumps(body).encode(), {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "memrove-relate/1"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise


def haiku_decide(old, new):
    # shortcut: haiku relation prompt not written yet (E92 measured its supersede precision at 0.40-0.60), abstain until it has a rubric.
    return _abstain("haiku decider not wired")


def none_decide(old, new):
    return _abstain("decider none")


DECIDERS = {"jev": (True, jev_decide), "haiku": (True, haiku_decide), "none": (False, none_decide)}  # name -> (remote, fn)


# ---------------------------------------------------------------- the one entry point
def relate(db, pairs, decider="jev", remote_ok=False, write=True, min_jaccard=0.1, min_shared=2):
    """pairs: [(rec_a, rec_b)], rec = {id, text, valid_from}. decider: a name in DECIDERS or a callable(old, new).
    -> [{src, dst, relation, status, why}] status: gated | abstain | edge | proposed. Only status "edge" touched the db."""
    remote, fn = DECIDERS[decider] if isinstance(decider, str) else (False, decider)
    name = decider if isinstance(decider, str) else getattr(decider, "__name__", "custom")
    out = []
    for a, b in pairs:
        old, new = sorted((a, b), key=lambda r: (str(r.get("valid_from") or ""), r["id"]))
        res = {"src": new["id"], "dst": old["id"], "relation": None, "status": "gated", "why": "overlap gate failed"}
        out.append(res)
        if not gate(old["text"], new["text"], min_jaccard, min_shared):
            continue
        if remote and not remote_ok:
            res.update(status="abstain", why="remote decider not allowed (remote_ok=False)")
            continue
        try:
            d = fn(old, new)
        except Exception as e:
            d = _abstain(f"decider error: {type(e).__name__}")
        rel = d.get("relation")
        if rel not in RELATIONS:
            res.update(status="abstain", why=d.get("why") or "abstain")
            continue
        res.update(relation=rel, why=d.get("why") or rel)
        same_day = str(old.get("valid_from") or "") == str(new.get("valid_from") or "")
        if rel == "supersedes":  # a proposal only; the caller (owner/core) closes records
            res["status"] = "proposed" if not same_day else "abstain"
            if same_day:
                res["why"] = "supersedes needs a strictly older record (same date)"
            continue
        if write:
            write_edge(db, res, new.get("valid_from"), name)
        res["status"] = "edge"
    return out


def write_edge(db, res, valid_from, name):
    """Write one decided result (relation set) as an edge. For callers that decide with write=False and open the db afterwards."""
    graph.add_edge(db, res["src"], res["dst"], res["relation"], res["why"], graph.provenance("jev", "model", model=name))
    db.execute("UPDATE graph_edges SET valid_from=? WHERE src=? AND dst=? AND type=?",
               (valid_from or graph.now(), res["src"], res["dst"], res["relation"]))
