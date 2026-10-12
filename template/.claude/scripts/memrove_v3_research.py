"""Research shelf: one shared catalog of every paper, repo, model and dataset the research reports name (internal evidence for `memrove wiki evidence`).

`memrove wiki research catalog` -> <vault>/wiki/_sources/catalog.json   (Haiku names the items, 0-token checks gate every mention)

Rules: never invent a URL or a GitHub owner. A name must appear verbatim in the report that mentions it, a URL must appear verbatim
in the report that gives it. Items merge only on the same id (same URL) or, for a name-only entry, on an equal normalized name
with exactly one URL candidate of the same kind. A wrong merge is worse than a duplicate.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
import os
import re
import time
from pathlib import Path

import memrove_v3_llm as llm

HOME = Path.home()
REPORT_DIRS = [(HOME / "Desktop/notes/docs/research", "*.md", None)]
# Research-like reports outside the research folder: {project: [file names]} (empty by default).
EXTRA_REPORTS = {}
VERDICTS = ("adopted", "inspired", "rejected", "reviewed")
KINDS = ("paper", "repo", "model", "dataset")
FACT_KEYS = ("year", "stars", "license", "venue")
EXTRACT_MODEL = "haiku"
CHUNK = 28000
URL_RE = re.compile(r"https?://[^\s)\]>\"'`|,]+")


# ---- ids and merging -------------------------------------------------------------------------------

def norm(name):
    name = re.sub(r"\([^)]*\)", " ", name or "")
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def item_id(url, name, kind=None):
    u = (url or "").strip().rstrip(".,;")
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", u, re.I)
    if m:
        return "arxiv:" + m.group(1)
    m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", u, re.I)
    if m:
        return f"github:{m.group(1).lower()}/{re.sub(r'[.]git$', '', m.group(2).lower())}"
    m = re.search(r"huggingface\.co/(datasets/)?([\w.-]+)/([\w.-]+)", u, re.I)
    if m:
        return f"hf:{'datasets/' if m.group(1) else ''}{m.group(2).lower()}/{m.group(3).lower()}"
    return "name:" + norm(name)


def _repo_part(i):
    return norm(i.split("/", 1)[1]) if i.startswith(("github:", "hf:")) else None


def merge(rows):
    """rows: (research slug, doc path, raw item) -> catalog items. Same id merges; name-only joins a URL item only when unambiguous."""
    by_id = {}
    for slug, doc, raw in rows:
        iid = item_id(raw.get("url"), raw["name"])
        it = by_id.setdefault(iid, {"id": iid, "kind": raw.get("kind") or "repo", "title": raw["name"], "url": raw.get("url") or None,
                                    "one_liner": raw.get("one_liner") or "", "facts": {}, "mentions": [], "concepts": [], "projects": []})
        if not it["url"] and raw.get("url"):
            it["url"] = raw["url"]
        if not it["one_liner"]:
            it["one_liner"] = raw.get("one_liner") or ""
        for k, v in (raw.get("facts") or {}).items():
            it["facts"].setdefault(k, {"value": v, "doc": doc})
        if any(m["research"] == slug and m["doc"] == doc for m in it["mentions"]):
            continue
        it["mentions"].append({"research": slug, "verdict": raw.get("verdict") or "reviewed", "why": raw.get("why") or "", "doc": doc,
                               "name": raw["name"], "quote": raw.get("quote") or "", **({"url": raw["url"]} if raw.get("url") else {})})
    for iid in [i for i in by_id if i.startswith("name:")]:
        it = by_id[iid]
        cands = [j for j in by_id if not j.startswith("name:") and _repo_part(j) == iid[5:] and by_id[j]["kind"] == it["kind"]]
        if len(cands) == 1:
            t = by_id[cands[0]]
            t["mentions"] += [m for m in it["mentions"] if not any(x["research"] == m["research"] and x["doc"] == m["doc"] for x in t["mentions"])]
            for k, v in it["facts"].items():
                t["facts"].setdefault(k, v)
            t["one_liner"] = t["one_liner"] or it["one_liner"]
            del by_id[iid]
    return sorted(by_id.values(), key=lambda i: i["id"])


# ---- checker (0 tokens) ----------------------------------------------------------------------------

def _ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def check_item(item, docs):
    """docs: path -> text. -> list of errors."""
    errs, iid = [], item.get("id", "?")
    if item.get("kind") not in KINDS:
        errs.append(f"{iid}: bad kind {item.get('kind')!r}")
    if not item.get("mentions"):
        errs.append(f"{iid}: no mentions")
    url_seen = False
    for m in item.get("mentions", []):
        text = _ws(docs.get(m.get("doc"), ""))
        who = f"{iid} [{m.get('research')}]"
        if not text:
            errs.append(f"{who}: doc {m.get('doc')} missing")
            continue
        if m.get("verdict") not in VERDICTS:
            errs.append(f"{who}: bad verdict {m.get('verdict')!r}")
        if not m.get("name") or m["name"] not in text:
            errs.append(f"{who}: name {m.get('name')!r} not verbatim in doc")
        q = _ws(m.get("quote"))
        if not q or q not in text or (m.get("name") and m["name"] not in q):
            errs.append(f"{who}: quote {q[:60]!r} not verbatim in doc or lacks the name")
        if m.get("url"):
            url_seen = True
            if m["url"] not in docs[m["doc"]]:
                errs.append(f"{who}: url {m['url']} not in doc")
    if item.get("url"):
        if not url_seen:
            errs.append(f"{iid}: item url has no mention carrying it")
        if item_id(item["url"], item.get("title", "")) != iid:
            errs.append(f"{iid}: id does not follow from url {item['url']}")
    elif not iid.startswith("name:"):
        errs.append(f"{iid}: id has a URL form but url is null")
    for k, f in (item.get("facts") or {}).items():
        if k not in FACT_KEYS or str(f.get("value")) not in docs.get(f.get("doc"), ""):
            errs.append(f"{iid}: fact {k}={f.get('value')!r} not verbatim in {f.get('doc')}")
    return errs


def check_catalog(cat, docs):
    errs = []
    for it in cat["items"]:
        if it.get("origin") != "grab":  # a capture has no report mention: its evidence is the card and the snapshot
            errs += check_item(it, docs)
    return errs


# ---- grab captures: kept across a catalog rebuild --------------------------------------------------

def _catalog_path(vault):
    return Path(vault) / "wiki" / "_sources" / "catalog.json"


def _write_catalog(vault, cat):
    path = _catalog_path(vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cat, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def add_capture(vault, cap):
    """cap: id, kind, title, url, card, reason (verbatim), snapshot, visibility. Adds an item with origin: grab, or, when a report
    already named the same id, a `captures` entry on that item. Idempotent per card."""
    try:
        cat = json.loads(_catalog_path(vault).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cat = {"generated": date.today().isoformat(), "reports": 0, "items": []}
    entry = {"card": cap["card"], "url": cap.get("url"), "reason": cap.get("reason") or "", "snapshot": cap.get("snapshot")}
    it = next((i for i in cat["items"] if i["id"] == cap["id"]), None)
    if it is None:
        it = {"id": cap["id"], "kind": cap["kind"], "title": cap["title"], "url": cap.get("url"), "one_liner": "", "facts": {},
              "mentions": [], "concepts": [], "projects": [], "origin": "grab", "card": cap["card"], "reason": entry["reason"],
              "snapshot": cap.get("snapshot"), "visibility": cap.get("visibility", "internal")}
        cat["items"].append(it)
        cat["items"].sort(key=lambda i: i["id"])
    it["captures"] = [c for c in it.get("captures", []) if c["card"] != cap["card"]] + [entry]
    _write_catalog(vault, cat)
    return it


def keep_captures(old_items, items):
    """A rebuild from reports must not drop what the owner captured: merge `captures` into the same id, re-add grab-only items."""
    by_id = {i["id"]: i for i in items}
    for o in old_items:
        if o["id"] in by_id:
            if o.get("captures"):
                by_id[o["id"]]["captures"] = o["captures"]
        elif o.get("origin") == "grab":
            by_id[o["id"]] = o
    return sorted(by_id.values(), key=lambda i: i["id"])


# ---- reports ---------------------------------------------------------------------------------------

def reports():
    """-> list of (path, project|None)."""
    out = []
    for d, glob, proj in REPORT_DIRS:
        names = sorted(p.name for p in d.glob("*.md")) if glob else EXTRA_REPORTS.get(proj, [])
        out += [(d / n, proj) for n in names if (d / n).is_file()]
    return out


def slug_of(path):
    return Path(path).stem


def _log(state, entry):
    with (Path(state) / "wiki-log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **entry}) + "\n")


def _tokens(usage):
    return sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"))


def _call(runner, vault, state, model, job, system, payload, label):
    usage = {}
    text, cost, err = runner(system, payload, usage, model)
    _log(state, {"job": job, "model": model, "label": label, "usage": usage, "tokens": _tokens(usage), "cost": cost, "err": err})
    return text, err


def default_runner(vault):
    return lambda system, payload, usage, model: llm._claude(system, payload, vault, model=model, usage=usage)


# ---- catalog ---------------------------------------------------------------------------------------

EXTRACT_SYSTEM = f"""You list the papers, GitHub repos, ML models and datasets that a research report names. Output ONLY a JSON array.
Each element: {{"name": exact text as written in the report (copy it, no rewording), "kind": one of {list(KINDS)},
"url": the URL for it exactly as written in the report, or null (never guess a URL or an owner),
"one_liner": one short Turkish sentence on what it is, from the report,
"verdict": one of {list(VERDICTS)} (adopted = we use it; inspired = idea taken, not the tool; rejected = rejected; reviewed = only looked at / compared),
"why": short Turkish reason from the report, "quote": a verbatim snippet (max 160 chars) of the report that contains the name and shows the verdict,
"facts": {{"year"|"stars"|"license"|"venue": value copied verbatim from the report}} (only those stated; {{}} if none)}}.
Include only named, specific artifacts (not generic concepts, companies, or the report's own tools). Skip anything you cannot quote. No prose."""


def _chunks(text):
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > CHUNK and cur:
            out.append(cur)
            cur = ""
        cur += line
    return out + ([cur] if cur else [])


def _parse(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except ValueError:
        return None
    return [d for d in data if isinstance(d, dict) and d.get("name")]


def extract(path, vault, state, runner):
    """-> (raw items checked against the doc, dropped count). Cached by content hash."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cache = Path(state) / "research-cache"
    cache.mkdir(parents=True, exist_ok=True)
    raw = []
    for n, chunk in enumerate(_chunks(text)):
        cf = cache / (hashlib.sha1((EXTRACT_SYSTEM + chunk).encode()).hexdigest() + ".json")
        if cf.exists():
            raw += json.loads(cf.read_text())
            continue
        for attempt in (0, 1):
            out, err = _call(runner, vault, state, EXTRACT_MODEL, "research-catalog", EXTRACT_SYSTEM, chunk, f"{slug_of(path)}#{n}.{attempt}")
            items = _parse(out) if not err else None
            if items is not None:
                break
        if items is None:
            raise RuntimeError(f"extraction failed for {path}: {err or 'bad JSON'}")
        cf.write_text(json.dumps(items, ensure_ascii=False))
        raw += items
    return raw, text


def _clean(r, text):
    """Normalize one raw Haiku item; None if it cannot pass the checker."""
    r = dict(r)
    r["name"] = str(r["name"]).strip()
    if r.get("kind") not in KINDS or r.get("verdict") not in VERDICTS:
        return None
    if r.get("url") and str(r["url"]) not in text:
        r["url"] = None  # never keep a URL the report does not contain
    r["url"] = r.get("url") or None
    r["facts"] = {k: str(v) for k, v in (r.get("facts") or {}).items() if k in FACT_KEYS and str(v) and str(v) in text}
    flat = _ws(text)
    q = _ws(r.get("quote"))
    if not q or q not in flat or r["name"] not in q or r["name"] not in flat:
        return None
    r["quote"] = q
    return r


def catalog(vault, state, runner=None, workers=4, paths=None):
    runner = runner or default_runner(vault)
    todo = [(Path(p), proj) for p, proj in reports()] if paths is None else [(Path(p), None) for p in paths]
    docs, rows, dropped = {}, [], 0
    def work(item):
        return item, extract(item[0], vault, state, runner)
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(work, todo))
    projects = {}
    for (p, proj), (raw, text) in results:
        docs[str(p)] = text
        for r in raw:
            c = _clean(r, text)
            if c is None:
                dropped += 1
                continue
            rows.append((slug_of(p), str(p), c))
            projects.setdefault((slug_of(p)), proj)
    items = merge(rows)
    for it in items:
        it["projects"] = sorted({projects[m["research"]] for m in it["mentions"] if projects.get(m["research"])})
    cat = {"generated": date.today().isoformat(), "reports": len(todo), "items": items}
    errs = check_catalog(cat, docs)
    if errs:
        raise RuntimeError("catalog fails its own check:\n" + "\n".join(errs[:20]))
    out = Path(vault) / "wiki" / "_sources"
    try:
        cat["items"] = keep_captures(json.loads(_catalog_path(vault).read_text(encoding="utf-8"))["items"], cat["items"])
    except (OSError, ValueError, KeyError):
        pass
    _write_catalog(vault, cat)
    by_kind = {k: sum(1 for i in items if i["kind"] == k) for k in KINDS}
    return {"items": len(items), "by_kind": by_kind, "with_url": sum(1 for i in items if i["url"]),
            "mentions": sum(len(i["mentions"]) for i in items), "dropped": dropped, "reports": len(todo), "path": str(out / "catalog.json")}
