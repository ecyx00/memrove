"""E6: repo docs (docs/, docs/state/, extra roots) -> typed nodes + code-built edges. Read-only, never copied.

Nodes `doc:<project>/<relpath>`; edges (in the links table): about, day, mentions, solves, related.
Tables: docs(id PK, project, path, mtime, size, kind, title, summary, day, data). Incremental by mtime.
"""
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

ROOTS = ("docs",)
EXTRA_ENV = "MEMROVE_DOC_ROOTS"  # os.pathsep-separated, relative to repo or absolute
SKIP_DIRS = {".git", "node_modules", ".obsidian", "__pycache__", "venv", ".venv"}
MAX_BYTES = 64000
REL_CAP, RELATED_K, RELATED_MIN = 12, 3, 0.15
OWN_RELS = ("about", "day", "mentions", "solves", "related")
DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
WIKI = re.compile(r"\[\[([^\]|#]+)")
PATHISH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+)")
WORD = re.compile(r"[a-z0-9]{3,}")
# first match wins
KIND_RULES = (
    # Turkish: cozum/çözüm = solution.
    ("solution", re.compile(r"solution|resolved|fixed?\b|workaround|cozum|çözüm", re.I)),
    ("problem", re.compile(r"problem|\bbug\b|issue|incident|sorun|hata|regression|outage", re.I)),
    ("rule", re.compile(r"\brules?\b|convention|guideline|policy|kural|standards?\b|must\b", re.I)),
    # Turkish: gunluk/günlük = daily log.
    ("daylog", re.compile(r"daylog|journal|gunluk|günlük|standup|\bsummary\b|^\d{4}-\d{2}-\d{2}")),
    # Turkish: 'yap[ıi]' = structure.
    ("reference", re.compile(r"architecture|structure|platform|reference|design|overview|\bui\b|\bapi\b|layout|yap[ıi]|readme|index|guide", re.I)),
)
STOP = set("the and for with that this from are was were not but have has been will can into when then than also".split())


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _terms(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


FM_KEYS = ("topic", "status", "decision", "next", "supersedes")
FM_STATUS = ("open", "done", "rejected")


def front_matter(text):
    """Leading `---` block of a write-up -> {topic, status, decision, next, supersedes} (only the keys present and valid).
    One `key: value` per line, no nesting; deterministic, no YAML library."""
    m = re.match(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", text, re.S)
    out = {}
    for ln in (m.group(1).splitlines() if m else []):
        k, sep, v = ln.partition(":")
        k, v = k.strip().lower(), v.strip().strip("\"'").strip()
        if sep and k in FM_KEYS and v:
            out[k] = v
    if out.get("status", "").lower() not in FM_STATUS:
        out.pop("status", None)
    elif "status" in out:
        out["status"] = out["status"].lower()
    return out


def classify(rel, text):
    """Deterministic kind from filename, then title, then headings. No pattern -> note."""
    stem = Path(rel).stem
    title = _title(rel, text)
    heads = " ".join(re.findall(r"^#{1,3}\s+(.+)$", text, re.M))
    has_p = re.search(r"^#{1,3}\s+.*(problem|sorun)", text, re.M | re.I)
    # Turkish words (cozum/çözüm = solution) match Turkish documents.
    has_s = re.search(r"^#{1,3}\s+.*(solution|çözüm|cozum|fix)", text, re.M | re.I)
    if has_p and has_s:
        return "solution"  # self-contained pair; nothing to link across docs
    for probe in (stem, title, heads):
        for kind, rx in KIND_RULES:
            if rx.search(probe):
                return kind
    return "note"


def _title(rel, text):
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return (m.group(1) if m else Path(rel).stem).strip()[:120]


def _summary(text):
    body = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    for para in re.split(r"\n\s*\n", body):
        lines = [l for l in para.strip().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        if lines:
            return " ".join(" ".join(lines).split())[:300]
    return ""


def _day(rel, text):
    m = DATE.search(Path(rel).name) or re.search(r"^(?:date|day)\s*:\s*[\"']?(20\d\d-\d\d-\d\d)", text[:600], re.M | re.I) or DATE.search(rel)
    return m.group(1) if m else ""


def _files(repo, extra):
    roots = [repo / r for r in ROOTS] + [Path(p) if os.path.isabs(p) else repo / p for p in extra]
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for n in sorted(names):
                p = Path(dirpath) / n
                if n.lower().endswith(".md") and p not in seen:
                    seen.add(p)
                    yield p


SCHEMA = ("CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, project TEXT, path TEXT, mtime REAL, size INT, "
          "kind TEXT, title TEXT, summary TEXT, day TEXT, data TEXT)")


def _parse(repo, p, project, mtime, size):
    rel = p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.as_posix()
    with p.open(encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_BYTES)
    areas = []
    for m in PATHISH.findall(text):
        d = m.rsplit("/", 1)[0].strip("./")
        if d and (repo / d).is_dir() and d not in areas and d != rel.rsplit("/", 1)[0]:
            areas.append(d)
    tf = Counter(_terms(text))
    for w in _terms(_title(rel, text)):
        tf[w] += 3
    data = {"areas": areas[:8], "wiki": sorted(set(w.strip() for w in WIKI.findall(text)))[:8],
            "tf": dict(tf.most_common(80)),
            "sha": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16], "fm": front_matter(text)}
    return (f"doc:{project}/{rel}", project, rel, mtime, size, classify(rel, text), _title(rel, text),
            _summary(text), _day(rel, text), json.dumps(data, sort_keys=True))


def _bm25(rows, data):
    """-> {(a,b): normalised score} for a<b, symmetric max of both directions."""
    n = len(rows)
    df = Counter(t for r in rows for t in data[r[0]]["tf"])
    idf = {t: math.log(1 + (n - c + .5) / (c + .5)) for t, c in df.items()}
    avg = sum(sum(data[r[0]]["tf"].values()) for r in rows) / max(1, n)

    def score(q, d):
        dl, tf = sum(d.values()), d
        return sum(idf[t] * tf[t] * 2.2 / (tf[t] + 1.2 * (.25 + .75 * dl / avg)) for t in q if t in tf)

    out = {}
    for a in rows:
        qa, ta = data[a[0]]["tf"], data[a[0]]["tf"]
        selfs = score(qa, ta) or 1
        for b in rows:
            if a[0] < b[0]:
                s = max(score(qa, data[b[0]]["tf"]) / selfs,
                        score(data[b[0]]["tf"], qa) / (score(data[b[0]]["tf"], data[b[0]]["tf"]) or 1))
                out[(a[0], b[0])] = s
    return out


def _stem_key(rel):
    s = Path(rel).stem.lower()
    for rx in (KIND_RULES[0][1], KIND_RULES[1][1]):
        s = rx.sub("", s)
    return _slug(s)


def derive_edges(repo, project, rows, add):
    """rows: docs rows of one project. Calls add(src, dst, rel, w)."""
    data = {r[0]: json.loads(r[9]) for r in rows}
    by_stem = {}
    for r in rows:
        by_stem.setdefault(Path(r[2]).stem.lower(), r[0])
    deg = Counter()

    def edge(src, dst, rel, w):
        add(src, dst, rel, w)
        deg[src] += 1

    for r in rows:
        id_, rel, kind, day = r[0], r[2], r[5], r[8]
        edge(id_, f"project:{project}", "about", 1.0)
        if day:
            edge(id_, f"day:{day}", "day", 0.6)
        for a in data[id_]["areas"]:
            edge(id_, f"area:{a}", "mentions", 0.5)
        for w in data[id_]["wiki"][:4]:
            tgt = by_stem.get(Path(w).stem.lower())
            if tgt and tgt != id_:
                edge(id_, tgt, "mentions", 0.7)
    problems = [r for r in rows if r[5] == "problem"]
    for s in (r for r in rows if r[5] == "solution"):
        best, bw = None, 0.0
        for p in problems:
            if _stem_key(s[2]) and _stem_key(s[2]) == _stem_key(p[2]):
                w = 1.0
            elif s[8] and s[8] == p[8]:
                a, b = set(data[s[0]]["tf"]), set(data[p[0]]["tf"])
                w = len(a & b) / max(1, len(a | b))
                w = w if w >= 0.2 else 0.0
            else:
                w = 0.0
            if w > bw:
                best, bw = p, w
        if best:
            edge(s[0], best[0], "solves", bw)
    scores = _bm25(rows, data)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    nb = Counter()
    linked = set()
    for (a, b), s in ranked:
        if s >= RELATED_MIN and nb[a] < RELATED_K and nb[b] < RELATED_K and deg[a] < REL_CAP and deg[b] < REL_CAP:
            edge(a, b, "related", min(1.0, s))
            edge(b, a, "related", min(1.0, s))
            nb[a] += 1
            nb[b] += 1
            linked |= {a, b}
    # no unlinked docs: anything with only `about` gets its single nearest neighbour
    for r in rows:
        if deg[r[0]] <= 1 and len(rows) > 1:
            cand = [(s, p) for p, s in scores.items() if r[0] in p]
            if cand:
                s, p = max(cand, key=lambda x: (x[0], x[1]))
                other = p[1] if p[0] == r[0] else p[0]
                edge(r[0], other, "related", max(s, 0.05))


def scan(store, repo, extra=None, budget_s=0.5):
    """Incremental by mtime. Returns {'docs','changed','removed','deferred'}. Never raises for a missing repo."""
    repo = Path(repo).resolve()
    res = {"docs": 0, "changed": 0, "removed": 0, "deferred": 0}
    if store is None or store.read_only or not repo.is_dir():
        return res
    import memrove_v3_links as links
    extra = list(extra or []) + [p for p in os.environ.get(EXTRA_ENV, "").split(os.pathsep) if p]
    project = repo.name
    deadline = time.monotonic() + budget_s
    lk = links.Links(store)
    with store._connect() as db:
        db.execute(SCHEMA)
        have = {r[0]: (r[1], r[2]) for r in db.execute("SELECT id,mtime,size FROM docs WHERE project=?", (project,))}
        live, changed = set(), 0
        for p in _files(repo, extra):
            try:
                st = p.stat()
                rel = p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.as_posix()
                id_ = f"doc:{project}/{rel}"
                live.add(id_)
                if have.get(id_) == (st.st_mtime, st.st_size):
                    continue
                if time.monotonic() > deadline:
                    res["deferred"] += 1
                    continue
                db.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)", _parse(repo, p, project, st.st_mtime, st.st_size))
                changed += 1
            except OSError:
                continue
        gone = set(have) - live
        for id_ in gone:
            db.execute("DELETE FROM docs WHERE id=?", (id_,))
        if changed or gone:  # neighbours depend on the whole corpus: rebuild this project's edges
            db.execute("DELETE FROM links WHERE src LIKE ? AND rel IN (%s)" % ",".join("?" * len(OWN_RELS)),
                       (f"doc:{project}/%", *OWN_RELS))
            rows = db.execute("SELECT * FROM docs WHERE project=? ORDER BY id", (project,)).fetchall()
            derive_edges(repo, project, rows, lambda s, d, r, w: lk.add(s, d, r, w, db=db))
        res.update(docs=len(live), changed=changed, removed=len(gone))
    return res


def all_docs(db, project=None):
    db.execute(SCHEMA)
    q, a = "SELECT * FROM docs" + (" WHERE project=?" if project else "") + " ORDER BY id", ((project,) if project else ())
    return db.execute(q, a).fetchall()


def all_edges(db):
    try:
        return db.execute("SELECT src,dst,rel,w FROM links WHERE src LIKE 'doc:%' ORDER BY src,rel,dst").fetchall()
    except Exception:
        return []


def doctor(store, transcript_for=None):
    """Doc graph health: per-doc degree, unlinked % (no edge but `about`), injected pointers, and how many injected
    docs the model later Read/Bash-ed per the transcript (None when no transcript is found)."""
    if store is None:
        return {}
    with store._connect() as db:
        rows = all_docs(db)
        edges = all_edges(db)
        try:
            inj = db.execute("SELECT session, block_id FROM injected WHERE block_id LIKE 'doc:%'").fetchall()
        except Exception:
            inj = []
    deg = Counter(s for s, d, rel, _ in edges if rel != "about")
    n = len(rows)
    unlinked = sum(1 for r in rows if not deg[r[0]])
    paths = {r[0]: r[2] for r in rows}
    read = known = 0
    seen = set()
    for sess, bid in inj:
        if (sess, bid) in seen or bid not in paths:
            continue
        seen.add((sess, bid))
        text = (transcript_for or (lambda s: _find_transcript(s, store)))(sess)
        if text is None:
            continue
        known += 1
        read += int(_was_read(text, paths[bid]))
    return {"docs": n, "edges": len(edges), "unlinked": unlinked, "unlinked_pct": round(100 * unlinked / n, 1) if n else 0.0,
            "avg_degree": round(sum(deg.values()) / n, 2) if n else 0.0, "max_degree": max(deg.values(), default=0),
            "injected_pointers": len(seen), "injected_read": read if known else None, "injected_read_checked": known}


def _find_transcript(session, store=None):
    """`session` is the hashed id the hooks use; the map written by the worker gives the raw id / path."""
    try:
        entry = json.loads((Path(store.state_dir) / "session-map.json").read_text(encoding="utf-8")).get(session) or {}
    except (OSError, ValueError, AttributeError):
        entry = {}
    if entry.get("path") and os.path.isfile(entry["path"]):
        return Path(entry["path"]).read_text(encoding="utf-8", errors="replace")
    for p in (Path.home() / ".claude" / "projects").glob(f"*/{entry.get('raw') or session}.jsonl"):
        return p.read_text(encoding="utf-8", errors="replace")
    return None


def _was_read(text, path):
    for line in text.splitlines():
        if path in line and re.search(r'"name"\s*:\s*"(Read|Bash|Grep|Glob)"', line):
            return True
    return False
