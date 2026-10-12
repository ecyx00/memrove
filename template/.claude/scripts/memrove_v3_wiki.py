"""Concept wiki tools for the wiki-writer agent (no LLM here).
`memrove wiki evidence --topic "Title::terms=a,b::sym=..::docs=..::msg=id,id::chat=q|q::roots=proj=/abs,..::skip=w"` -> one labelled evidence pack on stdout (<= PACK_CAP chars).
`memrove wiki check <page>` -> exit 0 when every source tag resolves, [general] lines hold no numbers and the structure holds.
`memrove wiki publish <page>` -> check, then write the page's entry into the index.json of every project in its front matter.
"""
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

import memrove_v3_chatlog as chatlog
import memrove_v3_chatsearch as chatsearch
import memrove_v3_sources as sources

TAG = re.compile(r"\[(doc|commit|memrove|m|grab):([^\]\s]+)\]")
GRAB_ID = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*")  # card file stem; the card is wiki/_sources/grab/<id>.md
GENERAL = "[general]"
INLINE_CODE = re.compile(r"`[^`]*`")
MATH = re.compile(r"(?<!\\)\$[^$\n]+?(?<!\\)\$")  # inline $...$ (an escaped \$ is a literal dollar)
NAME_WITH_DIGIT = re.compile(r"\b[A-Za-z][A-Za-z-]*\d+[A-Za-z\d]*\b")  # BM25, Qwen3, k1: a name, not a quantity
CODE = re.compile(r"\[[^\]]*\]\(code:([^)\s]+)\)")
SLUG = re.compile(r"\[\[([a-z0-9-]+)\]\]")
SECTIONS = ["Problem", "Skeleton", "Flow", "Solution and cost", "Key distinctions", "What we researched", "Ours", "Where you see it",
            "Subtopics to know", "Compression"]
# a backticked token that is only a quantity: 1024, 40%, 16x, 3 token, 1/61 + 1/63, 2.5ms (an identifier such as k1 or potion-code-16M is not)
QUANT_UNIT = re.compile(r"(?<=\d)\s*(?:ms|kb|mb|gb|tb|k|m|b|s|x|×|bytes?|tokens?|dims?|d)(?![A-Za-z])", re.I)
DOC_HEAD, DOC_TAIL, MAX_DOCS = 4500, 3500, 3
RECORD_KINDS = ("decision", "rejected", "lesson")
PACK_CAP = 30000
CODE_INDEX = os.environ.get("MEMROVE_CODE_INDEX", "code-index")  # external symbol-lookup CLI; absent = anchors skipped
RESEARCH_DIR = Path.home() / "Desktop/notes/docs/research"
SRC = sources.SRC


def pure_number(token):
    t = QUANT_UNIT.sub("", token)
    return bool(re.search(r"\d", t)) and re.fullmatch(r"[\d\s.,_%/:+\-–−*^()~≈<>=]*", t) is not None


def slugify(title):
    t = unicodedata.normalize("NFKD", title.replace("ı", "i").replace("İ", "I")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-") or "page"


def parse_topic(spec):
    """'Title::key=a,b::key=...' -> dict. Keys: terms, sym (symbols), docs (file names under docs/state), rec (pinned memrove ids),
    chat (queries separated by |), msg (pinned chat message ids), roots (project=abs path, ...; sym entries may be project:symbol), skip (words that mark evidence as belonging to another topic). A bare 'Title::t1,t2::s1,s2' still works."""
    parts = spec.split("::")
    t = {"title": parts[0].strip(), "terms": [], "sym": [], "docs": [], "rec": [], "chat": [], "msg": [], "skip": [], "roots": []}
    for n, part in enumerate(parts[1:]):
        key, sep, val = part.partition("=")
        if not sep or key not in t:
            key, val = ("terms", "sym")[n] if n < 2 else "terms", part
        t[key] = [x.strip() for x in val.split("|" if key == "chat" else ",") if x.strip()]
    t["terms"] = t["terms"] or t["title"].lower().split()
    return t


def _run(cmd, cwd, timeout=20):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""


def _clip(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


# ---- evidence ------------------------------------------------------------------------------------------------------------

def _owned(text, topic):
    return not any(w.lower() in text.lower() for w in topic["skip"])


def commit_evidence(root, topic, files, limit=15):
    seen, out = set(), []
    for t in topic["terms"] + topic["sym"]:
        _, o = _run(["git", "log", "-i", "--fixed-strings", f"--grep={t}", "--format=%h|%cs|%s", "-n", "30", "--", *files], root)
        for line in o.splitlines():
            sha = line.split("|", 1)[0]
            if re.fullmatch(r"[0-9a-f]{7,}", sha) and sha not in seen and _owned(line, topic):
                seen.add(sha)
                out.append(line)
    return sorted(out, key=lambda l: l.split("|")[1], reverse=True)[:limit]


def record_evidence(store, topic, limit=8):
    """Pinned ids when the topic names them; else a content match on the topic's symbols and the project name."""
    import memrove_v3_recall as recall
    ids = list(topic["rec"])
    if not ids:
        for hit in recall.search(store, " ".join(topic["terms"]), 40):
            if hit["kind"] in RECORD_KINDS and hit["id"] not in ids:
                ids.append(hit["id"])
    out = []
    for rid in ids:
        try:
            r = recall.show(store, rid)
        except ValueError:
            continue
        body = f"{r.get('text')} {r.get('why')}"
        if not _owned(body, topic) or (not topic["rec"] and not any(m.lower() in body.lower() for m in topic["sym"] + topic["terms"])):
            continue
        out.append(f"[memrove:{r['id']}] kind={r.get('kind')} status={r.get('status')} date={r.get('valid_from')} "
                   f"text={_clip(r.get('text'), 600)} why={_clip(r.get('why') or '', 300)}")
        if len(out) >= limit:
            break
    return out


USER_CLIP, ASSIST_CLIP, CHAT_LIMIT = 1200, 2000, 14


def chat_evidence(state, topic, name, limit=CHAT_LIMIT):
    """The core of a concept page. Pinned message ids first (msg=), then explicit queries (Turkish/English). The discussion is spread over
    several long assistant messages, so they are clipped long; the owner's own messages are kept first."""
    rows, seen = [], set()
    for i in topic["msg"]:
        r = chatlog.get(state, int(i)) if i.isdigit() else None
        if r and int(i) not in seen:
            seen.add(int(i))
            rows.append((int(i), (r.get("date") or "")[:10], r.get("kind"), r.get("text") or ""))
    for q in topic["chat"] or ([] if topic["msg"] else [" ".join(topic["terms"])]):
        for i, d, kind, text in chatsearch.hits(state, q, top=6):
            if i not in seen and _owned(text, topic):
                seen.add(i)
                rows.append((i, (d or "")[:10], kind, text))
    rows.sort(key=lambda r: (r[2] != "user", r[0]))
    return [f"[m:{i}] {kind} {d} {_clip(text, USER_CLIP if kind == 'user' else ASSIST_CLIP)}" for i, d, kind, text in rows[:limit]]


def anchor_evidence(root, symbols, roots=None, primary=None):
    """-> (anchor lines, files, code blocks, env vars read inside the anchored functions). Facts from code, not from the model.
    A symbol 'project:name' is looked up in roots[project]; links to it are code:@project/relpath."""
    lines, files, code, env = [], [], [], set()
    for s in symbols:
        proj, _, sym = s.partition(":") if ":" in s else (primary, "", s)
        r = Path((roots or {}).get(proj, root))
        ext = proj != primary and proj in (roots or {})
        rc, o = _run([CODE_INDEX, "def", sym], r)
        first = o.strip().splitlines()[:1] if rc == 0 else []
        m = re.match(r"(\S+?):(\d+)\s", first[0]) if first else None
        if not m:
            lines += first
            continue
        path, lo = m.group(1), int(m.group(2))
        lines.append(f"@{proj}/{first[0]}" if ext else first[0])
        if not ext and path not in files:
            files.append(path)
        src = (r / path).read_text(encoding="utf-8", errors="replace").splitlines()[lo - 1:]
        body = [src[0]]
        for l in src[1:80]:
            if re.match(r"(def|class) ", l):
                break
            body.append(l)
        text = "\n".join(body)
        env |= set(re.findall(r"[A-Z]+_[A-Z_]{3,}", text)) if False else set(re.findall(r"CODEINDEX_[A-Z_]+", text))
        code.append(f"CODE {s} [doc:{r / path if ext else path}] (link: code:{'@' + proj + '/' if ext else ''}{path}#{sym}):\n" + text[:2600])
    return lines, files, code, sorted(env)


def topic_roots(topic):
    return {k.strip(): str(Path(v.strip()).expanduser()) for k, _, v in (x.partition("=") for x in topic["roots"]) if v}


def existing_concepts(vault):
    d = Path(vault) / "wiki" / "_concepts"
    return sorted(p.stem for p in d.glob("*.md")) if d.is_dir() else []


def _fit(lines, budget):
    out, used = [], 0
    for l in lines:
        if used + len(l) + 1 > budget:
            out.append("[... cut: budget ...]")
            break
        out.append(l)
        used += len(l) + 1
    return "\n".join(out)


def _cap(text):
    return text if len(text) <= PACK_CAP else text[:PACK_CAP - 30] + "\n[... pack cut ...]"


def matched_docs(roots, topic):
    """-> [(path, score, [matched lines])] best first: each root's docs/state and the research folder; docs= names restrict the root docs."""
    dirs = [Path(r) / "docs/state" for r in roots.values()] + [RESEARCH_DIR]
    rows = []
    for d in dirs:
        for p in sorted(d.glob("*.md")) if d.is_dir() else []:
            if topic["docs"] and d != RESEARCH_DIR and not any(n in p.name for n in topic["docs"]):
                continue
            lines = [" ".join(l.split()) for l in p.read_text(encoding="utf-8", errors="replace").splitlines()]
            hit = [l for l in lines if any(t.lower() in l.lower() for t in topic["terms"]) and _owned(l, topic)]
            if hit:
                rows.append((str(p), len(hit), hit))
    return sorted(rows, key=lambda r: -r[1])


def _fetch_hint(item_id):
    kind, _, rest = item_id.partition(":")
    return {"arxiv": f"paper:{rest}", "github": f"repo:{rest}", "hf": f"model:{rest}"}.get(kind)


def catalog_evidence(vault, doc_paths):
    cat = Path(vault) / "wiki" / "_sources" / "catalog.json"
    try:
        items = json.loads(cat.read_text(encoding="utf-8"))["items"]
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for it in items:
        ms = [m for m in it.get("mentions", []) if m.get("doc") in doc_paths]
        if not ms:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", it["id"].split(":", 1)[-1].lower().rsplit("/", 1)[-1]).strip("-")
        snap = next((c for c in (slug, slugify(it.get("title") or "")) if sources.snap_dir(vault, c).is_dir()), None)
        hint = _fetch_hint(it["id"])
        out.append(f"CATALOG {it['id']} kind={it.get('kind')} url={it.get('url')} "
                   + (f"snapshot: {snap}" if snap else f"snapshot: none (fetch: memrove wiki source fetch {hint} --slug {slug})")
                   + "".join(f"\n  [doc:{m['doc']}] verdict={m.get('verdict')} quote: {m.get('quote')}" for m in ms[:3]))
    return out


def evidence_pack(store, state, vault, topic):
    """One evidence pack, labelled blocks the writer can cite, no LLM. Capped at PACK_CAP chars."""
    roots = topic_roots(topic)
    primary = next(iter(roots), None)
    docs = matched_docs(roots, topic)
    chat = chat_evidence(state, topic, primary) if state is not None or topic["msg"] else []
    anchors, code = [], []
    for sym in topic["sym"]:
        proj, sep, name = sym.partition(":") if ":" in sym else (None, "", sym)
        for r in ([proj] if proj in roots else [] if proj else list(roots)):
            ls, _, cb, _ = anchor_evidence(roots[r], [f"{r}:{name}"], roots, primary)
            anchors += [f"@{r}/{l}" if not l.startswith("@") else l for l in ls if re.match(r"@?\S+?:\d+\s", l)]
            code += cb
    doc_lines = []
    for path, score, hit in docs[:8]:
        doc_lines.append(f"DOC [doc:{path}] matches={score}")
        doc_lines += [f"  {_clip(l, 260)}" for l in hit[:12]]
    blocks = [
        f"TOPIC: {topic['title']}\nTERMS: {', '.join(topic['terms'])}\nROOTS: " + (", ".join(f"{k}={v}" for k, v in roots.items()) or "none"),
        "CHAT MESSAGES (kind user = the owner's own words; cite as [m:id]):\n" + _fit(chat, 14000) if chat else "CHAT MESSAGES: none",
        "MEMROVE RECORDS (cite as shown):\n" + (_fit(record_evidence(store, topic), 3000) if store is not None or topic["rec"] else "none"),
        "REPORTS (docs/state of each root + research folder; matched lines):\n" + (_fit(doc_lines, 6000) or "none"),
        "CATALOG (items named in those reports; quote is verbatim from the report):\n" + (_fit(catalog_evidence(vault, {d[0] for d in docs}), 4000) or "none"),
        "CODE ANCHORS (code index def; link as code:relpath#symbol, code:@project/relpath#symbol for another root):\n" + (_fit(anchors + code, 3000) or "none"),
    ]
    return _cap("\n\n".join(blocks))


# ---- check (0 tokens) ----------------------------------------------------------------------------------------------------

def _front(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    meta = dict(l.split(": ", 1) for l in m.group(1).splitlines() if ": " in l) if m else {}
    return meta, text[m.end():] if m else text


def _sections(body):
    out, cur = {}, None
    for line in body.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return out


def _is_claim(line):
    s = line.strip()
    return s.startswith("- ") or s.startswith("* ") or (s.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", s))


def _is_header_row(lines, n):
    """lines[n-1] is a table header when the next line is the |---| separator."""
    return n < len(lines) and re.match(r"^\|[\s:|-]+\|?$", lines[n].strip()) is not None


def check(page, store, state, vault=None):
    """-> list of error strings; empty = the page may be published."""
    import memrove_v3_recall as recall
    text = Path(page).read_text(encoding="utf-8")
    meta, body = _front(text)
    root = meta.get("root")
    if not root and meta.get("project") == "_general":
        root = str(vault if vault and Path(vault).is_dir() else Path.home())  # a cross-project page has no repo; doc tags then need absolute paths
    errors, cache = [], {}
    if not root or not Path(root).is_dir():
        return [f"frontmatter root missing or not a directory: {root!r}"]

    def resolve(kind, ident):
        """-> (ok, date or None, status or None); cached."""
        key = (kind, ident)
        if key in cache:
            return cache[key]
        res = (False, None, None)
        if kind == "doc":
            p = Path(ident) if ident.startswith("/") else Path(root) / ident
            if p.is_file():
                m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
                res = (True, m.group(1) if m else None, None)
        elif kind == "commit":
            if re.fullmatch(r"[0-9a-f]{7,40}", ident):
                rc, o = _run(["git", "show", "-s", "--format=%cs", ident + "^{commit}"], root)
                if rc == 0 and o.strip():
                    res = (True, o.strip()[:10], None)
        elif kind == "memrove":
            try:
                r = recall.show(store, ident)
                if str(r.get("id")) == ident:
                    res = (True, r.get("valid_from"), r.get("status"))
            except Exception:
                pass
        elif kind == "grab":
            if vault and GRAB_ID.fullmatch(ident) and (Path(vault) / "wiki" / "_sources" / "grab" / f"{ident}.md").is_file():
                res = (True, ident[:10], None)
        elif kind == "m":
            try:
                r = chatlog.get(state, int(ident)) if ident.isdigit() else None
                if r is not None:
                    res = (True, (r.get("date") or "")[:10] or None, None)
            except Exception:
                pass
        cache[key] = res
        return res

    for kind, ident in dict.fromkeys(TAG.findall(body)):
        if not resolve(kind, ident)[0]:
            errors.append(f"unresolved tag [{kind}:{ident}]")

    secs = _sections(body)
    heads = [h for h in secs]
    if [h for h in heads if h in SECTIONS] != SECTIONS:
        errors.append(f"sections must be exactly, in order: {SECTIONS}; found {heads}")
    for needle in ("Test yourself", "Layer 4"):
        if needle not in body:
            errors.append(f"missing <details> block: {needle}")

    for slug in dict.fromkeys(m[0] for m in SRC.findall(body)):  # one source per line; check_src judges that slug's lines only
        mine = ["" if not SRC.search(l) or {t[0] for t in SRC.findall(l)} != {slug} else l for l in body.splitlines()]
        errors += [f"[src:{slug}] {e}" for e in sources.check_src("\n".join(mine), slug, vault)] if vault else [f"[src:{slug}] needs the vault to resolve"]
    for n, l in enumerate(body.splitlines(), 1):
        if len({t[0] for t in SRC.findall(l)}) > 1:
            errors.append(f"line {n}: tags of two sources on one line")

    in_test, cur, own = False, None, False
    lines = body.splitlines()
    for n, line in enumerate(lines, 1):
        if line.startswith("## "):
            cur, own = line[3:].strip(), False
        if cur == "Ours" and "Our measurement" in line:
            own = True
        if "Test yourself" in line:
            in_test = True
        elif "Layer 4" in line or "</details>" in line:
            in_test = False
        if in_test or cur is None or not _is_claim(line):
            continue
        if GENERAL in line and any(pure_number(c.strip("`")) for c in INLINE_CODE.findall(line)):
            errors.append(f"line {n}: [general] line has a number in backticks (a pure quantity is not an identifier): {line.strip()[:70]}")
        if GENERAL in line and re.search(r"\d", NAME_WITH_DIGIT.sub("", TAG.sub("", CODE.sub("", MATH.sub("", INLINE_CODE.sub("", line)))))):
            errors.append(f"line {n}: [general] line has a number outside inline code: {line.strip()[:70]}")
        ok_code = cur == "Where you see it" and CODE.search(line)
        ok_link = cur == "Subtopics to know" and SLUG.search(line) and GENERAL in line
        src = TAG.search(line) or SRC.search(line)
        if "No source." in line:
            continue
        if own:
            if _is_header_row(lines, n):
                continue
            if not src:
                errors.append(f"line {n}: Our measurement line needs a real source tag ([m:]/[doc:]/[memrove:]/[commit:]), [general] does not count: {line.strip()[:70]}")
            continue
        if not src and GENERAL not in line and not ok_code and not ok_link:
            errors.append(f"line {n}: claim without source tag or [general]: {line.strip()[:70]}")

    try:
        roots = json.loads(meta.get("roots") or "{}")
    except ValueError:
        roots = {}
        errors.append("frontmatter roots must be a JSON object {project: path}")
    for rel in dict.fromkeys(CODE.findall(body)):
        base = Path(root)
        if rel.startswith("@"):
            proj, _, rel = rel[1:].partition("/")
            if proj not in roots or not Path(roots[proj]).is_dir():
                errors.append(f"code link: unknown project @{proj} (not in frontmatter roots)")
                continue
            base = Path(roots[proj])
        path, sym = rel, None
        if "#" in rel:
            path, sym = rel.split("#", 1)
        elif re.search(r":\d+$", rel):
            path = rel.rsplit(":", 1)[0]
        if not (base / path).is_file():
            errors.append(f"code link: file not found {path}")
        elif sym:
            rc, o = _run([CODE_INDEX, "def", sym], base)
            if rc == 0 and not o.startswith(CODE_INDEX + ":"):
                if o.startswith("no symbol") or path not in o:
                    errors.append(f"code link: symbol {sym} not defined in {path} (code index def)")

    wiki_dir = Path(page).parent
    homes = [wiki_dir] + ([Path(vault) / "wiki" / "_concepts"] if vault else [])
    for slug in dict.fromkeys(SLUG.findall(body)):
        if not any((h / f"{slug}.md").exists() for h in homes):
            errors.append(f"wiki link [[{slug}]]: no such page")

    return list(dict.fromkeys(errors))


# ---- build ---------------------------------------------------------------------------------------------------------------

def _index_entry(root_dir, meta, group):
    """Write or replace the page's entry in <root_dir>/index.json; the page file lives in <vault>/wiki/_concepts."""
    root_dir.mkdir(parents=True, exist_ok=True)
    ip = root_dir / "index.json"
    try:
        index = json.loads(ip.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    index.update(project=meta["project"], root=meta["root"], generated=meta["updated"])
    groups = index.setdefault("groups", [])
    g = next((g for g in groups if g["title"] == group), None) or groups.append({"title": group, "pages": []}) or groups[-1]
    entry = {"slug": meta["slug"], "title": meta["title"], "one_liner": meta["one_liner"], "updated": meta["updated"], "status": "ok",
             "file": f"../_concepts/{meta['slug']}.md"}
    g["pages"] = [p for p in g["pages"] if p["slug"] != meta["slug"]] + [entry]
    ip.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish(page, store, state, vault):
    """check, then list the page in the index of every project in its front matter. -> errors ([] = published). Idempotent."""
    errors = check(page, store, state, vault)
    if errors:
        return errors
    page = Path(page)
    meta, body = _front(page.read_text(encoding="utf-8"))
    try:
        roots = json.loads(meta.get("roots") or "{}")
    except ValueError:
        roots = {}
    proj = meta.get("project") or meta.get("root") and Path(meta["root"]).name
    if not roots and proj:
        roots = {proj: meta.get("root") or None}
    if meta.get("project") == "_general":
        roots = {"_general": None}
    h1 = re.search(r"^# (.+)$", body, re.M)
    first = next((l.strip() for l in body[h1.end():].splitlines() if l.strip()), "") if h1 else ""
    one_liner = (re.match(r".*?[.!?](?=\s|$)", first) or re.match(r".*", first)).group(0)
    m = {"slug": page.stem, "title": meta.get("title") or (h1.group(1) if h1 else page.stem), "one_liner": one_liner,
         "updated": meta.get("updated") or "", }
    for name, path in roots.items():
        _index_entry(Path(vault) / "wiki" / name, {**m, "project": name, "root": path}, meta.get("group") or "Concepts")
    return []
