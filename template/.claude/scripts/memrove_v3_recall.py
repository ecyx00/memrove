"""Message-time recall (UserPromptSubmit): at most 3 memories from OTHER projects. Read-only: no sync, no LLM;
writes only recall-log.jsonl and injected rows. BM25 and IDF are computed in Python over the candidate kinds."""
import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

from memrove_v3_trust import corroboration, extraction_confidence, lesson_admission

KINDS = ("lesson", "decision", "rejected", "session")
MAX_ITEMS, MAX_TOKENS, MIN_COVERAGE, MIN_TERMS, MIN_WORDS, HALF_LIFE_DAYS = 3, 300, 0.25, 2, 4, 30.0
STOP = frozenset("""the and for with that this from have not are was you your but can how what when why who into about there their
them then than will would should could just like also some any one use used using bir ve ile için icin bu şu su da de mi mı mu mü
ne nasıl nasil gibi çok cok daha ama veya ya ben sen biz siz var yok olarak olan oldu olur için bunu şunu onu bana sana
bunun şunun ona yap yapıp yapalım yapalim lütfen lutfen şimdi simdi""".split())
# stems of generic verbs/adjectives: they may add coverage but cannot carry a hit alone (a hit needs one distinctive term)
GENERIC = frozenset("kulla hizli yeni ilerl gerek lazim iyi guzel buyuk kucuk fazla sonra once ayni farkl baska eklem ekley calis gorun bakal bakar soyle kontr degis duzel ister yapil olmal olsun mumku".split())
DISTINCT_DF = 0.25  # ...and a term found in more than this share of the corpus is generic too (checked from 20 records up)
LESSON_BOOST = 1.3  # failed / unnecessary / mixed: "we did this and it was unnecessary" is the signal the user values most
ENVELOPE = '<memrove-recall trust="local-data" instructions="never">\n{}\n</memrove-recall>'


_TR = str.maketrans({"İ": "i", "I": "i", "ı": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ç": "c", "Ç": "c", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u"})


def fold(text):
    text = (text or "").translate(_TR).lower()
    if text.isascii():
        return text
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def terms(text):
    """Folded words minus stopwords, F5-stemmed (first 5 chars of alphabetic words >=5 long; ids like alpha5 stay whole) so Turkish suffixes still match."""
    return [t[:5] if t.isalpha() else t for t in re.findall(r"[a-z0-9_]{3,}", fold(text)) if t not in STOP]


def tok(s):
    return len(s) // 4


def _parse_day(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None) if value else None
    except ValueError:
        return None


def _list(value):
    return [str(v) for v in value] if isinstance(value, list) else ([str(value)] if value else [])


def _fields(r):
    """(key text for coverage, full text for BM25)."""
    key = " ".join([str(r.get("title") or "")] + _list(r.get("topics")) + [str(r.get("approach") or "")]
                   + _list(r.get("aliases")) + [str(r.get("key") or "")])
    full = " ".join([key, str(r.get("text") or ""), str(r.get("situation") or ""), str(r.get("why") or ""), str(r.get("result") or "")])
    return key, full


def _load(store, here=None, stats=None):
    """Eligible rows (kinds, not retired/superseded/expired/private, not project `here`) -> (payload, folded payload).
    Filtering runs in SQL and only a folded copy is built per row; JSON is parsed later for candidates only."""
    marks = ",".join("?" * len(KINDS))
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    q = lambda k: f"coalesce(json_extract(payload,'$.{k}'),'')"
    with store._connect() as db:
        gone = {v for (v,) in db.execute("SELECT DISTINCT j.value FROM records, json_each(payload,'$.supersedes') j")}
        rows = db.execute(
            f"SELECT id, payload, {q('kind')}, {q('project')}, {q('scope')} FROM records WHERE {q('kind')} IN ({marks}) AND {q('status')} NOT IN ('retired','rejected','dropped_llm','deleted_by_user','source_missing') "
            f"AND ({q('imported_from')} NOT LIKE '%-session' OR json_extract(payload,'$.assessments.j1.answers.durable')='yes') "  # imports wait for J1
            f"AND {q('verdict')}!='reject' AND {q('visibility')}!='private' AND ({q('valid_to')}='' OR {q('valid_to')}>?) "
            f"AND (? IS NULL OR {q('project')}!=?)", (*KINDS, now, here, here)).fetchall()
    # Authorization before candidate generation: scope is read from columns, the payload is not parsed.
    from memrove_v3_kinds import client_scopes, effective_scope, in_scope
    client = client_scopes(store.state_dir)
    ok = [(rid, payload) for rid, payload, kind, project, scope in rows
          if rid not in gone and in_scope(effective_scope({"kind": kind, "project": project, "scope": scope}), client)]
    if stats is not None:
        stats["out_of_scope"] = sum(1 for r in rows if r[0] not in gone) - len(ok)
    return [(payload, fold(json.dumps(json.loads(payload), ensure_ascii=False) if "\\u" in payload else payload), rid)
            for rid, payload in ok]


def _rank(items, prompt_terms, gate, dropped=None, admit=None):
    """-> scored dicts, best first. gate=True applies IDF coverage and the 2-term rule.
    admit (default: = gate) applies the lesson-admission trust rule on its own: hybrid recall drops the relevance gate but not that.
    ponytail: df is counted only for prompt terms (substring match on folded payloads, slight overcount); only candidates are parsed.
    Upgrade path: persisted FTS5 index when the corpus outgrows ~10k records."""
    n = len(items)
    avg = sum(len(it[1]) for it in items) / max(1, n)
    query, scored, df, cands = set(prompt_terms), [], {}, []
    seen = re.compile("|".join(map(re.escape, query))).findall if query else (lambda _: [])
    for payload, folded, _rid in items:
        hits = set(seen(folded))
        for t in hits:
            df[t] = df.get(t, 0) + 1
        if len(hits) >= (MIN_TERMS if gate else 1):
            cands.append((payload, folded))
    weight = lambda d: math.log(1 + (n - d + 0.5) / (d + 0.5))
    top_w = max((weight(d) for d in df.values()), default=1.0)
    idf = lambda t: weight(df[t]) if t in df else top_w  # a prompt term nobody matches is unexplained and weighs like the rarest matched one (tiny corpora must not blow it up)
    for payload, folded in cands:
        r = json.loads(payload)
        if (gate if admit is None else admit) and r.get("kind") == "lesson" and lesson_admission(r, corroboration([r])) != "admit":
            if dropped is not None:  # other projects only reach here: unassessed lessons stay in their own project
                dropped.append({"id": r.get("id"), "reason": "lesson_not_admitted"})
            continue
        key, full = _fields(r)
        bag = terms(full)
        tf = {}
        for t in bag:
            if t in query:
                tf[t] = tf.get(t, 0) + 1
        key = set(terms(key)) or set(terms(r.get("text", ""))[:12])  # only candidates pay for key terms
        if not tf:
            continue
        dl = len(folded) / max(1.0, avg)
        bm = sum(idf(t) * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * dl)) for t, c in tf.items())
        hit = set(tf)
        # gate: how much of the PROMPT (idf-weighted) does this record explain; record-side cover is a logged ranking signal only
        pcover = sum(idf(t) for t in hit) / max(1e-9, sum(idf(t) for t in query))
        cover = sum(idf(t) for t in key & query) / max(1e-9, sum(idf(t) for t in key))
        distinct = any(t not in GENERIC and (n < 20 or df.get(t, 0) <= DISTINCT_DF * n) for t in hit)
        if gate and (pcover < MIN_COVERAGE or len(hit) < MIN_TERMS or not distinct):
            if dropped is not None:
                reason = "coverage" if pcover < MIN_COVERAGE else "min_terms" if len(hit) < MIN_TERMS else "generic_only"
                dropped.append({"id": r.get("id"), "reason": "gate_failed:" + reason, "pcover": round(pcover, 3)})
            continue
        scored.append({"r": r, "bm": bm, "cover": cover, "pcover": pcover, "hit": hit})
    top = max((s["bm"] for s in scored), default=1.0) or 1.0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for s in scored:
        r = s["r"]
        when = _parse_day(r.get("updated_at") or r.get("valid_from") or r.get("created_at") or r.get("last_seen"))
        age = max(0.0, (now - when).total_seconds() / 86400) if when else 365.0
        conf = extraction_confidence(r)
        conf = 0.6 if conf is None else conf
        boost = LESSON_BOOST if r.get("kind") == "lesson" and r.get("result") in ("failed", "unnecessary", "mixed") else 1.0
        s["score"] = boost * conf * (s["bm"] / top) * 0.5 ** (age / HALF_LIFE_DAYS) * (1 + math.log(1 + int(r.get("used_n") or 0)))
        s["user"] = r.get("origin_type") == "user_explicit" or (not r.get("origin_type") and r.get("verified_by") == "user_quote")
    scored.sort(key=lambda s: (not s["user"], -s["score"]))
    return scored


TAIL_BYTES, CTX_CHARS, SHORT_PROMPT = 32768, 300, 70


def prev_assistant_tail(transcript, limit=CTX_CHARS):
    """End of the last assistant text turn, read from the last 32 KB of the transcript only. '' when unavailable. Never raises."""
    try:
        with open(transcript, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read().decode("utf-8", "replace")
        lines = data.splitlines()[1 if size > TAIL_BYTES else 0:]  # the first line may be cut
        for line in reversed(lines):
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            m = msg.get("message") if isinstance(msg.get("message"), dict) else msg
            if m.get("role") != "assistant":
                continue
            c = m.get("content")
            text = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text") if isinstance(c, list) else ""
            text = " ".join(text.split())
            if text:
                return text[-limit:]
    except Exception:
        pass
    return ""


def query_context(prompt, transcript):
    """A short prompt ('devam et', 'evet') means nothing alone: append the end of the previous assistant turn. Long prompts stand alone."""
    if transcript and len(prompt.strip()) < SHORT_PROMPT:
        tail = prev_assistant_tail(transcript)
        if tail:
            return prompt.strip() + "\n" + tail
    return prompt


def _clip(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _day(r):
    return str(r.get("updated_at") or r.get("valid_from") or r.get("created_at") or r.get("last_seen") or "")[:10]


def _body(r):
    """The record's own content in one string: the lesson's approach → result: why, or the decision/rule text."""
    if r.get("kind") == "lesson":
        return f"{r.get('approach', '')} → {r.get('result', '')}: {r.get('why', '')}" + (f" — {r['reuse']}" if r.get("reuse") else "")
    return r.get("text") or r.get("title") or ""


def _line(s, limit):
    r = s["r"]
    kind = r.get("kind")
    body = _body(r)
    sess = r.get("session") or ""
    tail = f" (bkz. session {sess})" if sess else ""
    head = f"[other project: {r.get('project', '?')} · {_day(r)} · {'you' if s['user'] else 'inferred'}] {kind}: "
    from memrove_v3_inject import reassess_tag
    head += reassess_tag(r).lstrip() + " " if reassess_tag(r) else ""
    return head + _clip(body, max(20, limit - len(head) - len(tail))) + tail


def _card(s):
    """Pointer, not content: the model decides whether to open it with `memrove show <id>`."""
    r = s["r"]
    title = re.sub(r"^\[[^\]]*\]\s*", "", r.get("title") or r.get("text") or r.get("approach") or "")
    from memrove_v3_inject import reassess_tag  # E17: a weakened assumption stays visible on the card
    return f"[{r.get('id')}] {r.get('kind')}: {_clip(title, 80)}{reassess_tag(r)} ({r.get('project', '?')} · {_day(r)} · {'you' if s['user'] else 'inferred'})"


def _content_card(s, limit=230):
    """Content, not a pointer: the decision / rule / lesson itself in one or two lines, with its id for `memrove show`."""
    r = s["r"]
    from memrove_v3_inject import reassess_tag
    body = re.sub(r"^\[[^\]]*\]\s*", "", _body(r))
    head = f"[{r.get('id')}] {r.get('kind')}{reassess_tag(r)} ({r.get('project', '?')} · {_day(r)} · {'you' if s['user'] else 'inferred'}): "
    return head + _clip(body, max(40, limit))


def _thread_index(store):
    try:
        return json.loads((Path(store.state_dir) / "threads-index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _thread_card(tid, t):
    return f"[{tid}] thread: {_clip(t.get('title', ''), 70)} [{t.get('status', '')}] — {_clip(t.get('context', ''), 90)} ({t.get('project', '?')} · {t.get('day', '')} · inferred)"


def _prefer_threads(store, scored, done, card=None):
    """Context before atoms (E25): a record that a thread links as evidence is shown as its thread card; one card per thread. -> [(block, card)]"""
    idx, entries, used = _thread_index(store), [], set()
    for s in scored:
        tid = (idx.get("records") or {}).get(s["r"]["id"])
        t = (idx.get("threads") or {}).get(tid)
        if t and t.get("status") != "closed":
            if tid not in used and "recall:" + tid not in done and tid not in done:
                used.add(tid)
                entries.append(("recall:" + tid, _thread_card(tid, t), ""))
        else:
            entries.append(("recall:" + s["r"]["id"], (card or _card)(s), ""))
    return entries


def _entity_line(hit, limit):
    entity, alias, head = hit
    return _clip(f"[profil: {entity} · sen] {alias}: {head}", limit)


def _entity_hits(store, prompt):
    try:
        from memrove_v3_entities import match  # E11; absent -> no alias candidates
    except ImportError:
        return []
    return list(match(store, prompt) or [])


def _decide(prompt, session, project, cands, passed, dropped, stats=None):
    """Same recall as one decision line for the local decision model's eval sets: every candidate with its gate result."""
    try:
        import memrove_v3_decide_log as dl
        opts = [{"id": c["id"], "score": c.get("score"), "pcover": c.get("pcover"), "gate": "alias" if "alias" in c else ("model" if "s1" in c else "pass"), "injected": c["included"],
                 **{k: c[k] for k in ("s1", "s2", "cos", "bm") if k in c}, **({"why_not": c["reason"]} if not c["included"] else {})} for c in cands]
        opts += [{"id": i, "gate": "context_unit", "injected": True} for i in passed if i not in {c["id"] for c in cands}]  # thread/card swapped in for a record
        opts += [{"id": d.get("id"), "pcover": d.get("pcover"), "gate": d["reason"], "injected": False} for d in dropped[:100]]
        dl.log_recall(prompt, session, project, opts, passed, extra={k: v for k, v in (stats or {}).items() if k in ("abstain", "mode", "embedder", "stage2", "score_key", "dense", "ms2")})
    except Exception:
        pass


def _log(store, prompt, cands, passed, started, excluded=None, counts=None):
    try:
        with (store.state_dir / "recall-log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
                                "candidates": cands, "passed": passed, "excluded": excluded or [], "counts": counts or {}, "ms": round((time.monotonic() - started) * 1000, 1)}) + "\n")
    except OSError:
        pass


def _epoch_done(store, session):
    """(epoch, block ids already injected this session). Read-only stores see nothing and write nothing."""
    with store._connect() as db:
        if store.read_only:
            return 0, set()
        db.execute("CREATE TABLE IF NOT EXISTS inject_epoch(session TEXT PRIMARY KEY, epoch INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS injected(session TEXT, epoch INTEGER, block_id TEXT, PRIMARY KEY(session,epoch,block_id))")
        row = db.execute("SELECT epoch FROM inject_epoch WHERE session=?", (session,)).fetchone()
        epoch = row[0] if row else 0
        return epoch, {r[0] for r in db.execute("SELECT block_id FROM injected WHERE session=? AND epoch=?", (session, epoch))}


class StateStore:
    """Just enough store for the hook (state_dir, _connect, read_only); importing the full MemoryStore costs ~25 ms."""
    read_only = False

    def __init__(self, state):
        self.state_dir = Path(state)
        self.database = self.state_dir / "memory.sqlite3"

    @contextlib.contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=1)
        try:
            with db:
                yield db
        finally:
            db.close()


def recall(store, prompt, cwd, session, transcript=None):
    if os.environ.get("MEMROVE_NO_RECALL") == "1" or not prompt or not prompt.strip():
        return ""
    started = time.monotonic()
    from memrove_v3_project import sticky
    here = sticky(store.state_dir, session, cwd, transcript=transcript)[0] or "unknown"
    prompt_terms = terms(prompt)
    shown = {}  # stem -> first original prompt word, for the "memrove search" hint
    for w in re.findall(r"[^\W_]{3,}", prompt):
        shown.setdefault(terms(w)[0] if terms(w) else "", w)
    epoch, done = _epoch_done(store, session)
    ents = [h for h in _entity_hits(store, prompt) if "recall:ent:" + h[0] not in done]
    scored, dropped, stats, res = [], [], {}, None
    from memrove_v3_hybrid import config
    cfg = config(store.state_dir)
    hybrid = cfg["mode"] != "gate"
    ctx = query_context(prompt, transcript) if hybrid else prompt
    stats["ctx"] = ctx != prompt
    if hybrid and len(set(terms(ctx))) >= 2:
        from memrove_v3_hybrid import select
        res = select(store, ctx, here, done, cfg, stats, dropped)
        scored = res["scored"]
    elif hybrid:
        stats["short_prompt"] = 1
    elif len(set(prompt_terms)) >= MIN_WORDS:
        ranked = _rank(_load(store, here, stats), prompt_terms, gate=True, dropped=dropped)
        scored = [s for s in ranked if "recall:" + s["r"]["id"] not in done]
        dropped += [{"id": s["r"]["id"], "reason": "already_injected"} for s in ranked if "recall:" + s["r"]["id"] in done]
    else:
        stats["short_prompt"] = 1
    lines, blocks, words = [], [], []
    room = MAX_TOKENS - tok(ENVELOPE.format("")) - 20  # 20 tokens reserved for the header line
    entries = [("recall:ent:" + h[0], _entity_line(h, 240), h[1]) for h in ents[:MAX_ITEMS]] + \
              _prefer_threads(store, scored, done, _content_card if hybrid else None)
    for block, line, why in entries[:MAX_ITEMS]:
        if tok(line) > room:
            line = line[:max(0, room * 4)]
        if not line or tok(line) > room:
            break
        room -= tok(line) + 1
        lines.append(line)
        blocks.append(block)
        words.append(why)
    passed = [b[7:] for b in blocks]
    cands = [{"id": "ent:" + h[0], "alias": h[1], "included": "ent:" + h[0] in passed, "reason": "alias" if "ent:" + h[0] in passed else "budget"} for h in ents]
    if res:  # hybrid: every stage-1 candidate with its scores; the abstain flag says "nothing cleared the threshold"
        cands += [dict(c, included=c["id"] in passed, reason="included" if c["id"] in passed else ("budget" if c["reason"] == "selected" else c["reason"])) for c in res["log"]]
        stats.update({k: v for k, v in res["info"].items() if k != "mode"})
    else:
        cands += [{"id": s["r"]["id"], "score": round(s["score"], 4), "cover": round(s["cover"], 3), "pcover": round(s["pcover"], 3), "included": s["r"]["id"] in passed,
                   "reason": "included" if s["r"]["id"] in passed else "budget"} for s in scored[:10]]
    abstain = not passed
    stats["abstain"], stats["mode"] = abstain, "hybrid" if hybrid else "gate"
    _log(store, prompt, cands, passed, started, dropped[:50], stats)
    _decide(prompt, session, here, cands, passed, dropped, stats)
    if not lines:
        return ""
    if not store.read_only:
        with store._connect() as db:
            db.executemany("INSERT OR IGNORE INTO injected VALUES(?,?,?)", [(session, epoch, b) for b in blocks])
    header = "Related memories (summary text; `memrove show <id>` for detail):" if hybrid else "Possibly related memories; open with `memrove show <id>` if needed:"
    return ENVELOPE.format("\n".join([header] + lines))


def search(store, query, limit=10):
    """Same ranking, no gate, includes the current project. -> list of result dicts."""
    out = []
    for s in _rank(_load(store), terms(query), gate=False)[:limit]:
        r = s["r"]
        out.append({"id": r["id"], "kind": r.get("kind"), "project": r.get("project"), "updated_at": r.get("updated_at"),
                    "origin": "you" if s["user"] else "inferred", "score": round(s["score"], 4), "line": _line(s, 300)})
    return out


def show(store, rid):
    """Full record, compact, for `memrove show <id>` (unique id prefix accepted)."""
    if rid.startswith(("th:", "t-", "ls:")):  # context units: the thread / card document in the companion files
        import memrove_v3_threads as th
        idx = _thread_index(store)
        key = rid[3:] if rid.startswith("th:") else rid
        key = (idx.get("alias") or {}).get(key, key)  # a thread merged into another keeps resolving
        doc = th.card_doc(store.vault_root, rid) if rid.startswith("ls:") else th.thread_doc(store.vault_root, key)
        t = (idx.get("threads") or {}).get(key)
        if doc:
            return doc
        if t:  # the block is gone from Threads.md (archived): the index still has its summary
            return {"id": rid, "kind": "thread", **{k: v for k, v in t.items() if v}}
        if rid.startswith(("th:", "ls:")):
            raise ValueError(f"no thread or card for {rid!r}")
    with store._connect() as db:
        rows = db.execute("SELECT payload FROM records WHERE id=? OR id LIKE ? LIMIT 2", (rid, rid + "%")).fetchall()
    if len(rows) != 1:
        raise ValueError(f"no unique record for {rid!r}")
    r = json.loads(rows[0][0])
    keys = ("id", "kind", "project", "title", "text", "approach", "result", "situation", "why", "reuse", "evidence", "quote", "verified_by",
            "origin_type", "session", "updated_at", "valid_from", "created_at", "status")
    return {k: r[k] for k in keys if r.get(k) not in (None, "", [], {})}
