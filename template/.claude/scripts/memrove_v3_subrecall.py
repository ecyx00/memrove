#!/usr/bin/env python3
"""Task-specific memory for a subagent: records and owner messages that match the agent's brief, ranked by relevance only (never age).

Entry point for the SubagentStart hook (memrove_v3_jit) and for evals: `echo "<brief>" | python3 memrove_v3_subrecall.py --state STATE [--project P] [--json]`
or `recall(state, brief, project)` / `block(state, brief, project)`. Stdlib only; read-only on the store except the brief stash.
"""
import fcntl
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memrove_v3_chatsearch as cs
from memrove_v3_recall import STOP

MAX_ITEMS, QUOTAS, LINE, BUDGET_CHARS = 6, (3, 2, 1), 330, 2400   # ~600 tokens at 4 chars/token
QUERY_WORDS, MIN_HITS = 24, 3
MIN_COS = 0.6   # dense noise floor: dev-set hits scored 0.65-0.74 and the top misses 0.70-0.76, so cosine cannot separate them; it only drops junk
NEG = re.compile(r"\b(?:not|never|no|don'?t|doesn'?t|instead|rather than|stop|without|avoid|rejected|wrong)\b|\bdeğil\b|\byok\b", re.I)
HEAD = "Memory for this task (owner's words first; id, date; `memrove show <id>` opens one):\n"
STASH_TTL = 600
MAX_MSG = 1000   # shortcut: the owner's chat messages are short; longer "user" messages are pasted briefs or logs. Upgrade when the chat index tags the author.
DELEGATION = re.compile(r"\s*(?:spawn|launch|execute only|use the agent tool|do these \d|you are (?:the|an?) )", re.I)


def query_of(brief):
    words = re.findall(r"\w+", cs.fold(brief))
    freq = {}
    for w in words:
        if len(w) >= 4 and w not in STOP and not w.isdigit():
            freq[w] = freq.get(w, 0) + 1
    return " ".join(sorted(freq, key=lambda w: (-freq[w], -len(w)))[:QUERY_WORDS])


def _stems(text):
    return {w[:5] for w in re.findall(r"\w+", cs.fold(text)) if len(w) >= 4 and w not in STOP}


def _meta(state, ids):
    """{record id: (project, origin_type, scope)} for the candidate records."""
    out = {}
    if not ids:
        return out
    db = sqlite3.connect(f"file:{Path(state) / 'memory.sqlite3'}?mode=ro", uri=True, timeout=2)
    try:
        for rid, p, o, s in db.execute(f"SELECT id, json_extract(payload,'$.project'), json_extract(payload,'$.origin_type'), json_extract(payload,'$.scope') "
                                       f"FROM records WHERE id IN ({','.join('?' * len(ids))})", list(ids)):
            out[rid] = (p, o, s)
    finally:
        db.close()
    return out


def _messages(state, query, top=40):
    """-> [(bm25, (id, date, 'user', text))] from the chat search index, opened read-only without syncing: a hook must not wait on another session's index write."""
    q = cs.terms(query)
    path = cs.chatlog.root(state) / "search.db"
    if q is None or not path.exists():
        return []
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
    try:
        score = {}
        for expr, weight in ((q[0], 1.0), (q[1], cs.PREFIX)):
            for i, b in db.execute("SELECT m.id, bm25(fts) FROM fts JOIN m ON m.id = fts.rowid WHERE fts MATCH ? AND m.kind = 'user' ORDER BY bm25(fts) LIMIT ?", (expr, top)):
                score[i] = score.get(i, 0.0) + weight * b
        keep = sorted(score, key=score.get)[:top]
        rows = {r[0]: r for r in db.execute(f"SELECT id, date, kind, text FROM m WHERE id IN ({','.join('?' * len(keep))})", keep)}
        return [(score[i], rows[i]) for i in keep if i in rows]
    finally:
        db.close()


def _dense(state, brief, top=40):
    """-> [(-cosine, (id, date, kind, text))]: owner messages ("msg:<id>") and live records by EmbeddingGemma cosine to the brief, best first. The vectors are built at sync time
    (embed.refresh), so an English brief reaches Turkish messages by meaning. Raises OSError when the model daemon does not answer: the caller falls back to bm25."""
    import memrove_v3_embed as emb
    qv = emb.DaemonEmbedder(timeout=1.0).embed_query([emb.clip_query(brief, 1500)])[0]
    vecs = emb.VecStore(state, readonly=True).load(emb.DaemonEmbedder.name)
    if not any(k.startswith("msg:") for k in vecs):
        raise OSError("no message vectors yet")
    live = {rid: (r, day) for rid, r, day in cs._live(state)}
    sims = sorted(((sum(map(float.__mul__, qv, v)), k) for k, v in vecs.items() if k.startswith("msg:") or k in live), reverse=True)[:top]
    msg = [int(k[4:]) for _, k in sims if k.startswith("msg:")]
    fused = {}   # idea cards (English card per owner message): reciprocal-rank fusion with the message cosine, when that module and its store exist
    try:
        import memrove_v3_ideacards as ic
        cards = [int(i) for i, _ in ic.search_cards(state, brief, 40)]
        order = list(msg)
        for lst in (order, cards):
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0.0) + 1 / (60 + r + 1)
        msg = sorted(set(msg) | set(cards))
    except Exception:
        fused = {}
    db = sqlite3.connect(f"file:{cs.chatlog.root(state) / 'search.db'}?mode=ro", uri=True, timeout=1)
    try:
        rows = {r[0]: r for r in db.execute(f"SELECT id, date, kind, text FROM m WHERE id IN ({','.join('?' * len(msg))})", msg)}
    finally:
        db.close()
    out = []
    for c, k in sims:
        if k.startswith("msg:") and int(k[4:]) in rows:
            if c >= MIN_COS:
                out.append((-fused.get(int(k[4:]), c), rows[int(k[4:])]))
        elif k in live and c >= MIN_COS:
            r, day = live[k]
            out.append((-c, (f"rec:{k}", day, r.get("kind"), " ".join(str(r.get(f) or "") for f in ("title", "text", "approach")))))
    for i in set(fused) - {int(k[4:]) for _, k in sims if k.startswith("msg:")}:   # card-only hits: no dense score, valid by the card match
        if i in rows:
            out.append((-fused[i], rows[i]))
    return sorted(out, key=lambda x: x[0])


def candidates(state, brief, project=None):
    """-> [dict(id, date, kind, text, user, score)] best first (lower score is better); records of other projects dropped (owner messages carry no project).
    Meaning-level (dense) when the model daemon answers; else the old bm25 over words, which needs a shared vocabulary."""
    try:
        scored, dense = _dense(state, brief), True
    except Exception:
        q = query_of(brief)
        if not q:
            return []
        scored, dense = [(cs.REC_WEIGHT * sc, row) for sc, row in cs.records(state, q, 30)] + _messages(state, q), False
    q = query_of(brief)
    want = _stems(q)
    need = min(MIN_HITS, max(1, len(want) // 2))
    meta = _meta(state, [row[0][4:] for _, row in scored if str(row[0]).startswith("rec:")])
    out = []
    for sc, (rid, day, kind, text) in scored:
        isrec = str(rid).startswith("rec:")
        if isrec:
            proj, origin, scope = meta.get(rid[4:], (None, None, None))
            if proj not in (None, "", "unknown", project) and scope != "global":
                continue
        elif text.lstrip().startswith("<") or "[Subagent hand-back]" in text[:300] or len(text) > MAX_MSG or DELEGATION.match(text):
            continue   # hand-backs, notifications, pasted logs/briefs and harness-test prompts arrive as "user" messages but are not the owner's words
        else:
            origin = "user_explicit"
        if not dense and len(want & _stems(text)) < need:
            continue
        out.append({"id": rid if isrec else f"msg:{rid}", "date": str(day or "")[:10], "kind": kind if isrec else "message", "text": " ".join(str(text).split()),
                    "user": origin == "user_explicit", "score": sc})
    return sorted(out, key=lambda c: c["score"])


def _group(c):
    return 0 if c["user"] else 1 if c["kind"] in ("decision", "rejected") else 2


def pick(cands):
    """Quota by group (owner words 3, decisions 2, lessons/notes 1), then the best leftovers up to MAX_ITEMS; shown owner-first, then decisions, then lessons."""
    chosen, seen = [], set()
    for g, quota in enumerate(QUOTAS):
        for c in [c for c in cands if _group(c) == g][:quota]:
            chosen.append(c)
            seen.add(c["id"])
    chosen += [c for c in cands if c["id"] not in seen][:MAX_ITEMS - len(chosen)]
    return sorted(chosen, key=lambda c: (_group(c), c["score"]))


def _conflicts(items):
    """{id: [other id date, ...]}: two picked items on one topic (>=35% shared content stems) where one negates and the other does not. Both stay, dated."""
    out = {}
    st = [_stems(c["text"]) for c in items]
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            both = st[i] & st[j]
            if both and len(both) / min(len(st[i]), len(st[j])) >= 0.35 and bool(NEG.search(a["text"])) != bool(NEG.search(b["text"])) and a["date"] != b["date"]:
                out.setdefault(a["id"], []).append(f"{b['id']} {b['date']}")
                out.setdefault(b["id"], []).append(f"{a['id']} {a['date']}")
    return out


def _line(c, q, conflict):
    tag = {"message": "you said", "decision": "decision", "rejected": "rejected", "lesson": "lesson"}.get(c["kind"], c["kind"])
    tag = f"you, {tag}" if c["user"] and c["kind"] != "message" else tag
    head = f"- [{c['id']} {c['date']} {tag}] "
    tail = f" [conflicts with {'; '.join(conflict)}]" if conflict else ""
    body = cs.redact(cs.snippet(c["text"], q, max(40, LINE - len(head) - len(tail))))
    return head + body + tail


def recall(state, brief, project=None):
    """-> the picked items (dicts), [] when nothing matches."""
    return pick(candidates(state, brief, project))


def from_picks(state, picks):
    """The Haiku selector's item ids ("m<id>", "rec:<id>") as item dicts in the selector's order, owner words first, then decisions, then lessons."""
    msgs = [int(i[1:]) for i in picks if re.fullmatch(r"m\d+", i)]
    rows = {}
    if msgs:
        db = sqlite3.connect(f"file:{cs.chatlog.root(state) / 'search.db'}?mode=ro", uri=True, timeout=1)
        try:
            rows = {r[0]: r for r in db.execute(f"SELECT id, date, kind, text FROM m WHERE id IN ({','.join('?' * len(msgs))})", msgs)}
        finally:
            db.close()
    live = {f"rec:{rid}": (r, day) for rid, r, day in cs._live(state)}
    out = []
    for n, i in enumerate(picks):
        if i in live:
            r, day = live[i]
            out.append({"id": i, "date": str(day)[:10], "kind": r.get("kind"), "user": False, "score": n,
                        "text": " ".join(" ".join(str(r.get(f) or "") for f in ("title", "text", "approach")).split())})
        elif re.fullmatch(r"m\d+", i) and int(i[1:]) in rows:
            _, day, _, text = rows[int(i[1:])]
            out.append({"id": f"msg:{i[1:]}", "date": str(day)[:10], "kind": "message", "text": " ".join(text.split()), "user": True, "score": n})
    return sorted(out, key=_group)[:MAX_ITEMS]


def block(state, brief, project=None, picks=None, ids=None):
    """The injected text ('' when nothing relevant), within BUDGET_CHARS. picks: the Haiku selector's ids; they replace the dense/bm25 result."""
    items = from_picks(state, picks) if picks else recall(state, brief, project)
    q = query_of(brief)
    while items:
        conf = _conflicts(items)
        text = HEAD + "\n".join(_line(c, q, conf.get(c["id"])) for c in items)
        if len(text) <= BUDGET_CHARS:
            if ids is not None:  # the caller's exposure log
                ids.extend(c["id"] for c in items)
            return text
        items = items[:-1]
    return ""


# --- brief capture -------------------------------------------------------------------------------------------------

def _locked(path):
    """Exclusive lock for a read-modify-write of `path` (parallel Agent spawns in one message race on it)."""
    f = open(str(path) + ".lock", "a")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f   # closing the file releases the lock


def stash(state, payload):
    """PreToolUse(Agent): remember the brief for the SubagentStart that follows. Never raises."""
    try:
        ti = payload.get("tool_input") or {}
        text = ti.get("prompt")
        if not isinstance(text, str) or not text.strip():
            return
        path = Path(state) / "agent-briefs.jsonl"
        lock = _locked(path)
        keep = [l for l in (path.read_text(encoding="utf-8").splitlines() if path.exists() else []) if time.time() - json.loads(l)["t"] < STASH_TTL][-20:]
        keep.append(json.dumps({"t": time.time(), "session": payload.get("session_id"), "type": ti.get("subagent_type") or "general-purpose", "prompt": text[:6000]}, ensure_ascii=False))
        path.write_text("\n".join(keep) + "\n", encoding="utf-8")
        lock.close()
        import memrove_v3_selector
        memrove_v3_selector.start(state, text, payload.get("project"))
    except Exception:
        pass


def _from_stash(state, payload):
    path = Path(state) / "agent-briefs.jsonl"
    with _locked(path):
        return _take(path, payload)


def _take(path, payload):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kind = str(payload.get("agent_type") or "").rsplit(":", 1)[-1]
    for r in rows:   # oldest unconsumed first: parallel spawns of one type start in spawn order
        if r.get("session") == payload.get("session_id") and r.get("type", "").rsplit(":", 1)[-1] == kind and time.time() - r["t"] < STASH_TTL:
            rows.remove(r)
            path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
            return r["prompt"]
    return ""


def _from_transcript(payload):
    p = payload.get("agent_transcript_path")
    if not p or not Path(p).exists():
        return ""
    for line in Path(p).read_text(encoding="utf-8").splitlines()[:5]:
        m = (json.loads(line).get("message") or {})
        c = m.get("content")
        if m.get("role") == "user" and c:
            return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def brief_of(state, payload):
    """The agent's brief: a prompt field, else the transcript's first user message, else the stash. '' when none."""
    if isinstance(payload.get("prompt"), str) and payload["prompt"].strip():
        return payload["prompt"]
    for get in (lambda: _from_transcript(payload), lambda: _from_stash(state, payload)):
        try:
            t = get()
            if t.strip():
                return t
        except Exception:
            pass
    return ""


def main():
    a = sys.argv[1:]
    arg = lambda k: a[a.index(k) + 1] if k in a[:-1] else None
    state, project, brief = arg("--state"), arg("--project"), sys.stdin.read()
    if "--json" in a:
        print(json.dumps(recall(state, brief, project), ensure_ascii=False, indent=1))
    else:
        print(block(state, brief, project))


if __name__ == "__main__":
    main()
