"""Evidence-first memory (vision point 5): a record is `verified` only when CODE finds its quote verbatim in a chat-log message.
No model judges this (Reflect v2: Haiku agreed with itself 43% of the time, and most records come from tool output that is not in the log).

quote candidates   ctx (the message part of a user quote's ctx), a "double-quoted" span of the text, the record's `quote` (the owner's
                   exact words an agent passed with --user; user-only) and, for a user quote (verified_by=user_quote), its text.
match              whitespace collapsed + casefold, substring of one message, at least MIN characters (shorter strings match everywhere).
                   A user-quote candidate may only match a user message, so an agent echo never verifies an owner quote.
result             verify() -> [{"kind": "msg", "ref": <message k>}] (up to REFS, user messages first), [] = unverified.
stored             record fields evidence_state ("verified"|"unverified") and evidence_refs. Not `evidence` (a j2 lesson text) and not
                   `unverified` / `verified_by` (reflect's mark / the owner's quote): this module never changes who said a record.
Stdlib only; no network, no model. Batch: memrove_v3_evidence.backfill(Core(vault, state), dry=False) -> counts.
"""
import bisect
import json
import re

import memrove_v3_chatlog as chatlog

MIN, REFS = 16, 3
SPAN = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|«([^»\n]+)»')
WHY_SPAN = re.compile(SPAN.pattern + r"|'([^'\n]{16,})'|‘([^’\n]{16,})’")  # agents write why="Owner: '<words>'"
CTX_MSG = re.compile(r"\bmessage: (.*?)(?: \| (?:owner|assistant): |$)", re.S)
SEP = "\0"


def norm(s):
    return " ".join(str(s).split()).casefold()


class Log:
    """The chat log as two searchable strings (all messages, user messages) with message keys by offset."""

    def __init__(self, msgs):
        msgs = list(msgs)
        self.all, self.user = self._index([m for m in msgs if m.get("k")]), self._index([m for m in msgs if m.get("k") and m.get("kind") == "user"])

    @staticmethod
    def _index(ms):
        offs, pos = [], 0
        for m in ms:
            offs.append(pos)
            pos += len(norm(m["text"])) + 1
        return SEP.join(norm(m["text"]) for m in ms), offs, [m["k"] for m in ms]

    @staticmethod
    def find(idx, needle):
        hay, offs, keys = idx
        out, at = [], hay.find(needle)
        while at >= 0 and len(out) < REFS:
            k = keys[bisect.bisect_right(offs, at) - 1]
            if k not in out:
                out.append(k)
            at = hay.find(needle, at + 1)
        return out


def load(state):
    return Log(m for f in chatlog._files(chatlog.root(state)) for m in chatlog.read_file(f))


def quotes(r):
    """-> [(normalized quote, user_only)]"""
    out = []
    own = r.get("verified_by") == "user_quote"
    m = CTX_MSG.search(r.get("ctx") or "")
    if m:
        out.append((m.group(1), True))
    if own:
        out.append((r.get("text", ""), True))
    if r.get("quote"):
        out.append((r["quote"], True))
    out += [(next(g for g in s if g), False) for s in SPAN.findall(r.get("text", ""))]
    return [(norm(q), u) for q, u in out if len(norm(q)) >= MIN]


def owner_quote(state, quote, why):
    """The --user gate: the first candidate (the --quote value, then each quoted span of --why) found verbatim in a user message, else None."""
    cands = [quote or ""] + [next(g for g in m if g) for m in WHY_SPAN.findall(why or "")]
    log = load(state)
    return next((c for c in cands if len(norm(c)) >= MIN and Log.find(log.user, norm(c))), None)


def verify(state, record, log=None):
    """-> evidence list [{"kind": "msg", "ref": k}] for a record whose quote is verbatim in the chat log, else []."""
    qs = quotes(record)
    refs = []
    if log is None and qs:  # one record: scan the messages, norm() only those holding a quote's longest word (indexing 13k messages costs ~70 ms)
        anchors = [(q, u, max(q.split(), key=len)) for q, u in qs]
        for m in (m for f in chatlog._files(chatlog.root(state)) for m in chatlog.read_file(f)):
            low = str(m["text"]).casefold()
            if not m.get("k") or not any(w in low for _, _, w in anchors):
                continue
            n = norm(m["text"])
            for q, user_only, _ in anchors:
                if q in n and (m.get("kind") == "user" or not user_only) and m["k"] not in refs:
                    refs.append(m["k"])
        return [{"kind": "msg", "ref": k} for k in refs[:REFS]]
    for q, user_only in qs:
        for k in Log.find(log.user if user_only else log.all, q):
            if k not in refs:
                refs.append(k)
    return [{"kind": "msg", "ref": k} for k in refs[:REFS]]


def mark(core, rid, log=None):
    """Write the verdict onto one record (write time and backfill). -> the evidence list."""
    with core.store._connect() as db:
        r = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])
    ev = verify(core.state, r, log)
    state = "verified" if ev else "unverified"
    if (r.get("evidence_state"), r.get("evidence_refs")) != (state, ev):
        core._patch(rid, "evidence", evidence_state=state, evidence_refs=ev)
    return ev


def backfill(core, dry=False):
    """Verify every record not yet verified. -> {"total", "verified", "by_kind": {kind: [verified, total]}, "by_origin": {user|agent: [verified, total]}}."""
    log = load(core.state)
    with core.store._connect() as db:
        rows = [(i, json.loads(p)) for i, p in db.execute("SELECT id, payload FROM records")]
    res = {"total": len(rows), "verified": 0, "by_kind": {}, "by_origin": {}}
    for rid, r in rows:
        ev = r.get("evidence_refs") if r.get("evidence_state") == "verified" else verify(core.state, r, log)
        if not dry and ev and r.get("evidence_state") != "verified":
            core._patch(rid, "evidence", evidence_state="verified", evidence_refs=ev)
        elif not dry and not ev and r.get("evidence_state") != "unverified":
            core._patch(rid, "evidence", evidence_state="unverified", evidence_refs=[])
        origin = "user" if r.get("origin_type") == "user_explicit" or r.get("verified_by") == "user_quote" else "agent"
        for bucket, key in ((res["by_kind"], r.get("kind")), (res["by_origin"], origin)):
            c = bucket.setdefault(key, [0, 0])
            c[0] += bool(ev)
            c[1] += 1
        res["verified"] += bool(ev)
    return res

