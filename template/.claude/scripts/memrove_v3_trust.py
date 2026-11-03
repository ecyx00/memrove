"""Trust model (design section 15): split confidence, assessment state, egress class, honest corroboration.
Pure functions over record dicts. Old fields (confidence, origin_type, verified_by, llm_verdict) stay readable: nothing here
requires a new field to exist, and stamp() only adds what is missing."""

EGRESS = ("device_only", "self_hosted", "paid_ok", "external_ok")  # most restrictive first; paid_ok: a paid remote model (Haiku) yes, a free tier no
TIERS = {"local": (), "paid": EGRESS[:2], "free": EGRESS[:3]}  # reader tier -> egress classes it never sees (local: this machine and the owner's own harnesses)
OWNER_ENV = ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CODEX_THREAD_ID", "CODEX_SANDBOX")  # set in the shells of Claude Code and Codex, the owner's paid harnesses
AGENT_SOURCES = ("memrove-cli", "beyin-cli")  # source_system of `memrove remember` (beyin: its old name)
RELIABILITY = {"user_explicit": 0.95, "external_source": 0.6, "derived": 0.5, "agent_inference": 0.4}
STATES = ("assessed", "not_assessed", "insufficient", "unknown")
FROM_VERIFIED = {"user_quote": "user_explicit", "llm": "agent_inference"}
# Our own generated summaries are projections, never evidence (design 15, items 223-224).
GENERATED_KINDS = ("session", "note")


TRANSCRIPT = "transcript:"


def with_transcript(old, tp):
    """derived_from of record `old` plus the transcript file (absolute path) the record was derived from."""
    have = list((old or {}).get("derived_from") or [])
    return sorted({*have, TRANSCRIPT + str(tp)}) if tp else have


def reconcile_transcripts(engine):
    """Soft source state for records derived from Claude Code transcripts (deleted when Claude Code prunes old sessions).
    Every transcript of a record gone -> source_state=transcript_pruned, nothing else changes: the lesson or inference keeps its own
    verbatim evidence, stays active, injected and recalled; the normal decay clock still retires it when nobody repeats it.
    A transcript that is back clears the state. Docs get the hard source_missing instead (memrove_v3_sync). -> records changed."""
    import json
    import os
    from memrove_v3_projections import update_record
    n = 0
    with engine.store._connect() as db:
        rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.derived_from') IS NOT NULL")]
    for r in rows:
        paths = [s[len(TRANSCRIPT):] for s in r["derived_from"] if isinstance(s, str) and s.startswith(TRANSCRIPT)]
        if not paths or r.get("status") == "deleted_by_user":
            continue
        pruned = not any(os.path.isfile(p) for p in paths)
        if pruned == (r.get("source_state") == "transcript_pruned"):
            continue
        change = {"source_state": "transcript_pruned" if pruned else "ok"}
        with engine.store._connect() as db:
            if db.execute("SELECT 1 FROM markdown_sources WHERE id=?", (r["id"],)).fetchone():
                n += update_record(engine, r["id"], change, db)
                continue
            new = dict(r, **change, revision=r["revision"] + 1)  # store-only record (papercut): no note to re-project
            payload = json.dumps(new, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            db.execute("UPDATE records SET payload=? WHERE id=?", (payload, r["id"]))
            db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES ('update',?,?,?)", (r["id"], new["revision"], payload))
            n += 1
    return n


def origin_of(r):
    return r.get("origin_type") or FROM_VERIFIED.get(r.get("verified_by"))


def extraction_confidence(r):
    v = r.get("extraction_confidence", r.get("confidence"))
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def source_reliability(r):
    v = r.get("source_reliability")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else RELIABILITY.get(origin_of(r))


def assessment_state(r):
    """assessed | not_assessed | insufficient | unknown. Stored assessments win, then the legacy llm_verdict, then the stamp."""
    states = [a.get("state") for a in (r.get("assessments") or {}).values() if isinstance(a, dict)]
    if "assessed" in states:
        return "assessed"
    if "unclear" in states:
        return "insufficient"
    if r.get("llm_verdict") is not None:
        return "assessed"
    if r.get("assessment_state") in STATES:
        return r["assessment_state"]
    return "not_assessed" if r.get("verified_by") in FROM_VERIFIED else "unknown"


def combine_egress(classes):
    """Most restrictive class wins. No input -> device_only (unlabelled data never leaves)."""
    known = [c for c in classes if c in EGRESS]
    return min(known, key=EGRESS.index) if known else "device_only"


def visible_to(egress, tier):
    """May a reader of this tier see a record with this egress label? Fail closed: an unknown or empty label counts as device_only, an unknown tier as free."""
    return (egress if egress in EGRESS else "device_only") not in TIERS.get(tier, TIERS["free"])


def reader_tier(env=None, tty=None):
    """Tier of whoever runs the memrove CLI. MEMROVE_READER (local|paid|free) wins; Claude Code, Codex or an interactive terminal -> local
    (sees all, as before); anything else (opencode, Gemini, a script with piped output) -> free. An unknown MEMROVE_READER value -> free."""
    import os
    import sys
    env = os.environ if env is None else env
    if env.get("MEMROVE_READER"):
        return env["MEMROVE_READER"] if env["MEMROVE_READER"] in TIERS else "free"
    if any(env.get(k) for k in OWNER_ENV):
        return "local"
    return "local" if (sys.stdout.isatty() if tty is None else tty) else "free"


def agent_written(r):
    """Written by a Claude Code agent through `memrove remember` (not --user) or `memrove card`: the text already went to Anthropic in
    that session, so a paid Anthropic model may read it; a free tier may not. Capture, papercut, chat and doc writers never match."""
    return r.get("origin_type") == "agent_inference" and not r.get("verified_by") and (
        r.get("source_system") in AGENT_SOURCES or r.get("origin") == "agent")


def egress_of(r, db=None):
    """Own class (explicit, else Claude Code transcript origin = external_ok, else agent-written = paid_ok, else device_only) combined with the
    classes of every record in derived_from (looked up in `db`, a missing source counts as device_only)."""
    own = r.get("egress")
    if own not in EGRESS:
        own = "external_ok" if r.get("verified_by") in FROM_VERIFIED or r.get("origin") == "llm" else "paid_ok" if agent_written(r) else "device_only"
    classes = [own]
    for sid in r.get("derived_from") or []:
        if sid.startswith(TRANSCRIPT):
            continue  # a Claude Code transcript already went to Anthropic: that is the external_ok the own class carries
        row = db.execute("SELECT payload FROM records WHERE id=?", (sid,)).fetchone() if db is not None else None
        if row:
            import json
            classes.append(egress_of(json.loads(row[0]), db))
        else:
            classes.append("device_only")
    return combine_egress(classes)


def stamp(meta):
    """Add the split trust fields to new-record metadata without overwriting anything the caller set."""
    m = dict(meta)
    conf = extraction_confidence(m)
    if conf is not None:
        m.setdefault("extraction_confidence", conf)
    rel = source_reliability(m)
    if rel is not None:
        m.setdefault("source_reliability", rel)
    m.setdefault("assessment_state", assessment_state(m))
    if m.get("egress") not in EGRESS and (m.get("verified_by") in FROM_VERIFIED or m.get("origin") == "llm"):
        m["egress"] = "external_ok"  # Claude Code transcripts already go to Anthropic
    elif m.get("egress") not in EGRESS and agent_written(m):
        m["egress"] = "paid_ok"
    return m


def session_ids(r):
    """Source sessions a record vouches for. Generated session notes vouch for nothing."""
    if r.get("kind") in GENERATED_KINDS and r.get("origin") == "llm":
        return set()
    ids = r.get("seen_sessions") or ([r["session"]] if r.get("session") else [])
    return {str(i) for i in ids if i}


def corroboration(records):
    """Number of distinct source sessions across records (same claim). Own notes never count; one session counts once."""
    return len(set().union(*(session_ids(r) for r in records))) if records else 0


def lesson_admission(r, sessions=1):
    """-> drop | admit | own_project. assessed no drops; assessed yes, recurrence in >=2 sessions or a user-written lesson admits; else own project only."""
    a = (r.get("assessments") or {}).get("lesson_reusable") or {}
    ans = a.get("answers", {}).get("reusable") if a.get("state") == "assessed" else None
    if ans == "no":
        return "drop"
    return "admit" if ans == "yes" or sessions >= 2 or origin_of(r) == "user_explicit" else "own_project"  # a lesson the user wrote needs no vote
