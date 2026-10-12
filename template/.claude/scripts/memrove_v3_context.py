"""observe (write API) and context --task (read API with --explain). Both sit on the existing contracts:
observe -> sync.note_create (validate -> intent -> commit -> readback); context -> scope/status gates before ranking."""
import hashlib
import json
from datetime import datetime, timezone

import memrove_v3_kinds as kinds
from memrove_v3_trust import corroboration, lesson_admission
from memrove_v3_recall import fold, terms

SECTIONS = (("user", "USER RULES", 25), ("project", "PROJECT", 25), ("decisions", "DECISIONS", 20),
            ("lessons", "LESSONS", 15), ("entities", "ENTITIES", 10), ("related", "RELATED (other projects)", 5))
DEAD = ("retired", "rejected", "dropped_llm", "deleted_by_user", "source_missing")


def observe(sync, kind, source, payload, scope=None, core=False):
    """-> {id, revision, status: created|exists}. Idempotent on (source, kind, source_item_id)."""
    spec = kinds.spec(kind)
    if not (spec.get("observable") or core):
        raise ValueError(f"kind {kind!r} is not writable through observe")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("provenance required: --source (source id)")
    if payload.get("origin_type") not in kinds.ORIGINS:
        raise ValueError(f"provenance required: origin_type must be one of {list(kinds.ORIGINS)}")
    item = payload.get("source_item_id")
    if not isinstance(item, str) or not item.strip():
        raise ValueError("provenance required: source_item_id")
    for field in spec.get("required", []):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{kind} requires {field}")
    scope = scope or payload.get("scope")
    if scope is not None and not kinds.valid_scope(scope):
        raise ValueError("invalid scope; use project:<name>, user, technical, learning or personal")
    digest = hashlib.sha256(f"{source}\0{kind}\0{item}".encode()).hexdigest()[:24]
    rid = "obs-" + digest
    with sync.store._connect() as db:
        row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
    if row:
        return {"id": rid, "revision": json.loads(row[0])["revision"], "status": "exists"}
    meta = {k: v for k, v in payload.items() if k not in ("text", "scope", "id", "revision", "kind")}
    meta.update(id=rid, kind=kind, source_system=source, recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if scope:
        meta["scope"] = scope
    if spec.get("initial"):
        meta.setdefault("status", spec["initial"])
    sync.note_create(f"notes/observed/{kind}/{digest}.md", payload.get("text") or payload.get("title"), meta)
    with sync.store._connect() as db:
        row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
    return {"id": rid, "revision": json.loads(row[0])["revision"], "status": "created"}


def _section(r, project):
    k = r.get("kind")
    if k == "preference":
        return "user"
    if k == "entity":
        return "entities"
    if project and r.get("project") and r["project"] != project:
        return "related"
    return "decisions" if k in ("decision", "rejected") else "lessons" if k == "lesson" else "project"


def _line(r, scope, n=220):
    text = " ".join(str(r.get("quote") or r.get("text") or r.get("title") or "").split())
    return f"#{r['id']} {r.get('kind')} [{scope}]: " + (text if len(text) <= n else text[:n - 1] + "…")


def context_pack(store, task, project=None, scope="all", budget=800, client="claude_code"):
    """Task-based pack. Every record is classified once; explain lists all of them. budget = tokens (len//4)."""
    allowed = kinds.client_scopes(store.state_dir, client)
    want = None if scope in (None, "all") else [s.strip() for s in scope.split(",")]
    q = set(terms(task))
    with store._connect() as db:
        records = [json.loads(p) for (p,) in db.execute("SELECT payload FROM records ORDER BY id")]
    gone = {v for r in records for v in (r.get("supersedes") or [])}
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    explain, cands = [], []
    for r in records:
        sc = kinds.effective_scope(r)
        e = {"id": r["id"], "kind": r.get("kind"), "scope": sc, "included": False}
        explain.append(e)
        if not kinds.in_scope(sc, allowed) or (want and not kinds.in_scope(sc, want)):
            e["reason"] = "out_of_scope"
        elif r["id"] in gone or r.get("status") == "superseded":
            e["reason"] = "superseded"
        elif r.get("status") in DEAD or r.get("verdict") == "reject" or (r.get("valid_to") and str(r["valid_to"]) <= now) or r.get("visibility") == "private":
            e["reason"] = "retired"
        elif r.get("kind") == "lesson" and project and r.get("project") != project and lesson_admission(r, corroboration([r])) != "admit":
            e["reason"] = "lesson_not_admitted"
        else:
            e["section"] = _section(r, project)
            hit = q & set(terms(json.dumps(r, ensure_ascii=False)))
            e["overlap"] = sorted(hit)
            # gate: user rules always apply; everything else needs a shared term with the task
            if e["section"] != "user" and not hit:
                e["reason"] = "gate_failed:no_term_overlap"
            else:
                cands.append((r, e, sc))
    cands.sort(key=lambda c: str(c[0].get("updated_at", "")), reverse=True)
    cands.sort(key=lambda c: (-int(c[0].get("origin_type") == "user_explicit"), -len(c[1]["overlap"])))
    sections, seen, text = {}, set(), []
    for name, head, pct in SECTIONS:
        room, lines = budget * pct // 100 - len(head) // 4 - 1, []
        for r, e, sc in cands:
            if e["section"] != name:
                continue
            norm = fold(" ".join(str(r.get("text") or r.get("title") or "").split()))
            if norm in seen:
                e["reason"] = "duplicate"
                continue
            line = _line(r, sc)
            if len(line) // 4 + 1 > room:
                e["reason"] = "budget"
                continue
            room -= len(line) // 4 + 1
            seen.add(norm)
            lines.append(line)
            e.update(included=True, reason="included")
        if lines:
            sections[name] = lines
            text.append(head + ":\n" + "\n".join(lines))
    body = "\n".join(text)
    return {"task": task, "scope": scope, "client_scopes": allowed, "budget": budget, "tokens": len(body) // 4,
            "sections": sections, "text": body, "explain": explain}


REMEMBER_KINDS = ("decision", "rejected", "lesson", "preference", "note")


def remember(sync, kind, text, why=None, project=None, user=False, session=None, source="memrove-cli", supersedes=()):
    """Model-written memory: same observe contract (validate, commit, readback), core kinds allowed. Idempotent on (kind, text)."""
    from memrove_v3_capture import has_secret, key_of
    if kind not in REMEMBER_KINDS:
        raise ValueError(f"kind must be one of {list(REMEMBER_KINDS)}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("remember requires text")
    if has_secret(text) or (why and has_secret(why)):
        raise ValueError("text looks like it contains a secret; not saved")
    text = " ".join(text.split())
    payload = {"text": text, "origin_type": "user_explicit" if user else "agent_inference",
               "source_item_id": hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()[:24],
               "status": "active", "confidence": 1.0 if user else 0.7,
               "valid_from": datetime.now(timezone.utc).date().isoformat(), "key": key_of(text)}
    if user:
        payload["verified_by"] = "user_quote"
    if why:
        payload["why"] = " ".join(why.split())
    if project:
        payload["project"] = project
    if session:
        payload["session"] = session
    if kind == "rejected":
        payload["polarity"] = "neg"
    if supersedes:  # the record(s) this one replaces; each must exist (unique id or prefix)
        with sync.store._connect() as db:
            found = []
            for old in supersedes:
                old = old.removeprefix("rec:")
                rows = db.execute("SELECT id FROM records WHERE id=? OR id LIKE ? LIMIT 2", (old, old + "%")).fetchall()
                if len(rows) != 1 and not any(r[0] == old for r in rows):
                    raise ValueError(f"no unique record to supersede for {old!r}")
                found.append(old if any(r[0] == old for r in rows) else rows[0][0])
        payload["supersedes"] = found
    result = observe(sync, kind, source, payload, core=kind != "note")
    try:
        if result.get("status") != "exists":
            import memrove_v3_decide_log as dl
            dl.kind_corrected(text, result.get("id"), kind, "user" if user else "remember")
            dl.log_capture_kind(text, kind, result.get("id"), "remember", source="human" if user else "model", project=project, session=session)
    except Exception:
        pass
    return result
