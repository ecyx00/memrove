"""Write-time provenance: a new record is born with pointers that code has checked (fields as in memrove_v3_msgbridge.verify).
content_msg / authority_msg / trigger_msg are chat-log ids or `transcript:<session>#<uuid>`; content_span is [start, end) verbatim in the content message;
authority_msg only when the cited content is an owner message (verified); trigger_msg = the session's last owner message at write time, a position, never authority.
prov_stage: W (code-verified at write time) | trigger-only | unverified (written, but no content pointer proved)."""
import json
from pathlib import Path

import memrove_v3_msgbridge as bridge
from memrove_v3_capture import user_text


def ptr(sid, uuid):
    return f"transcript:{sid}#{uuid}"


def last_owner_ptr(session, home=None):
    """Code only: the newest owner message of `session` in its Claude transcript -> pointer, or None."""
    best = None
    for p in (Path(home) if home else Path.home()).glob(f".claude/projects/*/{session}.jsonl"):
        for raw in p.read_bytes().splitlines():
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("uuid") and user_text(r):
                best = r["uuid"]
    return ptr(session, best) if best else None


def from_chat(state, session, content_msg, span, home=None):
    """`remember --from <chat id> --span ...`: the pointer fields, or ValueError when code cannot verify (the record is then not saved)."""
    chat = bridge.Chat(state, home)
    sid = chat.session({"session": session})
    v = bridge.verify(chat.msg, chat.ids(sid), content_msg, span, content_msg) if sid else {}  # authority survives only for an owner message
    if "content_msg" not in v:
        raise ValueError("--from/--span not verified (message not in this session, or span not verbatim in it, 5+ words); record not saved")
    return v


def remember_fields(state, session, frm=None, span=None, home=None):
    out = from_chat(state, session, frm, span, home) if frm is not None else {}
    if "authority_msg" in out:
        out["authority_verdict"] = "said"
    a = last_owner_ptr(session, home) if session else None
    if a:
        out["trigger_msg"] = a
    out["prov_stage"] = "W" if "content_msg" in out else "trigger-only" if a else "unverified"
    return out


def window_fields(msgs, sid, cite):
    """msgs: [(uuid, full text)] numbered 1.. as shown to the model; cite: {"msg": n, "span": str} or None.
    The cited message is owner text, so it is also the authority. -> payload fields (content, authority, stage) or {prov_stage: unverified}."""
    book = {i: {"kind": "user", "text": t} for i, (_, t) in enumerate(msgs, 1)}
    c = cite if isinstance(cite, dict) else {}
    n = c.get("msg")
    v = bridge.verify(book, set(book), n, c.get("span"), n) if isinstance(n, int) and not isinstance(n, bool) else {}
    if "content_msg" not in v or not msgs[n - 1][0]:  # no uuid on the line -> nothing to point at
        return {"prov_stage": "unverified"}
    return {"content_msg": ptr(sid, msgs[n - 1][0]), "content_span": v["content_span"], "authority_msg": ptr(sid, msgs[n - 1][0]), "authority_verdict": "said", "prov_stage": "W"}
