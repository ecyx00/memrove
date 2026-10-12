"""Push observability for the session opening: one warning line, shown only when something is wrong; doctor shows the same signals in full.

Signals: J2 windows pending or quarantined, daily J2 cap near its limit, the last hook error, title-only context, rules not fully visible.
Everything here is best-effort and read-only: a health check must never cost the session its context.
"""
import json
from pathlib import Path

CAP_WARN = .8  # share of the daily J2 cap


def _json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def j2_state(state):
    """The J2 queue's own status (memrove_v3_queue.status); last_error is kept only when it is newer than the last success."""
    from memrove_v3_queue import status
    st = status(state)
    err = st.get("last_error")
    if err and (st.get("last_ok") or 0) >= (err.get("at") or 0):
        err = None
    return dict(st, last_error=err)


def signals(state, title_only=(), rules_total=0, rules_shown=0):
    """-> [short fault descriptions]; empty when everything is fine."""
    out = []
    try:
        j = j2_state(state)
        if j["quarantined"]:
            out.append(f"J2: {j['quarantined']} window(s) quarantined")
        if j["pending"]:
            out.append(f"J2: {j['pending']} session(s) not yet summarized")
        by = j.get("by_job") or {}
        used = ", ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1]))
        if j["pending"] and j["cap_used"] >= j["cap"]:
            out.append(f"J2 starved: cap spent ({used}) with {j['pending']} session(s) waiting")
        elif j["cap_hit_today"] or j["cap_used"] >= j["cap"] * CAP_WARN:
            out.append(f"daily LLM cap {j['cap_used']}/{j['cap']} ({used})")
        if j["last_error"]:
            e = j["last_error"]
            out.append(f"{'J2' if e.get('where') == 'j2' else 'LLM'} last error at {e.get('where')}: {e.get('error')}")
        err = _json(Path(state) / "hook-error.json", {})
        if err.get("error"):
            out.append(f"last hook error: {err['error']}")
    except Exception:
        pass
    if title_only:
        out.append("title-only context (" + ", ".join(title_only) + ")")
    if rules_shown < rules_total:
        out.append(f"rules not fully visible ({rules_shown}/{rules_total}; `memrove search rule`)")
    return out


def line(sigs):
    return ("memrove warning: " + "; ".join(sigs) + ". Run `memrove doctor`.\n") if sigs else ""


def report(state):
    """Doctor view: the signals, the J2 state, and the last opening's per-section token counts (from inject-log)."""
    state = Path(state)
    last = {}
    try:
        for ln in (state / "inject-log.jsonl").read_text(encoding="utf-8").splitlines()[-50:]:
            e = json.loads(ln)
            if e.get("source") in ("startup", "clear") and "sections" in e:
                last = e
    except (OSError, ValueError):
        pass
    return {"signals": signals(state, last.get("title_only", ()), last.get("rules_total", 0), last.get("rules_shown", 0)),
            "j2": j2_state(state), "last_opening": {k: last.get(k) for k in ("source", "tokens", "sections", "rules_total", "rules_shown", "title_only")} if last else None}
