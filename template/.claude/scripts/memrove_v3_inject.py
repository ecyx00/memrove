"""Small helpers shared by recall and the llm jobs: the reassess tag of a record and the J1 advice view."""


def reassess_tag(r):
    m = r.get("reassess")
    return f" [reassess -> #{m['by']}: {m.get('reason', '')}]" if isinstance(m, dict) and m.get("by") else ""


def j1_view(r):
    """J1 advice: assessments.j1 first, legacy llm_* fields when absent -> {assessed, durable, drop}."""
    a = (r.get("assessments") or {}).get("j1")
    if a:
        ans = a.get("answers") or {}
        ok = a.get("state") in ("assessed", "unclear")
        return {"assessed": ok, "durable": ok and ans.get("durable") == "yes",
                "drop": a.get("state") == "assessed" and ans.get("standing_rule") == "no"}
    return {"assessed": r.get("llm_verdict") is not None, "durable": bool(r.get("llm_durable")),
            "drop": r.get("llm_verdict") == "drop"}
