#!/usr/bin/env python3
"""One-off history import: every past owner session on this Mac (Claude Code, Codex, Antigravity) -> J2 records + cards, oldest first,
each stamped with its real date. Resumable (J2 cursors skip finished windows). Created record ids go to state/history-import/<JOB>.jsonl.
Usage: history_import.py --vault V --state S [--max-cost 38] [--list]   then: rules restate + graph link of journal ids (--finish)."""
import argparse, glob, json, os, sqlite3, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "template/.claude/scripts"))
JOB = "history-2026-10-19"
TMP_PREFIXES = ("/tmp", "/private/tmp", "/private/var/folders", "/var/folders")


def local_day(ts):
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (d if d.tzinfo is None else d.astimezone()).date().isoformat()


def order(items):
    """Oldest first across all sources; ties by source then sid so the order is stable."""
    return sorted(items, key=lambda s: (s["start"], s["source"], s["sid"]))


def claude_items(state, home):
    from memrove_v3_capture import user_text
    try:
        cursors = json.loads((Path(state) / "llm-cursors.json").read_text())
    except (OSError, ValueError):
        cursors = {}
    out = []
    for p in glob.glob(f"{home}/.claude/projects/*/*.jsonl"):
        sid, cwd, entry, first, last, users = Path(p).stem, None, None, None, None, 0
        for raw in open(p, "rb"):
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            cwd, entry = cwd or r.get("cwd"), entry or r.get("entrypoint")
            if r.get("timestamp"):
                last = r["timestamp"]
                if user_text(r):
                    users += 1
                    first = first or r["timestamp"]
        if not users or entry == "sdk-cli" or (cwd or "").startswith(TMP_PREFIXES) or cursors.get(sid, 0) >= os.path.getsize(p):
            continue
        out.append({"source": "claude", "sid": sid, "tp": p, "cwd": cwd, "start": local_day(first), "last": local_day(last), "first_ts": first})
    return out


def synth_items(state, vault, home):
    """Codex and Antigravity sessions -> Claude-format jsonl under state/history-import/synth/ so J2 and cards read them unchanged."""
    import memrove_v3_sessions as S
    synth = Path(state) / "history-import" / "synth"
    synth.mkdir(parents=True, exist_ok=True)
    out = []
    for source, reader in (("codex", S.codex_sessions), ("antigravity", S.antigravity_sessions)):
        for s in reader(home):
            turns = [t for t in s["turns"] if t[1]]
            if s["skipped"] or not turns:
                continue
            tp = synth / f"{s['sid']}.jsonl"
            tp.write_text("".join(json.dumps({"type": "user", "timestamp": f"{d}T12:00:00", "cwd": s["cwd"], "sessionId": s["sid"],
                                              "message": {"content": t}}, ensure_ascii=False) + "\n" for t, d, _ in turns), encoding="utf-8")
            out.append({"source": source, "sid": s["sid"], "tp": str(tp), "cwd": s["cwd"], "start": min(t[1] for t in turns), "last": max(t[1] for t in turns),
                        "project": S.project_of(s, state, vault)})
    return out


def spent(state, since):
    t = 0.0
    for l in open(Path(state) / "llm-log.jsonl"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("ts", 0) >= since:
            t += r.get("cost") or 0
    return t


def new_ids(state, sid, before):
    db = sqlite3.connect(Path(state) / "memory.sqlite3", timeout=30)
    rows = db.execute("SELECT id FROM records WHERE json_extract(payload,'$.session')=?", (sid,)).fetchall()
    db.close()
    return sorted({r[0] for r in rows} - before)


def all_ids(state):
    db = sqlite3.connect(Path(state) / "memory.sqlite3", timeout=30)
    s = {r[0] for r in db.execute("SELECT id FROM records")}
    db.close()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--max-cost", type=float, default=38.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--finish", action="store_true", help="restate rules once, then graph-link the journal ids")
    a = ap.parse_args()
    vault, state, home = Path(a.vault).expanduser(), Path(a.state), str(Path.home())
    import memrove_v3_llm as llm, memrove_v3_queue as queue, memrove_v3_card as card
    from memrove_v3_sync import SyncEngine
    from memrove_v3_project import resolve
    queue.config = lambda s, _c=queue.config: {**_c(s), "daily_cap": 10 ** 7}  # this process only: the owner asked for the whole history now
    items = synth_items(state, vault, home) + claude_items(state, home)
    for s in items:
        s.setdefault("project", resolve(s["cwd"], state, vault)[0] or "unknown" if s["cwd"] else "unknown")
    items = order(items)
    if a.list:
        for s in items:
            print(s["start"], s["source"], s["sid"][:12], s["project"])
        return print(len(items), "sessions")
    jdir = Path(state) / "history-import"
    journal, t0 = jdir / f"{JOB}.jsonl", time.time()
    start_file = jdir / f"{JOB}.start"
    if start_file.exists():
        t0 = float(start_file.read_text())  # cost counts from the first run, across resumes
    else:
        start_file.write_text(str(t0))
    engine = SyncEngine(vault, state)
    if a.finish:
        import memrove_v3_graph as graph, memrove_v3_restate as restate
        ids = [json.loads(l)["id"] for l in open(journal)]
        print("restated", restate.restate_rules(engine, vault, state, limit=len(ids) + 200))
        print(graph.link_new(engine, vault, state, limit=len(ids), ids=ids, budget={"left": 10 ** 6, "calls": 0, "cost": 0.0}, daily=False,
                             token_cap=200000, max_cost=max(0.0, a.max_cost - spent(state, t0))))
        return
    for i, s in enumerate(items):
        if spent(state, t0) >= a.max_cost:
            print("STOP: cost", spent(state, t0)); break
        ev = {"event": "SessionEnd", "session": s["sid"], "transcript_path": s["tp"], "project": s["project"], "day": s["start"],
              "at": datetime.fromisoformat(s["start"]).timestamp(), "history": True}
        while not (lock := llm._locked(state)):
            time.sleep(5)  # a live worker holds it; we take it per session only
        try:
            before = all_ids(state)
            n = llm.session_notes(engine, vault, state, [ev], None, {"left": 10 ** 6, "calls": 0, "cost": 0.0})
            card.draft_all(engine, vault, state, [dict(ev, day=s["last"])])
        finally:
            lock.unlink(missing_ok=True)
        ids = new_ids(state, s["sid"], before)
        with open(journal, "a") as f:
            f.writelines(json.dumps({"id": x, "session": s["sid"], "source": s["source"], "day": s["start"]}) + "\n" for x in ids)
        print(f"{i + 1}/{len(items)} {s['start']} {s['source']} {s['sid'][:8]} windows={n} new={len(ids)} cost={spent(state, t0):.2f}", flush=True)


if __name__ == "__main__":
    main()
