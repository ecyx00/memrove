#!/usr/bin/env python3
"""memrove command line: chat memory, saved records, project views and maintenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True  # a read-only command must not write .pyc files next to the installed scripts
HERE = Path(__file__).resolve().parent
MODULES = HERE if (HERE / "memrove_v3_sync.py").exists() else HERE.parent / "template/.claude/scripts"
sys.path.insert(0, str(MODULES))

RECORD_KINDS = ("decision", "rejected", "lesson", "preference", "note")
WRITES_STATE = {"sync", "remember", "backfill"}  # every other command opens the store read-only


def default_state(vault) -> Path:
    """Mutable state lives outside the vault, one directory per canonical vault path."""
    if sys.platform == "darwin":
        base = Path.home().joinpath("Library", "Application Support")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return base / "memrove-v3" / hashlib.sha256(str(vault).encode()).hexdigest()[:16]


def project_name(cwd: Path, state=None) -> str:
    """Project of a directory: the same root rule the hook and recall use."""
    from memrove_v3_project import project_root
    return project_root(cwd, state).name


def current_session(cwd: Path):
    """Session id from the environment, else the newest Claude Code transcript of this directory."""
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID") or os.environ.get("MEMROVE_SESSION")
    if sid:
        return sid
    folder = Path.home() / ".claude/projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    newest = max(folder.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, default=None) if folder.is_dir() else None
    return newest.stem if newest else None


# --- parser ---------------------------------------------------------------------------------------

def opt(*flags, **kw):
    return flags, kw


ON_OFF = dict(choices=("on", "off"))
COMMANDS = {
    "init": ("Initialise local state; installs no hooks or services", []),
    "sync": ("Reconcile Markdown sources into local state", []),
    "wake": ("Chat memory: print the view (id+n|summary lines for the whole history), one part at a time", [
        opt("part", nargs="?", type=int, default=1), opt("--all", action="store_true", help="print every part")]),
    "zoom": ("Chat memory: open line id+n into the two lines it was made from; n=1 gives message id whole", [
        opt("id", type=int), opt("n", type=int), opt("--page", type=int, default=1)]),
    "date": ("Chat memory: the date and time of message id", [opt("id", type=int)]),
    "doc": ("Print a docs write-up (or one section) that a search handle points to; indexed doc roots only", [
        opt("path"), opt("--section", help="heading of the section to print; default the whole file")]),
    "rebrief": ("Metric: owner prompts that ask for the same thing again in another session", [
        opt("--out", default="/tmp/e71-out/rebrief-pairs.jsonl"), opt("--min-score", type=float, default=0.5),
        opt("--since", help="only prompts from YYYY-MM-DD on")]),
    "chatlog": ("Chat memory: run, status, stability, prepare, apply, ingest, search", [
        opt("action", choices=("run", "status", "stability", "prepare", "apply", "ingest", "search")),
        opt("--since", help="run: only messages from YYYY-MM-DD on; search: from this ISO date or datetime on"),
        opt("--lean-before", metavar="YYYY-MM-DD", help="run: before that day keep only user messages and final agent answers"),
        opt("--home", type=Path, help="run: where .claude/.codex/.gemini live (default: the home directory)"),
        opt("--max-calls", type=int, help="run: claude -p calls for this run (default: chatlog_cap in llm-config.json minus today's)"),
        opt("--dry-run", action="store_true", help="run: ingest and project the calls, make none"),
        opt("--no-force", action="store_true", help="run: keep the gate (20 nodes ready, view over its limit, or oldest waited 4 h)"),
        opt("batch_id", nargs="?", help="apply: the batch_id `prepare` printed (raw reply on stdin); search: the query"),
        opt("--no-ingest", action="store_true", help="prepare: skip the ingest (the transport ran it already)"),
        opt("--until", help="search: only messages up to this ISO date (inclusive) or datetime"),
        opt("--json", action="store_true", help="prepare: print {batch_id, system, prompt, node_ids} (or {})"),
        opt("--error", help="apply: the transport's error text instead of a reply (a refusal halves the batch)"),
        opt("--last", type=int, default=50, help="stability: appends measured")]),
    "backfill": ("Run the session-note Haiku job over old transcripts, oldest first, under its own call cap", [
        opt("--transcripts", required=True, type=Path, help="directory of *.jsonl transcripts"),
        opt("--max-calls", type=int, default=60, help="hard cap on Haiku calls for this run"),
        opt("--since", help="only transcripts modified on or after YYYY-MM-DD"),
        opt("--rescan", metavar="SESSION_ID|all", help="rewind that session's (or every) J2 cursor to 0 and reprocess"),
        opt("--rebuild-threads", metavar="PROJECT|all", help="drop memrove-owned threads and cards of that project, reset its cursors, rebuild"),
        opt("--dry-run", action="store_true", help="print the session count and estimated cost only")]),
    "import-sessions": ("Import past Codex / Antigravity sessions: the user's own turns, no model calls, idempotent", [
        opt("--harness", choices=("codex", "antigravity"), required=True),
        opt("--dry-run", action="store_true", help="report what would be imported; write nothing"),
        opt("--home", type=Path, help="home directory holding .codex / .gemini"),
        opt("--sample", type=int, default=5, help="sample size in the report")]),
    "backup": ("Copy vault and state (consistent SQLite snapshot plus manifest) into an empty --dest", [
        opt("--dest", required=True, type=Path)]),
    "restore": ("Restore a backup into an empty vault and state directory; never overwrites", [
        opt("--from", dest="source", required=True, type=Path)]),
    "doctor": ("Hook health, pending queue, writer heartbeats and size reports", []),
    "recall-stats": ("Were injected cards opened, cited or ignored? Counts and rates; read-only", [
        opt("--days", type=int, default=7), opt("--json", action="store_true"),
        opt("--real-use", action="store_true", help="instead: do the injected memory lines fire in the live transcripts, and get followed?"),
        opt("--backfill", action="store_true", help="first run the detector over transcripts into a temp copy of the decision log"),
        opt("--backfill-live", action="store_true", help="like --backfill but append the outcomes to the live decision log")]),
    "preferences": ("Control the automatic background sync, the secret filter and recall cards", [
        opt("--profile", choices=("normal", "economical", "manual")), opt("--auto-sync", **ON_OFF),
        opt("--interval-minutes", type=int), opt("--secret-filter", **ON_OFF),
        opt("--recall-cards", help="Claude per-prompt recall cards (default off; MEMROVE_NO_RECALL=1 still wins)", **ON_OFF)]),
    "companion-compact": ("Move older Last-Session/Threads entries verbatim into a private archive; deletes nothing", [
        opt("--dry-run", action="store_true", help="report what would move without writing")]),
    "search": ("Chat messages and saved records in one ranked list: id|date|kind|snippet (rec:<id> for a record)", [
        opt("query", nargs="?", default=""), opt("--limit", type=int, help="hits (default 8; with --records 10)"),
        opt("--since", help="only from this date on (YYYY-MM-DD or ISO datetime)"), opt("--until", help="only up to this date, inclusive"),
        opt("--history", action="store_true", help="also show entries a newer topic head replaced"),
        opt("--records", action="store_true", help="records only, all kinds, JSON"),
        opt("--compact", action="store_true", help="with --records: one line per hit")]),
    "show": ("One memory in full (id or unique prefix from a recall card)", [opt("id")]),
    "wiki": ("Concept wiki tools for the writer agent: evidence | check | publish | source fetch | research catalog", [
        opt("action", choices=("check", "research", "source", "evidence", "publish")),
        opt("page", nargs="?", help="check/publish: the page file; research: catalog; source: fetch"),
        opt("rest", nargs="*", help="source fetch: <repo|paper|model>:<id>"),
        opt("--topic", action="append", default=[], help='evidence: "Title::terms=a,b::sym=s1::docs=a::msg=id::chat=q1|q2::roots=proj=/abs::skip=w"'),
        opt("--slug", help="source: page slug (default: last part of the id)")]),
    "grab": ("One-key capture into wiki sources: install | remove | status | hotkey | add | process | done | list", [
        opt("rest", nargs=argparse.REMAINDER, help="arguments of the grab subcommand (see: memrove grab --help)")]),
    "project": ("Where a project stands: latest decisions (superseded marked), open loops, last session", [
        opt("name", nargs="?", help="project name (default: the project of the current directory)"),
        opt("--max-bytes", type=int, default=5000)]),
    "graph": ("Link graph: neighbors <id|title> | hub <project> | related \"<text>\" | link <a> <b> --type T --why W | concept <name> | run", [
        opt("rest", nargs=argparse.REMAINDER, help="arguments of the graph subcommand (see: memrove graph --help)")]),
    "remember": ("Save one decision, rejected option, lesson, preference or note; prints the id", [
        opt("text"), opt("--kind", required=True, choices=RECORD_KINDS), opt("--why"),
        opt("--project", help="default: project root name of --cwd"),
        opt("--user", action="store_true", help="the text quotes the user verbatim (origin user_explicit)"),
        opt("--session", help="default: CLAUDE_CODE_SESSION_ID, else the newest transcript of --cwd"),
        opt("--supersedes", action="append", default=[], metavar="ID", help="record id (or unique prefix) this one replaces; repeatable"),
        opt("--topic", help="slug of the topic this decision belongs to; the newest entry per topic is its head (the nudge lists existing slugs)"),
        opt("--cwd", default=".")]),
}


def parser():
    root = argparse.ArgumentParser(prog="memrove", description=__doc__)
    root.add_argument("--vault", required=True, type=Path, help="existing vault directory")
    root.add_argument("--state", type=Path, help="local state directory outside the vault")
    sub = root.add_subparsers(dest="command", metavar="command", required=True)
    for name, (help_text, options) in COMMANDS.items():
        command = sub.add_parser(name, help=help_text)
        for flags, kw in options:
            command.add_argument(*flags, **kw)
    return root


# --- helpers --------------------------------------------------------------------------------------

class Context:
    """Vault and state of one run, with lazily opened store and sync engine."""

    def __init__(self, vault: Path, state: Path):
        self.vault, self.state = vault, state

    def store(self, read_only=True):
        from memrove_v3 import MemoryStore
        return MemoryStore(self.state, self.vault, read_only=read_only)

    def engine(self):
        from memrove_v3_sync import SyncEngine
        return SyncEngine(self.vault, self.state)


def emit(result, ensure_ascii=True):
    print(json.dumps(result, ensure_ascii=ensure_ascii, indent=2))
    return 0


# --- commands -------------------------------------------------------------------------------------

def cmd_init(args, ctx):
    return emit({"initialized": True, "state": str(ctx.state), "network": False, "hooks_installed": False, "optional_provider": None})


def cmd_sync(args, ctx):
    return emit(ctx.engine().sync())


def cmd_chat(args, ctx):
    """wake / zoom / date / chatlog read and write the chat log only; no record store."""
    import memrove_v3_chattree as tree
    import memrove_v3_chatsearch as chat_search
    state, command = ctx.state, args.command
    if command == "wake":
        print(tree.wake_all(state) if args.all else tree.wake(state, args.part))
    elif command == "zoom":
        print(tree.zoom(state, args.id, args.n, args.page))
    elif command == "date":
        print(tree.date_of(state, args.id))
    elif args.action == "run":
        log = lambda message: print(message, file=sys.stderr)
        print(json.dumps(tree.run(state, ctx.vault, args.home, args.since, args.max_calls, args.dry_run, log,
                                  args.lean_before, not args.no_force), indent=2))
    elif args.action == "prepare":
        print(json.dumps(tree.prepare(state, ctx.vault, args.home, ingest=not args.no_ingest), ensure_ascii=False))
    elif args.action == "ingest":
        print(json.dumps(tree.ingest_only(state, args.home)))
    elif args.action == "apply":
        if not args.batch_id:
            raise ValueError("apply needs the batch_id")
        print(json.dumps(tree.apply(state, args.batch_id, None if args.error else sys.stdin.read(), args.error)))
    elif args.action == "search":
        if not (args.batch_id or args.since or args.until):
            raise ValueError("search needs a query or --since / --until")
        print(chat_search.search(state, args.batch_id or "", since=args.since, until=args.until))
    elif args.action == "status":
        print(json.dumps(tree.status(state), indent=2))
    else:
        print(json.dumps(tree.replay_stability(state, args.last), indent=2))
    return 0


def cmd_rebrief(args, ctx):
    import memrove_v3_pastref
    print(memrove_v3_pastref.rebrief_command(ctx.state, args.out, args.min_score, args.since))
    return 0


def cmd_doc(args, ctx):
    import memrove_v3_docsearch
    print(memrove_v3_docsearch.show(args.path, args.section))
    return 0


def cmd_grab(args, ctx):
    """Capture tool. It reads no store; only `process` writes a memrove note, through the same remember path as the CLI."""
    import memrove_v3_grab as grab

    def note(text, why):
        from memrove_v3_context import remember
        # a capture belongs to no project: _general is the wiki's cross-project marker
        return remember(ctx.engine(), "note", text, why, "_general", False, current_session(Path.cwd()))

    return grab.main(args.rest, vault=ctx.vault, state=ctx.state, note=note)


def cmd_project(args, ctx):
    import memrove_v3_project as project
    name = args.name or project.resolve(Path.cwd(), ctx.state, ctx.vault)[0]
    if not name:
        print("no project for this directory; pass a name", file=sys.stderr)
        return 2
    with ctx.store()._connect() as db:
        print(project.entry(db, ctx.vault, name, args.max_bytes, state=ctx.state))
    return 0


def cmd_graph(args, ctx):
    import memrove_v3_graph as graph
    return graph.main(args.rest, ctx.vault, ctx.state, lambda read_only: ctx.store(read_only), ctx.engine)


def cmd_wiki(args, ctx):
    if args.action == "source":  # wiki source fetch <repo|paper|model>:<owner/name | arxiv id> [--slug S]
        import memrove_v3_sources as sources
        if args.page != "fetch" or len(args.rest) != 1 or ":" not in args.rest[0]:
            raise ValueError("usage: wiki source fetch <repo|paper|model>:<owner/name | arxiv id> [--slug S]")
        kind, ident = args.rest[0].split(":", 1)
        slug = args.slug or re.sub(r"[^a-z0-9]+", "-", ident.lower().rsplit("/", 1)[-1]).strip("-")
        print(json.dumps(sources.fetch(ctx.vault, kind, ident, slug), indent=1))
        return 0
    if args.action == "research":
        import memrove_v3_research as research
        if args.page != "catalog":
            raise ValueError("usage: wiki research catalog")
        return emit(research.catalog(ctx.vault, ctx.state), ensure_ascii=False)
    import memrove_v3_wiki as wiki
    store = ctx.store()
    if args.action in ("check", "publish"):
        if not args.page:
            raise ValueError(f"wiki {args.action} needs the page file")
        run = wiki.check if args.action == "check" else wiki.publish
        errors = run(args.page, store, ctx.state, ctx.vault)
        print(("OK" if args.action == "check" else "PUBLISHED") if not errors else "FAIL\n" + "\n".join(errors))
        return 1 if errors else 0
    if not args.topic:
        raise ValueError('wiki evidence needs --topic "Title::terms=a,b::roots=proj=/abs::msg=id,id"')
    print(wiki.evidence_pack(store, ctx.state, ctx.vault, wiki.parse_topic(args.topic[0])))
    return 0


def cmd_backup(args, ctx):
    import memrove_v3_backup as backups
    return emit(backups.backup(ctx.vault, ctx.state, args.dest.expanduser()))


def cmd_restore(args, ctx):
    import memrove_v3_backup as backups
    return emit(backups.restore(args.source.expanduser(), ctx.vault, ctx.state))


def cmd_preferences(args, ctx):
    import memrove_v3_preferences as preferences
    import memrove_v3_push as push
    changes = {"interval_minutes": args.interval_minutes} if args.interval_minutes is not None else {}
    for flag in ("auto_sync", "secret_filter"):
        if getattr(args, flag) is not None:
            changes[flag] = getattr(args, flag) == "on"
    saved = preferences.save(ctx.vault, changes, args.profile) if changes or args.profile else preferences.read(ctx.vault)
    pushed = {"recall_cards": args.recall_cards == "on"} if args.recall_cards else {}
    return emit({"status": "saved" if changes or args.profile or pushed else "current", "preferences": saved,
                 "push": push.save(ctx.state, pushed) if pushed else push.read(ctx.state),
                 "model_calls": False, "timer_installed": False})


def cmd_recall_stats(args, ctx):
    from datetime import datetime, timedelta
    if args.real_use:
        import memrove_v3_realuse as realuse
        since = (datetime.now() - timedelta(days=args.days - 1)).strftime("%Y-%m-%d")
        result = realuse.stats(Path.home() / ".claude" / "projects", since=since)
        if not args.json:
            print(realuse.render(result))
            return 0
        return emit(result)
    import memrove_v3_outcomes as outcomes
    lookup = outcomes.make_lookup(ctx.state, ctx.store()._connect)
    note = None
    if args.backfill or args.backfill_live:
        home, resolved, made = outcomes.backfill(ctx.state, lookup, days=args.days, live=args.backfill_live)
        note = {"decide_home": home, "sessions_resolved": resolved, "injections_synthesised": made}
        if not args.backfill_live:
            os.environ["DECIDE_HOME"] = home
    result = dict(outcomes.stats(lookup, days=args.days), backfill=note)
    if not args.json:
        print(outcomes.render(result) + (f"\nbackfill: {note}" if note else ""))
        return 0
    return emit(result)


def _guarded(report, name, call):
    """Doctor sections are advisory: one that fails reports its error and never hides the others."""
    try:
        report[name] = call()
    except Exception as error:
        report[name] = {"status": "unavailable", "error": type(error).__name__}


def cmd_doctor(args, ctx):
    state = ctx.state
    queue, done = (len(list((state / d).glob("*.json"))) for d in ("hook-queue", "hook-done"))
    report = {"pending_events": queue, "acknowledged_events": done}
    for name in ("hook-health.json", "hook-error.json"):
        report[name] = json.loads((state / name).read_text(encoding="utf-8")) if (state / name).exists() else None
    seen = {"codex": set(), "claude": set()}
    for path in (state / "hook-done").glob("*.json"):
        event = json.loads(path.read_text(encoding="utf-8"))
        if event.get("harness") in seen:
            seen[event["harness"]].add(event.get("event", "unknown"))
    report["lifecycle"] = {name: {"status": "observed_metadata" if events else "never_seen", "events": sorted(events)}
                           for name, events in seen.items()}
    import memrove_v3_companion as companion
    import memrove_v3_docs
    import memrove_v3_health
    import memrove_v3_heartbeat
    import memrove_v3_llm
    import memrove_v3_preferences
    import memrove_v3_references
    from memrove_v3_secrets import health as secret_health
    store = ctx.store(read_only=False)  # a fresh runtime is created, not an error
    report["preferences"] = memrove_v3_preferences.read(ctx.vault)
    report["secret_filter"] = secret_health(state)
    report["secrets_redacted"] = report["secret_filter"]["total"]
    _guarded(report, "writers", lambda: memrove_v3_heartbeat.check(state))
    _guarded(report, "docs", lambda: memrove_v3_docs.doctor(store))
    def topic_heads():
        import memrove_v3_heads
        total, bare = memrove_v3_heads.topicless_recent()
        return {"writeups_7d": total, "without_topic": bare, "front_matter_share": round((total - bare) / total, 2) if total else None,
                "heads": "off (MEMROVE_HEADS=0)" if not memrove_v3_heads.enabled() else "on",
                "line": f"{bare} write-ups in the last 7 days without a topic" + (f"; front matter on {total - bare}/{total} ({round(100 * (total - bare) / total)}%)" if total else "")}
    _guarded(report, "topic_heads", topic_heads)
    _guarded(report, "llm", lambda: memrove_v3_llm.doctor(store))
    _guarded(report, "context_health", lambda: memrove_v3_health.report(state))
    _guarded(report, "companion_hygiene", lambda: companion.hygiene(ctx.vault, state))
    _guarded(report, "instruction_references", lambda: memrove_v3_references.check(ctx.vault))
    sync_status = (report["hook-health.json"] or {}).get("sync", {}).get("status")
    report["status"] = ("needs_attention" if sync_status in ("conflict", "degraded") or report["hook-error.json"]
                        else "pending" if queue else "observed_metadata" if done else "never_seen")
    return emit(report)


def cmd_compact(args, ctx):
    import memrove_v3_compact
    result = memrove_v3_compact.compact(ctx.vault, ctx.state, dry_run=args.dry_run)
    if any(entry["status"] == "compacted" for entry in result["files"].values()):
        result["sync"] = {"status": ctx.engine().sync().get("status")}  # index the shorter files and the archive
    return emit(result)


def cmd_backfill(args, ctx):
    transcripts = args.transcripts.expanduser()
    if args.max_calls < 1 or (args.since and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.since)):
        raise ValueError("--max-calls must be >= 1 and --since YYYY-MM-DD")
    if not transcripts.is_dir():
        raise ValueError("--transcripts must be an existing directory")
    import memrove_v3_llm
    return emit(memrove_v3_llm.backfill(ctx.engine(), ctx.vault, ctx.state, transcripts, args.max_calls, args.since,
                                        args.dry_run, rescan=args.rescan, rebuild_threads=args.rebuild_threads))


def cmd_import_sessions(args, ctx):
    import memrove_v3_sessions as sessions
    return emit(sessions.run(args.harness, ctx.vault, ctx.state, args.dry_run, args.home, args.sample))


def cmd_search(args, ctx):
    if args.records:
        from memrove_v3_recall import search
        hits = search(ctx.store(), args.query, args.limit or 10)
        if not args.compact:
            return emit(hits)
        for hit in hits:
            print(f"#{hit['id']} {hit['line'].replace('[other project: ', '[', 1)}")
        return 0
    if not args.query.strip() and not (args.since or args.until):
        raise ValueError("search needs a query or --since / --until")
    import memrove_v3_chatsearch as chat_search
    print(chat_search.search(ctx.state, args.query, *([args.limit] if args.limit else []), since=args.since, until=args.until, history=args.history))
    return 0


def cmd_show(args, ctx):
    from memrove_v3_recall import show
    return emit(show(ctx.store(), args.id.removeprefix("rec:")), ensure_ascii=False)


def cmd_remember(args, ctx):
    from memrove_v3_context import remember
    cwd = Path(args.cwd).expanduser().resolve()
    project = args.project or project_name(cwd, ctx.state)
    result = remember(ctx.engine(), args.kind, args.text, args.why, project, args.user,
                      args.session or current_session(cwd), supersedes=args.supersedes, topic=args.topic)
    if result["status"] == "exists":
        print(result["id"] + " (exists)", file=sys.stderr)
    print(result["id"])
    if result["status"] != "exists" and not args.supersedes and args.kind != "note":
        try:
            from memrove_v3_nudge import nudge
            for line in nudge(ctx.state, project, args.text, exclude=result["id"]):
                print(line, file=sys.stderr)
        except Exception:  # the nudge is a courtesy; the record is already saved
            pass
    if args.kind in ("decision", "rejected"):
        try:
            from memrove_v3_nudge import topic_line
            line = topic_line(ctx.state, project)
            if line:
                print(line, file=sys.stderr)
        except Exception:
            pass
    return 0


HANDLERS = {"init": cmd_init, "sync": cmd_sync, "wake": cmd_chat, "zoom": cmd_chat, "date": cmd_chat, "chatlog": cmd_chat,
            "rebrief": cmd_rebrief, "doc": cmd_doc, "grab": cmd_grab, "project": cmd_project, "graph": cmd_graph,
            "wiki": cmd_wiki, "backup": cmd_backup, "restore": cmd_restore, "preferences": cmd_preferences,
            "recall-stats": cmd_recall_stats, "doctor": cmd_doctor, "companion-compact": cmd_compact,
            "backfill": cmd_backfill, "import-sessions": cmd_import_sessions, "search": cmd_search, "show": cmd_show,
            "remember": cmd_remember}


def resolve_paths(args):
    """-> (vault, state), both absolute; the state must live outside the vault."""
    vault = args.vault.expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("--vault must be an existing directory")
    state = (args.state.expanduser() if args.state else default_state(vault)).resolve()
    if vault == state or vault in state.parents:
        raise ValueError("--state must be outside the vault")
    return vault, state


def main(argv=None):
    for stream in (sys.stdin,):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        return HANDLERS[args.command](args, Context(*resolve_paths(args)))
    except Exception as exc:  # no traceback and no input echo: the caller still has its source files
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
