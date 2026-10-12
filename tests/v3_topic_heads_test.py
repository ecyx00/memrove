#!/usr/bin/env python3
"""Topic heads: write-up front matter, remember --topic, heads view, project/search presentation, threads off, doctor line, agent rule."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True


class TopicHeads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name).resolve()
        self.vault, self.state, self.proj = tmp / "Vault", tmp / "state", tmp / "work" / "myproj"
        self.vault.mkdir()
        (self.proj / ".git").mkdir(parents=True)
        self.docs = self.proj / "docs" / "state"
        self.docs.mkdir(parents=True)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}
        self.env.update(HOME=str(tmp), CLAUDE_CODE_SESSION_ID="s1", MEMROVE_NO_LLM="1", MEMROVE_DOC_REPOS=str(self.proj))
        patch = mock.patch.dict(os.environ, {"MEMROVE_DOC_REPOS": str(self.proj)})
        patch.start()
        self.addCleanup(patch.stop)

    def cli(self, *a):
        return subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), *a],
                              env=self.env, capture_output=True, text=True)

    def remember(self, text, topic, *extra, kind="decision"):
        r = self.cli("remember", text, "--kind", kind, "--topic", topic, "--why", "because", "--cwd", str(self.proj), *extra)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip(), r.stderr

    def test_front_matter_and_filename_fallback(self):
        import memrove_v3_docs as docs
        import memrove_v3_heads as H
        fm = docs.front_matter("---\ntopic: e92-ledger\nstatus: Done\ndecision: \"Drop it, 0.4 precision\"\nnext: none\nsupersedes: old-topic\n---\n# T\n")
        self.assertEqual(fm, {"topic": "e92-ledger", "status": "done", "decision": "Drop it, 0.4 precision", "next": "none", "supersedes": "old-topic"})
        self.assertEqual(docs.front_matter("# no front matter\n"), {})
        self.assertNotIn("status", docs.front_matter("---\nstatus: maybe\n---\n"))
        (self.docs / "2026-10-12-e92-ledger.md").write_text("# Ledger\nbody\n")
        w = H._writeup(self.docs / "2026-10-12-e92-ledger.md", "myproj", self.docs)
        self.assertEqual((w["topic"], w["status"], w["day"], w["keyed"]), ("e92", "", "2026-10-12", False))  # experiment prefix groups

    def test_remember_topic_and_nudge_lists_slugs(self):
        _, err = self.remember("Use zebrafish queue for ingestion", "ingest-queue")
        self.assertIn("topics of myproj (reuse with --topic): ingest-queue", err)
        time.sleep(1.1)  # recorded_at has second resolution
        _, err = self.remember("Paint the bike shed green", "shed-colour")
        self.assertIn("topics of myproj (reuse with --topic): shed-colour, ingest-queue", err)
        import sqlite3
        db = sqlite3.connect(self.state / "memory.sqlite3")
        topics = {json.loads(p).get("topic") for (p,) in db.execute("SELECT payload FROM records")}
        db.close()
        self.assertEqual(topics, {"ingest-queue", "shed-colour"})

    def test_heads_newest_wins_user_is_kept_history_stays(self):
        import memrove_v3_heads as H
        a = {"project": "p", "topic": "t", "keyed": True, "status": "", "decision": "x", "next": "", "supersedes": [], "ts": "", "kind": "decision"}
        old = dict(a, id="rec:old", ref="rec:old", day="2026-10-01", user=False)
        new = dict(a, id="rec:new", ref="rec:new", day="2026-10-02", user=False)
        usr = dict(a, id="rec:usr", ref="rec:usr", day="2026-10-03", user=True)
        late = dict(a, id="rec:late", ref="rec:late", day="2026-10-04", user=False)
        hs, rep = H.build([old, new])
        self.assertEqual(([h["id"] for h in hs], rep), (["rec:new"], {"rec:old": "rec:new"}))
        hs, rep = H.build([old, usr, late])  # an agent record never replaces a --user record
        self.assertEqual(([h["id"] for h in hs], rep), (["rec:usr"], {"rec:old": "rec:usr"}))
        hs, rep = H.build([old, dict(late, topic="other", supersedes=["rec:old"])])  # explicit supersedes across topics
        self.assertEqual(rep, {"rec:old": "rec:late"})

    def test_project_search_and_history(self):
        old, _ = self.remember("Use zebrafish queue for ingestion", "ingest-queue")
        time.sleep(1.1)
        new, _ = self.remember("Switch ingestion to a pelican stream", "ingest-queue")
        (self.docs / "2026-10-12-pelican-plan.md").write_text("---\ntopic: pelican-plan\nstatus: open\ndecision: Build the pelican stream first because it is cheaper\nnext: write the adapter\n---\n# Pelican plan\nbody\n")
        out = self.cli("project", "myproj").stdout
        lines = out.splitlines()
        self.assertTrue(lines[1].startswith("heads"), out)
        self.assertRegex(out, r"\[pelican-plan\] open 2026-10-12 Build the pelican stream first because it is cheaper \| next: write the adapter \| docs/state/2026-10-12-pelican-plan.md")
        self.assertRegex(out, r"\[ingest-queue\] [^\n]*pelican stream[^\n]*rec:" + new)
        self.assertNotIn("zebrafish queue for ingestion (why", out)  # replaced entry is not a head
        s = self.cli("search", "ingestion zebrafish queue").stdout
        self.assertRegex(s, r"rec:" + old + r"\|[^\n]*\[replaced by rec:" + new)  # stays in search, tagged
        self.assertGreater(s.index("rec:" + old), s.index("head|"))  # below the head
        self.assertIn("rec:" + old, self.cli("search", "ingestion zebrafish queue", "--history").stdout)
        s = self.cli("search", "pelican stream adapter").stdout.splitlines()
        self.assertTrue(s[0].startswith("head|"), s)  # the head sits above raw hits

    def test_same_day_writeups_newer_filename_wins_and_supersedes(self):
        (self.docs / "2026-10-10-alpha.md").write_text("---\ntopic: shared\nstatus: open\ndecision: first\n---\n")
        (self.docs / "2026-10-11-beta.md").write_text("---\ntopic: shared\nstatus: done\ndecision: second\n---\n")
        (self.docs / "2026-10-09-gamma.md").write_text("# gamma\n")
        (self.docs / "2026-10-12-delta.md").write_text("---\ntopic: delta\nsupersedes: 2026-10-09-gamma.md\n---\n")
        import memrove_v3_heads as H
        hs, rep = H.heads(self.state, "myproj")
        self.assertEqual([h["topic"] for h in hs], ["delta", "shared"])
        self.assertEqual(hs[1]["decision"], "second")
        self.assertEqual(len(rep), 2)

    def test_backfill_groups_experiments_and_extracts_decision(self):
        import memrove_v3_heads as H
        (self.docs / "2026-10-01-e87-arms.md").write_text("# Arms\n\nintro line\n")
        (self.docs / "2026-10-05-e87-baseline-v2-report.md").write_text(
            "# Baseline\n\nSome intro.\n\n## Context\nx\n\n## Verdict\n- Baseline holds at 0.68. More detail follows.\n\n## Next steps\n1. Run the reader once\n")
        (self.docs / "2026-10-06-plain-note.md").write_text("# Plain\n\nFirst real line here. Second.\n")
        hs, rep = H.heads(self.state, "myproj")
        by = {h["topic"]: h for h in hs}
        self.assertEqual(sorted(by), ["e87", "plain-note"])
        self.assertEqual(by["e87"]["ref"], "docs/state/2026-10-05-e87-baseline-v2-report.md")
        self.assertEqual(rep, {})  # backfilled topics group as related, never replaced
        self.assertEqual(by["e87"]["related"], ["docs/state/2026-10-01-e87-arms.md"])
        self.assertEqual(by["e87"]["decision"], "Baseline holds at 0.68.")
        self.assertEqual(by["e87"]["next"], "Run the reader once")
        self.assertEqual(by["plain-note"]["decision"], "First real line here.")

    def test_threads_cards_off_and_heartbeat_not_silent(self):
        import memrove_v3_heartbeat as hb
        import memrove_v3_threads as th
        import memrove_v3_llm as llm
        self.assertNotIn("j2_threads", hb.WRITERS)
        self.assertFalse(th.enabled())
        calls = []
        orig = th.apply_threads
        th.apply_threads = lambda *a, **k: calls.append(a) or []
        try:
            llm._thread_files(self.vault, self.state, {"threads": [{"title": "x"}]}, {"project": "p"}, "sid12345", "n", "2026-10-12", [])
        finally:
            th.apply_threads = orig
        self.assertEqual(calls, [])
        out = self.cli("project", "myproj").stdout
        self.assertNotIn("last session", out)

    def test_doctor_counts_topicless_recent_writeups(self):
        today = date.today()
        (self.docs / f"{today.isoformat()}-a.md").write_text("---\ntopic: a\n---\n")
        (self.docs / f"{today.isoformat()}-b.md").write_text("# b\n")
        (self.docs / f"{(today - timedelta(days=30)).isoformat()}-c.md").write_text("# c\n")
        import memrove_v3_heads as H
        self.assertEqual(H.topicless_recent(today=today), (2, 1))
        r = self.cli("doctor", "--json") if False else self.cli("doctor")
        self.assertIn("1 write-ups in the last 7 days without a topic", r.stdout, r.stderr[-300:])

    def test_agent_rule_asks_for_front_matter(self):
        import memrove_v3_jit as jit
        self.assertIn("front matter: topic", jit.SUBAGENT_RULE)
        self.assertEqual(jit.RULE.count("front matter"), 1)

    def test_h3_backfilled_facets_stay_visible(self):
        import memrove_v3_heads as H
        (self.docs / "2026-10-01-e87-job-set.md").write_text("# Job set\n\nThe job set is fixed. More.\n")
        (self.docs / "2026-10-05-e87-arms.md").write_text("# Arms\n\nThree arms. More.\n")
        hs, rep = H.heads(self.state, "myproj")
        self.assertEqual(rep, {})
        self.assertIn("e87-job-set", " ".join(hs[0]["related"]))
        self.assertIn("related: docs/state/2026-10-01-e87-job-set.md", H.line(hs[0]))
        (self.docs / "2026-10-06-e87-arms-v2.md").write_text("---\ntopic: e87-arms\n---\n")  # an explicit key still replaces keyed entries only
        (self.docs / "2026-10-07-e87-arms-v3.md").write_text("---\ntopic: e87-arms\n---\n")
        _, rep = H.heads(self.state, "myproj")
        self.assertEqual(list(rep), [str(self.docs / "2026-10-06-e87-arms-v2.md")])

    def test_h1_start_view_newest_five_capped_and_empty(self):
        import memrove_v3_heads as H
        import memrove_v3_hook as hook
        payload = {"source": "startup", "cwd": str(self.proj)}
        self.assertNotIn("heads", hook.session_notice(self.state, payload))
        for i in range(7):
            (self.docs / f"2026-10-0{i + 1}-t{i}.md").write_text(f"---\ntopic: topic-{i}\nstatus: open\ndecision: decision number {i}\nnext: do {i}\n---\n")
        out = hook.session_notice(self.state, payload)
        self.assertIn("[topic-6] open decision number 6 | next: do 6", out)
        self.assertIn("[topic-2]", out)
        self.assertNotIn("[topic-1]", out)
        self.assertEqual(H.start_view(self.state, str(self.proj), cap=60).count("[topic-"), 1)  # cap drops lines, never cuts one
        self.assertEqual(H.start_view(self.state, str(self.proj.parent)), "")  # another cwd, no heads

    def test_h2_write_time_check(self):
        import memrove_v3_jit as jit
        (self.docs / "2026-10-01-ingest-queue.md").write_text("---\ntopic: ingest-queue\nstatus: open\ndecision: x\nnext: y\n---\n")
        bare = self.docs / "2026-10-13-ingest-queue-retry.md"
        bare.write_text("# Ingest queue retry\n\nbody\n")
        call = lambda path, tool="Write": jit.handle({"hook_event_name": "PostToolUse", "tool_name": tool, "tool_input": {"file_path": str(path)}}, str(self.state))
        out = call(bare)["hookSpecificOutput"]
        self.assertEqual(out["hookEventName"], "PostToolUse")
        for key in ("topic", "status", "decision", "next"):
            self.assertIn(key, out["additionalContext"])
        self.assertIn("existing topics: ingest-queue", out["additionalContext"])
        self.assertEqual(out["additionalContext"].count("\n"), 0)
        new = self.docs / "2026-10-13-queue-retry.md"
        new.write_text("---\ntopic: queue-retry\nstatus: open\ndecision: d\nnext: n\n---\n")
        self.assertIn("existing topics: ingest-queue", call(new, "Edit")["hookSpecificOutput"]["additionalContext"])
        ok = self.docs / "2026-10-13-ingest-queue-b.md"
        ok.write_text("---\ntopic: ingest-queue\nstatus: open\ndecision: d\nnext: n\n---\n")
        self.assertIsNone(call(ok))
        self.assertIsNone(call(self.proj / "README.md"))
        self.assertIsNone(call(self.docs / "missing.md"))
        self.assertIsNone(call(bare, "Read"))
        with mock.patch.dict(os.environ, {"MEMROVE_HEADS": "0"}):
            self.assertIsNone(call(bare))
        r = subprocess.run([sys.executable, str(ROOT / "template/.claude/scripts/memrove_v3_jit.py"), "--state", str(self.state)], input=json.dumps(
            {"hook_event_name": "PostToolUse", "tool_name": "MultiEdit", "tool_input": {"file_path": str(bare)}}), env=self.env, capture_output=True, text=True)
        self.assertIn("additionalContext", r.stdout)

    def test_h4_installer_wires_jit_idempotently(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import install_v3
        settings = Path(self.tmp.name) / "settings.json"
        settings.write_text(json.dumps({"hooks": {"SubagentStart": [{"hooks": [{"type": "command", "command": "other"}]}]}}))
        kw = dict(settings=settings, bin_dir=Path(self.tmp.name) / "bin")
        install_v3.install_global(self.vault, self.state, **kw)
        first = settings.read_text()
        self.assertEqual(install_v3.install_global(self.vault, self.state, **kw)["status"], "unchanged")
        hooks = json.loads(first)["hooks"]
        jit = [(e, g.get("matcher")) for e, gs in hooks.items() for g in gs for h in g["hooks"] if "memrove_v3_jit.py" in h["command"]]
        self.assertEqual(sorted(jit, key=str), sorted([("SubagentStart", None), ("PreToolUse", "AskUserQuestion"), ("PostToolUse", "Write|Edit|MultiEdit")], key=str))
        self.assertEqual(len(hooks["SubagentStart"]), 2)  # the foreign entry stays
        install_v3.install_global(self.vault, self.state, uninstall=True, **kw)
        self.assertNotIn("memrove", settings.read_text())

    def test_h5_kill_switch_everywhere_and_doctor_share(self):
        import memrove_v3_heads as H
        import memrove_v3_hook as hook
        today = date.today().isoformat()
        (self.docs / f"{today}-a.md").write_text("---\ntopic: a\nstatus: open\ndecision: d\nnext: n\n---\n")
        (self.docs / f"{today}-b.md").write_text("# b\n")
        self.assertIn("front matter on 1/2 (50%)", self.cli("doctor").stdout)
        self.remember("Anything to make the project known", "known-topic")
        env = dict(self.env, MEMROVE_HEADS="0")
        with mock.patch.dict(os.environ, {"MEMROVE_HEADS": "0"}):
            self.assertEqual(H.heads(self.state, "myproj"), ([], {}))
            self.assertEqual((H.render(self.state, "myproj"), H.topics(self.state, "myproj"), H.match(self.state, "a")), ([], [], []))
            self.assertNotIn("heads", hook.session_notice(self.state, {"source": "startup", "cwd": str(self.proj)}))
        on = subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), "project", "myproj"], env=self.env, capture_output=True, text=True).stdout
        off = subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), "project", "myproj"], env=env, capture_output=True, text=True).stdout
        self.assertIn("heads (newest", on)
        self.assertNotIn("heads (newest", off)


if __name__ == "__main__":
    unittest.main()
