#!/usr/bin/env python3
"""Standing rules: capture of the owner's rule forms, the pushed block (scope, contradiction, cap, kill switch), SubagentStart, search of q-/p- records."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "template/.claude/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
from memrove_v3_capture import capture  # noqa: E402
import memrove_v3_rules as rules  # noqa: E402

TRACED = [
    "widgetco ile ilgili hiçbir şey olmasın", "widgetco diye bir şey yok artık her şey acme olmalı",
    "widgetco ile işimiz yok artık", "widgetco ile hiçbir bağımız olmamalı", "widgetco ile ilgili hiçbir şey kalmamalı demiştim",
    "raporlar ingilizce olmalı", "her şey ingilizce olmalı", "ingilizce olmalı her şey haberin olsun.",
    "her şey ingilizce olmalı, sadece benle konuşurken fransızca olsun", "silebilirsin, bir daha benden izin isteme",
    "arayüz acme temasına uygun olsun", "bu tema acme ile uyumlu olmalı",
]


def judged(sentence, scope="global", project="", standing="yes", state="assessed"):
    """The restate step's judgement as stored on a record (assessments.restate)."""
    return {"restate": {"answers": {"standing_rule": standing, "scope": scope, "project": project, "sentence": sentence}, "state": state,
                        "provider": "haiku", "rubric_version": 1, "revision": 1}}


def rec(i, kind, text, day, project="", **kw):
    r = dict(id=i, kind=kind, text=text + "\n", origin_type="user_explicit", verified_by="user_quote", status="active", project=project,
             recorded_at=day + "T10:00:00+00:00", valid_from=day, visibility="internal", source=f"notes/captured/{i}.md")
    r.update(kw)
    return r


def store(records):
    d = tempfile.mkdtemp(prefix="v3-rules-")
    db = sqlite3.connect(d + "/memory.sqlite3")
    db.execute("CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.executemany("INSERT INTO records VALUES (?,?)", [(r["id"], json.dumps(r)) for r in records])
    db.commit()
    db.close()
    return d


class RulesTest(unittest.TestCase):
    def test_traced_statements_are_captured(self):
        self.assertEqual([t for t in TRACED if not capture(t)], [])

    def test_ordinary_task_prompts_are_not(self):
        for t in ("testleri çalıştır ve sonucu söyle", "bunu yapabilirsin", "şu butonu mavi yap", "paketi kurabilirsin istersen"):
            self.assertEqual(capture(t), [], t)

    def test_block_scope_contradiction_cap(self):
        d = store([
            rec("q-1", "preference", "clay orange #D97757 accent on every surface", "2026-10-07", restate_dup_of="q-2",   # the merge step marks a reversed rule
                assessments=judged("The owner wants a clay orange accent on every surface.")),
            rec("q-2", "rejected", "no orange accent on any surface", "2026-10-10", assessments=judged("The owner wants no orange accent on any surface.")),
            rec("q-3", "rejected", "widgetco ile hiçbir bağımız olmamalı artık", "2026-10-12", project="acme",
                assessments=judged("Acme has no tie to widgetco at all.", "project", "acme")),
            rec("q-4", "preference", "her zaman fransızca cevap ver", "2026-10-06", project="alpha",
                assessments=judged("In alpha, Claude replies to the owner in French.", "project", "alpha")),
            rec("q-5", "preference", "ingilizce olmalı her şey", "2026-10-13", status="dropped_llm",
                assessments=judged("Everything Claude writes except replies to the owner is English.")),   # J1 dropped it on a wrong referent; the wider window says rule
            rec("q-6", "preference", "kısa cevap olsun", "2026-10-13", status="dropped_llm", assessments=judged("", standing="no")),
            rec("p-7", "decision", "agent inferred rule", "2026-10-13", origin_type="agent_inference", verified_by="llm",
                assessments=judged("Claude infers a rule on its own.")),
        ])
        b = rules.block(d, "alpha")
        self.assertIn("no orange", b)
        self.assertNotIn("clay orange", b)           # the older contradicting rule is hidden
        self.assertIn("replies to the owner in French", b)   # the current project's rule
        self.assertNotIn("widgetco", b)                  # another project's rule
        self.assertIn("Everything Claude writes", b)  # the restate judgement, not the J1 drop, decides
        self.assertNotIn("kısa cevap", b)            # not a standing rule
        self.assertNotIn("infers a rule", b)         # inferred records are not owner rules
        self.assertNotIn("French", rules.block(d, "alpha", global_only=True))
        many = store([rec(f"q-{i}", "preference", "x", "2026-10-%02d" % (i % 28 + 1),
                          assessments=judged(f"The owner wants rule number {i} to hold with {'x' * 80} everywhere.")) for i in range(60)])
        self.assertLessEqual(len(rules.block(many).encode()), rules.CAP)

    def test_kill_switch(self):
        d = store([rec("q-1", "preference", "ingilizce olmalı her şey", "2026-10-13", assessments=judged("Everything Claude writes is English."))])
        self.assertIn("Everything Claude writes", rules.block(d))   # on without the switch, so the empty block below is the switch
        os.environ["MEMROVE_RULES"] = "0"
        try:
            self.assertEqual(rules.block(d), "")
        finally:
            del os.environ["MEMROVE_RULES"]

    def test_subagent_start_carries_global_rules(self):
        d = store([rec("q-1", "preference", "ingilizce olmalı her şey", "2026-10-13", assessments=judged("Everything Claude writes is English.")),
                   rec("q-2", "rejected", "widgetco yok", "2026-10-13", project="acme", assessments=judged("Acme has no widgetco.", "project", "acme"))])
        r = subprocess.run([sys.executable, str(SCRIPTS / "memrove_v3_jit.py")], input=json.dumps({"hook_event_name": "SubagentStart", "agent_type": "coder"}),
                           text=True, capture_output=True, env={**os.environ, "MEMROVE_STATE": d}, timeout=10)
        ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Everything Claude writes is English.", ctx)
        self.assertNotIn("widgetco", ctx)                # global only

    def test_near_duplicate_rules_give_one_line(self):
        d = store([rec("q-1", "preference", "x", "2026-10-01", assessments=judged("The owner wants Claude to search open source for existing parts before building anything new.")),
                   rec("q-2", "preference", "x", "2026-10-02", assessments=judged("The owner wants Claude to search open source for existing parts before building anything new.")),
                   rec("q-3", "preference", "x", "2026-10-03", assessments=judged("The owner wants Claude to search open source before building.")),
                   rec("q-4", "preference", "x", "2026-10-04", assessments=judged("In scout, X, Instagram and TikTok scouts run daily.", "project", "scout")),
                   rec("q-5", "preference", "x", "2026-10-05", assessments=judged("In scout, X, Instagram and TikTok posts are never stored whole.", "project", "scout")),
                   rec("q-6", "rejected", "x", "2026-10-06", assessments=judged("Claude does not search open source for existing parts before building anything new."))])
        texts = [t[2] for t in rules.rules(d, "scout")]
        self.assertEqual(sum("search open source" in t and "does not" not in t for t in texts), 1, texts)
        self.assertEqual(sum(t.startswith("In scout") for t in texts), 2)       # different rules on one topic both stay
        self.assertTrue(any("does not search" in t for t in texts))              # a negation is a different rule
        self.assertIn("The owner wants Claude to search open source for existing parts before building anything new.", texts)   # the fuller line wins
        self.assertEqual([t[2] for t in rules.rules(store([rec("q-1", "preference", "x", "2026-10-01", assessments=judged("Frontend work stays on Claude Sonnet.")),
                                                           rec("q-2", "preference", "x", "2026-10-02", assessments=judged("Frontend work stays on Claude Opus."))]))].__len__(), 2)   # a one-word swap is a different rule

    def test_subagent_block_has_the_session_cap(self):
        d = store([rec(f"q-{i}", "preference", "x", "2026-10-%02d" % (i % 28 + 1), assessments=judged(f"The owner wants rule number {i} about topic{i} kept here."))
                   for i in range(30)])
        r = subprocess.run([sys.executable, str(SCRIPTS / "memrove_v3_jit.py")], input=json.dumps({"hook_event_name": "SubagentStart", "agent_type": "coder"}),
                           text=True, capture_output=True, env={**os.environ, "MEMROVE_STATE": d}, timeout=10)
        ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(sum(f"rule number {i} " in ctx for i in range(30)), 8)   # the subagent form is capped at 8 global lines
        self.assertNotIn("left out", ctx)

    def test_subagent_start_carries_no_project_rules(self):
        d = store([rec("q-1", "preference", "x", "2026-10-13", assessments=judged("Everything Claude writes is English.")),
                   rec("q-2", "rejected", "x", "2026-10-13", project="acme", assessments=judged("In Acme, there is no widgetco.", "project", "acme")),
                   rec("q-3", "rejected", "x", "2026-10-13", project="beta", assessments=judged("In Beta, there is no gadgetco.", "project", "beta"))])
        for name in ("acme", "beta"):
            cwd = Path(tempfile.mkdtemp(prefix="v3-sub-")) / name
            (cwd / ".git").mkdir(parents=True)
            r = subprocess.run([sys.executable, str(SCRIPTS / "memrove_v3_jit.py")], input=json.dumps({"hook_event_name": "SubagentStart", "agent_type": "coder", "cwd": str(cwd)}),
                               text=True, capture_output=True, env={**os.environ, "MEMROVE_STATE": d}, timeout=10)
            ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
            self.assertIn("Everything Claude writes is English.", ctx)
            self.assertNotIn("widgetco", ctx)   # project rules reach a subagent only when recalled as relevant to its brief
            self.assertNotIn("gadgetco", ctx)

    def test_only_restated_standing_rules_enter_the_block(self):
        d = store([
            rec("q-1", "preference", "açıklama yapma.", "2026-10-12"),                                       # never restated
            rec("q-2", "preference", "kod parçacıkları istemiyorum.", "2026-10-12", assessments=judged("", standing="no")),
            rec("q-3", "preference", "x", "2026-10-12", assessments=judged("Don't explain.")),               # no subject
            rec("q-4", "preference", "x", "2026-10-12", assessments=judged("Maybe the owner wants short answers in …")),   # cut
            rec("q-5", "preference", "x", "2026-10-12", assessments=judged("Sahibi her zaman fransızca ister.")),   # not English
            rec("q-6", "preference", "x", "2026-10-12", assessments=judged("The owner wants tests first.", state="not_assessed")),
            rec("q-7", "preference", "x", "2026-10-12", assessments=judged("In Acme, the sidebar stays closed.", "project", "")),   # project unnamed
            rec("q-8", "preference", "x", "2026-10-12", assessments=judged("The owner wants commits in English.")),
        ])
        b = rules.block(d, "alpha")
        self.assertEqual([ln for ln in b.splitlines()[1:]], ["- The owner wants commits in English."])
        self.assertEqual(rules.block(store([])), "")

    def test_a_correction_hides_the_old_rule_only_once_it_is_restated(self):
        old = "The assistant must write everything in English."
        new = "The assistant writes files in English but replies to the user in their language."
        kw = dict(supersedes=["q-old"], source="notes/observed/preference/new.md")
        pending = store([rec("q-old", "preference", "x", "2026-10-12", assessments=judged(old)),
                         rec("obs-new", "preference", "Everything is English, but replies are in the user's language.", "2026-10-13", **kw)])
        # the owner's correction is a rule at once, in its own words and with the old rule's scope; the old rule goes
        self.assertEqual(rules.block(pending, "demo").splitlines()[1:], ["- Everything is English, but replies are in the user's language."])
        cut = store([rec("q-old", "preference", "x", "2026-10-12", assessments=judged(old)),
                     rec("obs-new", "preference", "Türkçe yanıt ver", "2026-10-13", **kw)])
        self.assertEqual(rules.block(cut, "demo").splitlines()[1:], ["- " + old])   # no gap while an unusable correction waits
        done = store([rec("q-old", "preference", "x", "2026-10-12", assessments=judged(old)),
                      rec("obs-new", "preference", "x", "2026-10-13", assessments=judged(new), **kw)])
        self.assertEqual(rules.block(done, "demo").splitlines()[1:], ["- " + new])

    def test_a_correction_marked_as_duplicate_of_the_old_rule_still_replaces_it(self):
        d = store([
            rec("q-old", "preference", "x", "2026-10-12", assessments=judged("The assistant must write everything in English.")),
            rec("obs-new", "preference", "x", "2026-10-13", supersedes=["q-old"], restate_dup_of="q-old",
                assessments=judged("The assistant writes files in English but replies to the user in their language.")),
        ])
        self.assertEqual(rules.block(d, "demo").splitlines()[1:],
                         ["- The assistant writes files in English but replies to the user in their language."])

    def test_project_rule_only_in_its_project_and_never_cut(self):
        d = store([rec("q-1", "rejected", "sidebar istemiyorum", "2026-10-07", project=".claude",
                       assessments=judged("In Acme, the mode indicators do not sit in a sidebar.", "project", "Acme")),
                   rec("q-2", "preference", "x", "2026-10-08", assessments=judged("The owner wants every reply to them in French."))])
        self.assertIn("In Acme,", rules.block(d, "acme"))
        for project, g in (("alpha", False), ("unknown", False), ("acme", True)):
            b = rules.block(d, project, g)
            self.assertNotIn("Acme", b, (project, g))
            self.assertIn("French", b)
        long = store([rec(f"q-{i}", "preference", "x", f"2026-10-{i + 1:02d}", assessments=judged(f"The owner wants rule number {i} about topic{i} kept {'long ' * 14}here."))
                      for i in range(20)])
        lines = rules.block(long).splitlines()[1:-1]   # the last line is the left-out notice
        self.assertTrue(lines and all(ln.endswith(".") and "…" not in ln for ln in lines))   # whole sentences, the rest is left out

    def test_hand_set_scope_wins_over_the_judgement(self):
        d = store([rec("q-1", "preference", "x", "2026-10-07", assessments=judged("The owner wants every reply in French.")),
                   rec("q-2", "preference", "x", "2026-10-08", assessments=judged("In Acme, the sidebar stays closed.", "project", "Acme"), rule_scope="global"),
                   rec("q-3", "preference", "x", "2026-10-09", assessments=judged("The owner wants tabs over spaces.", "global"), rule_scope="Beta")])
        self.assertEqual({t[2] for t in rules.rules(d, "alpha")}, {"The owner wants every reply in French.", "In Acme, the sidebar stays closed."})   # (a)(b)
        self.assertIn("The owner wants tabs over spaces.", [t[2] for t in rules.rules(d, "beta")])
        self.assertNotIn("tabs", rules.block(d, "alpha"))

    def test_cap_ends_with_the_left_out_count_and_no_partial_rule(self):
        d = store([rec(f"q-{i}", "preference", "x", f"2026-10-{i + 1:02d}", assessments=judged(f"The owner wants rule number {i} about topic{i} kept {'long ' * 14}here."))
                   for i in range(80)])
        b = rules.block(d, "alpha")
        lines = b.splitlines()[1:]
        self.assertLessEqual(len(b.encode()), rules.CAP)
        n = len(lines) - 1
        self.assertTrue(lines[-1].startswith(f"- ({80 - n} more rules left out") and "memrove rules --left-out --project 'alpha'" in lines[-1], lines[-1])
        self.assertTrue(all(ln.endswith(".") for ln in lines[:-1]))
        self.assertEqual(len(rules.left_out(d, "alpha")), 80 - n)
        self.assertNotIn("left out", rules.block(d, "alpha", cap=20000))

    def test_capture_keeps_a_window_for_the_restate_step(self):
        from memrove_v3_capture import context_window
        w = context_window(["a", "b", "c", "d"], "the assistant said x")
        self.assertEqual(w, "owner: b | owner: c | owner: d | assistant: the assistant said x")
        self.assertEqual(context_window(["my key sk-ant-api03-" + "a" * 40], ""), "")

    def test_restate_step_stores_the_sentence_and_undo_removes_it(self):
        import memrove_v3_llm as llm
        import memrove_v3_restate as rs
        from memrove_v3_sync import SyncEngine
        tmp = Path(tempfile.mkdtemp(prefix="v3-restate-"))
        (tmp / "vault").mkdir(), (tmp / "state").mkdir()
        engine = SyncEngine(tmp / "vault", tmp / "state")
        engine.note_create("notes/captured/q-9.md", "ingilizce olmalı her şey", dict(
            id="q-9", kind="preference", project=".claude", session="s", verified_by="user_quote", status="active", key="k", polarity="pos",
            origin_type="user_explicit", ctx="owner: acme ui | assistant: UI strings are French"))
        seen = []

        def runner(system, payload, vault=None):
            seen.append(json.loads(payload))
            return json.dumps({"items": [{"id": "q-9", "standing_rule": "yes", "scope": "project", "project": "acme",
                                          "sentence": "In Acme, every UI string is English."}]}), 0.0, None
        saved = {k: os.environ.pop(k, None) for k in ("MEMROVE_NO_LLM", "MEMROVE_INVOKED_BY", "MEMROVE_EVALUATOR")}
        llm.RUNNER = runner
        try:
            self.assertEqual(rs.restate_rules(engine, tmp / "vault", tmp / "state"), 1)
            self.assertIn("UI strings are French", json.dumps(seen[0]))   # the window reached the model, and the known projects with it
            self.assertEqual(seen[0]["items"][0]["quote"], "ingilizce olmalı her şey")
            with engine.store._connect() as db:
                r = json.loads(db.execute("SELECT payload FROM records WHERE id='q-9'").fetchone()[0])
            self.assertEqual((r["status"], r["text"].strip()), ("active", "ingilizce olmalı her şey"))   # quote and status untouched
            self.assertEqual(r["assessments"]["restate"]["answers"]["sentence"], "In Acme, every UI string is English.")
            self.assertEqual(rs.restate_rules(engine, tmp / "vault", tmp / "state"), 0)   # once only
            log = tmp / "log.jsonl"
            log.write_text(json.dumps({"id": "q-9"}) + "\n")
            self.assertEqual(rs.undo(engine, log), 1)
            with engine.store._connect() as db:
                self.assertNotIn("restate", json.loads(db.execute("SELECT payload FROM records WHERE id='q-9'").fetchone()[0]).get("assessments", {}))
        finally:
            llm.RUNNER = llm._claude
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_same_rule_in_two_languages_shows_once(self):
        import memrove_v3_llm as llm
        import memrove_v3_restate as rs
        from memrove_v3_sync import SyncEngine
        tmp = Path(tempfile.mkdtemp(prefix="v3-merge-"))
        (tmp / "vault").mkdir(), (tmp / "state").mkdir()
        engine = SyncEngine(tmp / "vault", tmp / "state")
        for i, (text, sent, day) in enumerate((("her şey ingilizce olmalı", "Everything Claude writes is English.", "2026-10-06"),
                                              ("ingilizce olmalı her şey haberin olsun", "All files and commits are English; only chat with the owner is French.", "2026-10-13"),
                                              ("her zaman fransızca cevap ver", "Claude replies to the owner in French.", "2026-10-09"))):
            engine.note_create(f"notes/captured/q-{i}.md", text, dict(id=f"q-{i}", kind="preference", project="", session="s", verified_by="user_quote", status="active",
                                                                   key=f"k{i}", polarity="pos", origin_type="user_explicit", valid_from=day, assessments=judged(sent)))
        llm.RUNNER = lambda system, payload, vault=None: (json.dumps({"groups": [["q-0", "q-1"]]}), 0.0, None)
        saved = {k: os.environ.pop(k, None) for k in ("MEMROVE_NO_LLM", "MEMROVE_INVOKED_BY")}
        try:
            self.assertEqual(rs.merge_duplicates(engine, tmp / "vault", tmp / "state"), 1)
        finally:
            llm.RUNNER = llm._claude
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        lines = rules.block(str(tmp / "state")).splitlines()[1:]
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("only chat with the owner" in ln for ln in lines) and not any("Everything Claude writes" in ln for ln in lines))   # the newer of the pair stays

    def test_one_yes_is_not_enough(self):
        import memrove_v3_llm as llm
        import memrove_v3_restate as rs
        tmp = Path(tempfile.mkdtemp(prefix="v3-votes-"))
        answers = iter(["yes", "yes", "no", "yes", "no"])   # draw 1: a yes, b yes; draw 2: a no, b yes; draw 3 (a only, the split): no

        def runner(system, payload, vault=None):
            items = json.loads(payload)["items"]
            return json.dumps({"items": [{"id": i["id"], "standing_rule": next(answers), "scope": "global", "project": "", "sentence": "The owner wants tests first."}
                                         for i in items]}), 0.0, None
        llm.RUNNER = runner
        saved = {k: os.environ.pop(k, None) for k in ("MEMROVE_NO_LLM", "MEMROVE_INVOKED_BY", "MEMROVE_EVALUATOR")}
        try:
            out = rs.two_votes(tmp, [{"id": "a", "text": "x", "revision": 1, "egress": "external_ok"}, {"id": "b", "text": "y", "revision": 1, "egress": "external_ok"}], tmp)
        finally:
            llm.RUNNER = llm._claude
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        self.assertEqual([(o["state"], o["answers"]["standing_rule"]) for o in out], [("unclear", "unclear"), ("assessed", "yes")])

    def test_adopt_writes_what_a_copy_computed_and_undo_restores(self):
        import memrove_v3_llm as llm
        import memrove_v3_restate as rs
        from memrove_v3_sync import SyncEngine
        tmp = Path(tempfile.mkdtemp(prefix="v3-adopt-"))
        for n in ("vault", "state"):
            (tmp / n).mkdir()
        engine = SyncEngine(tmp / "vault", tmp / "state")
        engine.note_create("notes/captured/q-9.md", "ingilizce olmalı her şey", dict(id="q-9", kind="preference", project="", session="s", verified_by="user_quote",
                                                                                 status="active", key="k", polarity="pos", origin_type="user_explicit"))
        for n in ("c/vault", "c/state"):                                         # the copy: the same record in a store of its own
            (tmp / n).mkdir(parents=True)
        cengine = SyncEngine(tmp / "c" / "vault", tmp / "c" / "state")
        cengine.note_create("notes/captured/q-9.md", "ingilizce olmalı her şey", dict(id="q-9", kind="preference", project="", session="s", verified_by="user_quote",
                                                                                  status="active", key="k", polarity="pos", origin_type="user_explicit"))
        llm.RUNNER = lambda system, payload, vault=None: (json.dumps({"items": [{"id": "q-9", "standing_rule": "yes", "scope": "global", "project": "",
                                                                                 "sentence": "Everything Claude writes is English."}]}), 0.0, None)
        saved = {k: os.environ.pop(k, None) for k in ("MEMROVE_NO_LLM", "MEMROVE_INVOKED_BY", "MEMROVE_EVALUATOR")}
        try:
            self.assertEqual(rs.restate_rules(cengine, tmp / "c" / "vault", tmp / "c" / "state"), 1)
        finally:
            llm.RUNNER = llm._claude
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        self.assertEqual(rules.block(str(tmp / "state")), "")                                  # the live store is untouched until adopt
        log = tmp / "log.jsonl"
        self.assertEqual(rs.adopt(engine, tmp / "state", tmp / "c" / "state", log), {"written": 1, "skipped": 0})
        self.assertIn("Everything Claude writes is English.", rules.block(str(tmp / "state")))
        self.assertEqual(rs.adopt(engine, tmp / "state", tmp / "c" / "state", log)["written"], 0)   # once only
        self.assertEqual(rs.undo(engine, log), 1)
        self.assertEqual(rules.block(str(tmp / "state")), "")

    def test_search_returns_captured_and_inferred(self):
        import memrove_v3_chatsearch as cs
        root = Path(tempfile.mkdtemp(prefix="v3-rules-s-"))
        (root / "state").mkdir()
        db = sqlite3.connect(root / "state" / "memory.sqlite3")
        db.execute("CREATE TABLE records(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for r in (rec("q-1", "preference", "zebrafish naming rule", "2026-10-01"),
                  rec("p-2", "decision", "zebrafish decision inferred", "2026-10-01", source="notes/inferred/p-2.md", origin_type="agent_inference")):
            db.execute("INSERT INTO records VALUES (?,?)", (r["id"], json.dumps(r)))
        db.commit()
        db.close()
        orig = cs.chatlog.root
        cs.chatlog.root = lambda state: root / "state" / "chatlog"
        try:
            ids = {h[1][0] for h in cs.records(str(root / "state"), "zebrafish")}
        finally:
            cs.chatlog.root = orig
        self.assertEqual(ids, {"rec:q-1", "rec:p-2"})


class LoadedClaudeMdTest(unittest.TestCase):
    def test_only_an_exact_normalized_match_in_claude_md_is_dropped_and_logged(self):
        from unittest.mock import patch
        home = Path(tempfile.mkdtemp(prefix="v3-home-"))
        (home / ".claude").mkdir()
        # same words, other case, punctuation and line breaks: the normalized text matches
        (home / ".claude" / "CLAUDE.md").write_text("# Rules\n- THE d1 decision model runs\n  always-on; never loaded or unloaded!\n")
        d = store([rec("q-1", "preference", "x", "2026-10-01", assessments=judged("The d1 decision model runs always-on, never loaded or unloaded.")),
                   # near match (one word differs): kept
                   rec("q-2", "preference", "x", "2026-10-02", assessments=judged("The d1 decision model runs always-on, never loaded or unloaded today.")),
                   rec("q-3", "preference", "x", "2026-10-03", assessments=judged("The exporter must write one file per customer report."))])
        with patch.object(Path, "home", return_value=home):
            b = rules.block(d, cwd=str(home))
            self.assertNotIn("unloaded.\n", b.replace("today.", "X"))   # q-1 gone
            self.assertIn("unloaded today.", b)                          # q-2 kept
            self.assertIn("exporter", b)
            self.assertIn("d1 decision", rules.block(d))                 # no cwd: nothing is filtered
        log = [json.loads(ln) for ln in (Path(d) / "rules-dropped.jsonl").read_text().splitlines()]
        self.assertEqual([x["ids"] for x in log], [["q-1"]])

    def test_words_stems(self):
        self.assertEqual(rules._words("needs add ids"), {"need", "add", "ids"})


if __name__ == "__main__":
    unittest.main()
