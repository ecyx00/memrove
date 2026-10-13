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
        self.assertLessEqual(len(rules.block(many).encode()), 1500)

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

    def test_a_correction_hides_the_old_rule_before_it_is_restated(self):
        d = store([
            rec("q-old", "preference", "x", "2026-10-12", assessments=judged("The assistant must write everything in English.")),
            rec("obs-new", "preference", "Everything is English, but replies to the user are in their language.", "2026-10-13", supersedes=["q-old"],
                source="notes/observed/preference/new.md"),
        ])
        self.assertNotIn("The assistant must write everything in English.", rules.block(d, "demo"))

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
        lines = rules.block(long).splitlines()[1:]
        self.assertTrue(lines and all(ln.endswith(".") and "…" not in ln for ln in lines))   # whole sentences, the rest is left out

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
            id="q-9", kind="preference", project=".claude", session="s", verified_by="user_quote", status="dropped_llm", key="k", polarity="pos",
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
            self.assertEqual((r["status"], r["text"].strip()), ("dropped_llm", "ingilizce olmalı her şey"))   # quote and status untouched
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
        answers = iter(["yes", "yes", "no", "yes"])   # draw 1: a yes, b yes; draw 2: a no, b yes

        def runner(system, payload, vault=None):
            items = json.loads(payload)["items"]
            return json.dumps({"items": [{"id": i["id"], "standing_rule": next(answers), "scope": "global", "project": "", "sentence": "The owner wants tests first."}
                                         for i in items]}), 0.0, None
        llm.RUNNER = runner
        saved = {k: os.environ.pop(k, None) for k in ("MEMROVE_NO_LLM", "MEMROVE_INVOKED_BY", "MEMROVE_EVALUATOR")}
        try:
            out = rs.two_votes(tmp, [{"id": "a", "text": "x", "revision": 1}, {"id": "b", "text": "y", "revision": 1}], tmp)
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


if __name__ == "__main__":
    unittest.main()
