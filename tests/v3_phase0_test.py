#!/usr/bin/env python3
"""Lifecycle phase 0: decision-log resolution over the whole log, supersedes closing the old record, exposure log, correction detector."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True


class DecideLogWhole(unittest.TestCase):
    def test_resolve_sees_a_decision_older_than_the_tail_window(self):
        tmp = Path(tempfile.mkdtemp())
        os.environ["DECIDE_HOME"] = str(tmp)
        self.addCleanup(os.environ.pop, "DECIDE_HOME")
        import memrove_v3_decide_log as dl
        rid = dl.log("memrove.recall", "q", options=[{"id": "rec-a"}], answer=["rec-a"], state={"session": "S1"})
        filler = json.dumps({"use": "memrove.capture_kind", "id": "x", "question": "y" * 4000, "answer": "note"}) + "\n"
        with dl.path().open("a") as f:
            f.write(filler * (4_000_000 // len(filler) + 50))
        tr = tmp / "t.jsonl"
        tr.write_text(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "see rec-a"}]}}) + "\n")
        self.assertEqual(dl.resolve_session("S1", str(tr)), 1)
        got = [r for r in dl._tail("memrove.recall") if r["id"] == rid][0]
        self.assertEqual(got["outcome"]["status"], "cited")


class SupersedesCloses(unittest.TestCase):
    def test_remember_supersedes_sets_valid_to_and_superseded_by(self):
        tmp = Path(tempfile.mkdtemp()).resolve()
        vault, state, proj = tmp / "Vault", tmp / "state", tmp / "w" / "p"
        vault.mkdir()
        (proj / ".git").mkdir(parents=True)
        env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}, "MEMROVE_READER": "local",
               "HOME": str(tmp), "CLAUDE_CODE_SESSION_ID": "s1", "MEMROVE_NO_LLM": "1"}

        def cli(*a):
            return subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(vault), "--state", str(state), *a],
                                  env=env, capture_output=True, text=True, cwd=proj)
        old = cli("remember", "Quokka cache is enabled", "--kind", "decision", "--cwd", str(proj)).stdout.strip()
        new = cli("remember", "Quokka cache is disabled now", "--kind", "decision", "--cwd", str(proj), "--supersedes", old).stdout.strip()
        row = sqlite3.connect(state / "memory.sqlite3").execute("SELECT payload FROM records WHERE id=?", (old,)).fetchone()
        rec = json.loads(row[0])
        self.assertTrue(rec.get("valid_to"))
        self.assertEqual(rec.get("superseded_by"), [new])


    def test_remember_supersedes_identical_text_does_not_close_the_record_on_itself(self):
        tmp = Path(tempfile.mkdtemp()).resolve()
        vault, state, proj = tmp / "Vault", tmp / "state", tmp / "w" / "p"
        vault.mkdir()
        (proj / ".git").mkdir(parents=True)
        env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}, "MEMROVE_READER": "local",
               "HOME": str(tmp), "CLAUDE_CODE_SESSION_ID": "s1", "MEMROVE_NO_LLM": "1"}
        cli = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(vault), "--state", str(state), *a],
                                        env=env, capture_output=True, text=True, cwd=proj)
        old = cli("remember", "Quokka cache is enabled", "--kind", "decision", "--cwd", str(proj)).stdout.strip()
        cli("remember", "Quokka cache is enabled", "--kind", "decision", "--cwd", str(proj), "--supersedes", old)
        rec = json.loads(sqlite3.connect(state / "memory.sqlite3").execute("SELECT payload FROM records WHERE id=?", (old,)).fetchone()[0])
        self.assertFalse(rec.get("valid_to"))
        self.assertFalse(rec.get("superseded_by"))


class Exposure(unittest.TestCase):
    def test_cli_show_and_search_log_ids_and_a_query_hash_never_the_query(self):
        tmp = Path(tempfile.mkdtemp()).resolve()
        vault, state, proj = tmp / "Vault", tmp / "state", tmp / "w" / "p"
        vault.mkdir()
        (proj / ".git").mkdir(parents=True)
        env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}, "MEMROVE_READER": "local",
               "HOME": str(tmp), "CLAUDE_CODE_SESSION_ID": "sess-1", "MEMROVE_NO_LLM": "1"}

        def cli(*a):
            return subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(vault), "--state", str(state), *a],
                                  env=env, capture_output=True, text=True, cwd=proj)
        rid = cli("remember", "Quokka cache is enabled", "--kind", "decision", "--cwd", str(proj)).stdout.strip()
        cli("show", rid)
        cli("search", "quokka zebra", "--records")
        import memrove_v3_exposure as ex
        rows = sqlite3.connect(state / "memory.sqlite3").execute("SELECT session, path, item_id, slot FROM exposure").fetchall()
        self.assertIn((ex.key("sess-1"), "show", rid, None), rows)
        found = [r for r in rows if r[1] == "search"]
        self.assertTrue(found and all(r[3] == ex.qhash("quokka zebra") for r in found))
        self.assertNotIn("zebra", json.dumps(rows))

    def _state(self):
        tmp = Path(tempfile.mkdtemp()).resolve()
        vault, state, proj = tmp / "Vault", tmp / "state", tmp / "w" / "p"
        vault.mkdir()
        (proj / ".git").mkdir(parents=True)
        env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}, "MEMROVE_READER": "local",
               "HOME": str(tmp), "CLAUDE_CODE_SESSION_ID": "sess-2", "MEMROVE_NO_LLM": "1"}
        cli = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(vault), "--state", str(state), *a],
                                        env=env, capture_output=True, text=True, cwd=proj)
        rid = cli("remember", "Quokka cache is enabled", "--kind", "decision", "--cwd", str(proj)).stdout.strip()
        db = sqlite3.connect(state / "memory.sqlite3")   # make it a standing global rule the way the restate step would
        rec = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])
        rec.update(origin_type="user_explicit", assessments={"restate": {"state": "assessed", "answers": {
            "standing_rule": "yes", "scope": "global", "sentence": "The quokka cache must stay enabled for every build."}}})
        db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(rec), rid))
        db.commit()
        db.close()
        return tmp, state, proj, env, cli, rid

    def rows(self, state, path):
        return [r[0] for r in sqlite3.connect(state / "memory.sqlite3").execute("SELECT item_id FROM exposure WHERE path=?", (path,))]

    def test_sessionstart_hook_and_subagent_jit_log_the_rule_ids_they_pushed(self):
        tmp, state, proj, env, cli, rid = self._state()
        hook = lambda *a, payload: subprocess.run([sys.executable, str(ROOT / "template/.claude/scripts" / a[0]), "--state", str(state), *a[1:]],
                                                  input=json.dumps(payload), env={**env, "MEMROVE_STATE": str(state)}, capture_output=True, text=True, cwd=proj)
        out = hook("memrove_v3_hook.py", "--vault", str(tmp / "Vault"), "--harness", "claude",
                   payload={"hook_event_name": "SessionStart", "session_id": "sess-2", "cwd": str(proj), "source": "startup"})
        self.assertIn("quokka cache", out.stdout.lower())
        self.assertEqual(self.rows(state, "rules"), [rid])
        out = hook("memrove_v3_jit.py", payload={"hook_event_name": "SubagentStart", "session_id": "sess-2", "cwd": str(proj), "agent_type": "general-purpose"})
        self.assertIn("quokka cache", out.stdout.lower())
        self.assertIn(rid, self.rows(state, "jit"))

    def test_cli_zoom_logs_the_record_id(self):
        tmp, state, proj, env, cli, rid = self._state()
        import memrove_v3_chatlog as chatlog
        import memrove_v3_exposure as ex
        chatlog.append(state, [{"kind": "user", "text": "pelican chat about sqlite", "date": "2026-11-01", "src": "claude", "proj": "P"}], today="2026-11-01")
        self.assertEqual(cli("chatlog", "run").returncode, 0)  # zoom reads the chat tree, which `chatlog run` builds
        out = cli("zoom", "0", "1")
        self.assertIn("pelican", out.stdout, out.stderr)
        got = sqlite3.connect(state / "memory.sqlite3").execute("SELECT session, item_id, slot FROM exposure WHERE path='zoom'").fetchall()
        self.assertEqual(got, [(ex.key("sess-2"), "0", "n=1")])

    def test_log_dedupes_ids_and_needs_the_db(self):
        import memrove_v3_exposure as ex
        tmp = Path(tempfile.mkdtemp())
        self.assertEqual(ex.log(tmp, "jit", ["a"], "x", None), 0)  # no DB yet: nothing created, nothing raised
        sqlite3.connect(tmp / "memory.sqlite3").close()
        self.assertEqual(ex.log(tmp, "jit", ["a", "a", "b"], "recall", None), 2)
        self.assertEqual(sqlite3.connect(tmp / "memory.sqlite3").execute("SELECT session, item_id FROM exposure ORDER BY item_id").fetchall(), [(None, "a"), (None, "b")])


class Detector(unittest.TestCase):
    TURNS = [{"i": 1, "kind": "user", "text": "use sqlite for the store"}, {"i": 2, "kind": "agent", "text": "ok, postgres then"},
             {"i": 3, "kind": "user", "text": "I told you, sqlite"}, {"i": 4, "kind": "user", "text": "hayır, o değil"}]

    def test_labels_are_validated_and_anchors_found(self):
        import memrove_v3_incidents as inc
        reply = {"labels": [{"id": 3, "kind": "correction-of-memory", "corrects": 1}, {"id": 4, "kind": "correction-other", "corrects": 99},
                            {"id": 2, "kind": "correction-other", "corrects": None}, {"id": 1, "kind": "bogus", "corrects": None}]}
        labels, failed = inc.detect(self.TURNS, lambda system, payload: reply)
        self.assertEqual((labels, failed), ([(3, "correction-of-memory", 1), (4, "correction-other", None)], 0))  # agent id and bad kind dropped, unknown target -> None
        self.assertEqual(inc.anchors(self.TURNS), [3, 4])
        self.assertEqual(inc.detect(self.TURNS, lambda s, p: None), ([], 1))
        self.assertEqual(inc.detect(self.TURNS, lambda s, p: reply, after=3)[0], [(4, "correction-other", None)])  # only turns after the done marker are labelled

    def test_metric_needs_a_past_reference_and_an_owner_typed_turn(self):
        import memrove_v3_incidents as inc
        T = lambda i, text, kind="user": {"i": i, "kind": kind, "text": text}
        turns = [T(1, "Bunu sana söylemiştim, sqlite kullan"), T(2, "git log --oneline"), T(3, "Here is a note offered by a side agent: we had decided x"),
                 T(4, "make it blue"), T(5, "We decided on sqlite."), T(6, "We decided on sqlite."), T(7, "```\nwe had it\n```")]
        labels = [(i, "correction-of-memory", None) for i in range(1, 8)]
        self.assertEqual(inc.metric_ids(turns, labels), [1, 5])  # 2 shell, 3 side agent, 4 no past form, 6 repeat of 5, 7 code block only
        self.assertEqual(inc.metric_ids(turns, [(1, "correction-other", None)]), [])

    def test_save_is_idempotent_and_records_the_marker(self):
        import memrove_v3_incidents as inc
        db = sqlite3.connect(":memory:")
        for _ in range(2):
            inc.save(db, "s", [(3, "correction-of-memory", 1)], 4)
        self.assertEqual(db.execute("SELECT count(*) FROM incident").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT last_msg_id FROM incident_done").fetchone()[0], 4)


if __name__ == "__main__":
    unittest.main()
