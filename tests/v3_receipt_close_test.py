#!/usr/bin/env python3
"""Receipt and close: a save books the neighbours it shows, --replaces / --rests-on answer in the call, settle resolves a pair,
card prints what is open. Real CLI on a temporary store; no model."""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A_TEXT = "Use zebrafish queue for ingestion pipeline"
B_TEXT = "Switch ingestion pipeline to the zebrafish queue v2"


class ReceiptClose(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp = Path(tmp.name).resolve()
        self.vault, self.state, self.proj = tmp / "Vault", tmp / "state", tmp / "work" / "myproj"
        self.vault.mkdir()
        (self.proj / ".git").mkdir(parents=True)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}
        self.env.update(HOME=str(tmp), CLAUDE_CODE_SESSION_ID="s1", MEMROVE_NO_LLM="1")

    def cli(self, cmd, *a, ok=True):
        extra = ["--cwd", str(self.proj)] if cmd in ("remember", "card") else []
        p = subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), cmd, *a, *extra],
                           env=self.env, capture_output=True, text=True)
        if ok:
            self.assertEqual(p.returncode, 0, p.stderr)
        return p

    def save(self, text, *a):
        p = self.cli("remember", text, "--kind", "decision", *a)
        return p.stdout.strip(), p.stderr

    def sql(self, q, *args):
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            return db.execute(q, args).fetchall()

    def status(self, rid):
        return json.loads(self.sql("SELECT payload FROM records WHERE id=?", rid)[0][0]).get("status")

    def pairs(self):
        return self.sql("SELECT new_id, nb_id, state FROM suspense")

    def test_unanswered_neighbour_is_shown_and_booked(self):
        a, _ = self.save(A_TEXT)
        b, err = self.save(B_TEXT)
        self.assertIn(f"may replace: rec:{a}", err)
        self.assertIn(f"memrove settle {b} {a} --replaces | --rests-on --why-link", err)
        self.assertEqual(self.pairs(), [(b, a, "open")])
        self.assertEqual(self.status(a), "active")   # nothing is changed by an unanswered neighbour

    def test_replaces_sets_status_and_books_nothing_for_it(self):
        a, _ = self.save(A_TEXT)
        b, err = self.save(B_TEXT, "--replaces", a)
        self.assertEqual(self.status(a), "superseded")
        self.assertEqual(json.loads(self.sql("SELECT payload FROM records WHERE id=?", b)[0][0])["supersedes"], [a])
        self.assertEqual(self.pairs(), [(b, a, "replaces")])   # booked before the link, cleared (settled) on success: never open
        self.assertNotIn(f"rec:{a}", err)

    def test_rests_on_is_an_edge_with_its_reason_readable_from_both_ends(self):
        a, _ = self.save(A_TEXT)
        b, err = self.save(B_TEXT, "--rests-on", a, "--why-link", "keeps the queue choice")
        self.assertEqual(self.pairs(), [(b, a, "rests_on")])
        self.assertEqual(self.status(a), "active")
        for rid, direction in ((b, "out"), (a, "in")):
            edges = json.loads(self.cli("show", rid).stdout)["edges"]
            self.assertEqual([(e["type"], e["dir"], e["why"]) for e in edges], [("rests_on", direction, "keeps the queue choice")])
        self.assertEqual(self.cli("remember", "x y z", "--kind", "decision", "--rests-on", a, ok=False).returncode, 1)   # no reason, nothing saved
        self.assertEqual(self.sql("SELECT count(*) FROM records")[0][0], 2)

    def test_settle_four_forms(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        self.cli("settle", b, a, "--replaces")
        self.assertEqual((self.status(a), self.pairs()), ("superseded", [(b, a, "replaces")]))
        self.cli("settle", b, a, "--undo")
        self.assertEqual((self.status(a), self.pairs()), ("active", [(b, a, "open")]))
        self.assertEqual(json.loads(self.sql("SELECT payload FROM records WHERE id=?", b)[0][0])["supersedes"], [])
        self.cli("settle", b, a, "--rests-on", "--why-link", "builds on it")
        self.assertEqual(self.sql("SELECT src,dst,type,why FROM graph_edges"), [(b, a, "rests_on", "builds on it")])
        self.cli("settle", b, a, "--undo")
        self.assertEqual(self.sql("SELECT count(*) FROM graph_edges")[0][0], 0)
        self.assertNotEqual(self.cli("settle", b, a, "--undo", ok=False).returncode, 0)   # nothing left to take back
        self.cli("settle", b, a, "--unrelated")
        self.assertEqual(self.pairs(), [(b, a, "unrelated")])
        self.assertEqual(self.status(a), "active")

    def test_unrelated_is_per_pair_only(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        self.cli("settle", b, a, "--unrelated")
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_suspense as suspense
        from memrove_v3_sync import SyncEngine
        lines = suspense.offer(SyncEngine(self.vault, self.state), self.state, "myproj", B_TEXT, b, "s1")   # the same pair again
        self.assertEqual((lines, [x for x in self.pairs() if x[:2] == (b, a)]), ([], [(b, a, "unrelated")]))
        c, err = self.save("Move the ingestion pipeline onto zebrafish queue v3")   # a different record may book the neighbour again
        self.assertIn(f"rec:{a}", err)
        self.assertIn((c, a, "open"), self.pairs())

    def test_status_guard_and_undo_only_restore_what_the_link_set(self):
        a, _ = self.save(A_TEXT)
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','retired') WHERE id=?", a)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.assertEqual(self.status(a), "retired")   # a replace never overwrites another status
        self.cli("settle", b, a, "--undo")
        self.assertEqual(self.status(a), "retired")
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','active') WHERE id=?", a)
        self.cli("settle", b, a, "--replaces")
        c, _ = self.save("Zebrafish queue ingestion pipeline moves to v3 now", "--replaces", a)
        self.cli("settle", b, a, "--undo")
        self.assertEqual(self.status(a), "superseded")   # c still replaces it
        self.cli("settle", c, a, "--undo")
        self.assertEqual(self.status(a), "active")

    def test_failed_link_leaves_an_open_pair(self):
        a, _ = self.save(A_TEXT)
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            db.executescript("CREATE TABLE IF NOT EXISTS graph_edges(src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, why TEXT NOT NULL, prov TEXT NOT NULL, ts TEXT NOT NULL, "
                                 "PRIMARY KEY(src,dst,type)); CREATE TRIGGER boom BEFORE INSERT ON graph_edges WHEN NEW.type='rests_on' BEGIN SELECT RAISE(ABORT,'boom'); END")
        b, err = self.save(B_TEXT, "--rests-on", a, "--why-link", "builds on it")
        self.assertIn("link not written", err)
        self.assertEqual(self.pairs(), [(b, a, "open")])

    def test_rerun_of_an_existing_record_still_writes_the_link(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        again, err = self.save(B_TEXT, "--replaces", a)
        self.assertEqual(again, b)
        self.assertIn("(exists)", err)
        self.assertEqual(self.status(a), "superseded")
        self.assertEqual(json.loads(self.sql("SELECT payload FROM records WHERE id=?", b)[0][0])["supersedes"], [a])
        self.save(B_TEXT, "--rests-on", a, "--why-link", "still builds on it")
        self.assertEqual(self.sql("SELECT src,dst,type FROM graph_edges"), [(b, a, "rests_on")])

    def test_settle_starts_with_begin_immediate(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_suspense as suspense
        from memrove_v3_sync import SyncEngine
        engine, seen = SyncEngine(self.vault, self.state), []
        connect = engine.store._connect
        def traced():
            cm = connect()
            db = cm.__enter__()
            db.set_trace_callback(seen.append)
            class Wrap:
                def __enter__(self_): return db
                def __exit__(self_, *e): return cm.__exit__(*e)
            return Wrap()
        engine.store._connect = traced
        suspense.settle(engine, b, a, "unrelated")
        self.assertEqual(seen[0], "BEGIN IMMEDIATE")

    def stored(self, rid):
        r = json.loads(self.sql("SELECT payload FROM records WHERE id=?", rid)[0][0])
        return r.get("status"), r.get("valid_to"), r.get("superseded_by")

    def recalled(self):
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        from memrove_v3_recall import _load
        from memrove_v3_sync import SyncEngine
        return {rid for _, _, rid in _load(SyncEngine(self.vault, self.state).store)}

    def test_undo_brings_the_replaced_record_back_to_recall(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.assertNotIn(a, self.recalled())
        self.assertEqual(self.stored(a)[2], [b])
        self.cli("settle", b, a, "--undo")
        self.assertEqual(self.stored(a), ("active", None, []))
        self.assertIn(a, self.recalled())
        self.assertIn(a, self.cli("search", "zebrafish queue ingestion").stdout)

    def test_settle_replaces_leaves_the_same_stored_state_as_remember_replaces(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        x, _ = self.save("Paint the bike shed green with weatherproof paint")
        y, _ = self.save("Repaint the bike shed with weatherproof green paint, not red")
        self.cli("settle", y, x, "--replaces")
        self.assertEqual(self.stored(a)[:2], self.stored(x)[:2])
        self.assertEqual((self.stored(a)[2], self.stored(x)[2]), ([b], [y]))
        self.assertEqual(self.recalled() & {a, x}, set())

    def test_a_record_cannot_replace_itself(self):
        a, _ = self.save(A_TEXT)
        again, _ = self.save(A_TEXT, "--replaces", a)
        self.assertEqual(again, a)
        self.assertEqual((self.pairs(), self.stored(a)[0]), ([], "active"))

    def test_card_ignores_replacements_made_before_this_feature(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.sql("DELETE FROM suspense")   # the old code path left no pair
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','active') WHERE id=?", a)
        self.assertEqual(len(self.cli("card", "--session", "s1").stdout.strip().splitlines()), 1)

    def test_card_skips_pairs_with_a_dead_side(self):
        a, _ = self.save(A_TEXT)
        self.save(B_TEXT)
        self.assertEqual(len(self.cli("card", "--session", "s1").stdout.strip().splitlines()), 2)
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','retired') WHERE id=?", a)
        self.assertEqual(len(self.cli("card", "--session", "s1").stdout.strip().splitlines()), 1)

    def test_card_line_counts_and_stays_silent_when_clean(self):
        self.assertEqual(self.cli("card", "--session", "s1").stdout.strip().count("\n"), 0)   # clean store: the card id only
        a, _ = self.save("Use zebrafish queue for ingestion pipeline")
        p, _ = self.save("Paint the bike shed green with weatherproof paint")
        q, _ = self.save("Repaint the bike shed with weatherproof green paint, not red", "--replaces", p)
        r, _ = self.save("Switch ingestion pipeline to the zebrafish queue v2")   # open pair (r, a); s also meets q: a second pair
        s, _ = self.save("Weatherproof green paint for the bike shed again", "--replaces", p)   # p now has two live replacers
        with sqlite3.connect(self.state / "memory.sqlite3") as db:   # one-sided: the links stay, the stored status is back to active
            db.execute("UPDATE records SET payload=json_set(payload,'$.status','active') WHERE id=?", (p,))
        out = self.cli("card", "--session", "s1").stdout.strip().splitlines()
        self.assertEqual(len(out), 2)
        line = out[1]
        self.assertTrue(line.startswith("memrove open:"), line)
        self.assertIn("open pairs: 2 in project, 2 this session", line)
        self.assertIn("one-sided replaces: 1", line)
        self.assertIn("records with two live replacers: 1", line)
        self.assertLessEqual(len([w for w in line.replace("[", " ").replace("]", " ").split() if w.startswith("obs-")]), 5)
        self.assertIn(f"{r}>{a}", line)   # settle needs both ids
        self.assertNotIn("zebrafish", line)

    def test_neighbour_step_error_still_saves_and_exits_zero(self):
        a, _ = self.save(A_TEXT)
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            db.executescript("DROP TABLE suspense; CREATE TABLE suspense(wrong TEXT)")   # booking will fail
        b, err = self.save(B_TEXT)
        self.assertEqual(self.sql("SELECT count(*) FROM records WHERE id=?", b)[0][0], 1)


if __name__ == "__main__":
    unittest.main()
