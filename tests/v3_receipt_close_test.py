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

    def force_link(self, new, old):
        """The state an old store can hold (two live replacers): `--replaces` refuses it now, so write the link by SQL."""
        self.sql("UPDATE records SET payload=json_set(payload,'$.supersedes',json_array(?)) WHERE id=?", old, new)

    def pairs(self):
        return self.sql("SELECT new_id, nb_id, state FROM suspense")

    def test_unanswered_neighbour_is_shown_and_booked(self):
        a, _ = self.save(A_TEXT)
        b, err = self.save(B_TEXT)
        self.assertIn(f"neighbour rec:{a}", err)
        self.assertIn(f"can the new record stand in for {a} everywhere?", err)
        for cmd in (f"yes: memrove settle {b} {a} --replaces\n", f"memrove settle {b} {a} --rests-on --why-link \"<why>\"\n", f"neither: memrove settle {b} {a} --unrelated"):
            self.assertIn(cmd, err)
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
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','untrusted') WHERE id=?", a)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.assertEqual(self.status(a), "untrusted")   # a replace never overwrites another status
        self.cli("settle", b, a, "--undo")
        self.assertEqual(self.status(a), "untrusted")
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','active') WHERE id=?", a)
        self.cli("settle", b, a, "--replaces")
        c, _ = self.save("Zebrafish queue ingestion pipeline moves to v3 now")
        self.force_link(c, a)
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
        s, _ = self.save("Weatherproof green paint for the bike shed again")
        self.force_link(s, p)   # p now has two live replacers
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

    def payload(self, rid):
        return json.loads(self.sql("SELECT payload FROM records WHERE id=?", rid)[0][0])

    def test_replaced_record_prints_the_boundary_line_first_and_a_current_one_does_not(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        c, _ = self.save("Paint the bike shed green with weatherproof paint", "--rests-on", a, "--why-link", "was built on the old queue")
        day = self.payload(b)["valid_from"][:10]
        boundary = f"REPLACED by {b} on {day}: history, do not use as current."
        self.assertEqual(self.cli("show", a).stdout.splitlines()[0], boundary)
        self.assertNotIn("REPLACED", self.cli("show", b).stdout)   # a current record carries no boundary line
        self.sql("UPDATE records SET payload=json_remove(payload,'$.valid_to') WHERE id=?", a)   # an older replace may have no closed window: the neighbour summary must not print its text
        self.assertEqual(json.loads(self.cli("show", c).stdout)["edges"][0]["summary"], boundary)

    def test_replace_status_and_both_link_sides_are_one_transaction(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        boom = "CREATE TRIGGER boom BEFORE UPDATE ON records WHEN json_extract(NEW.payload,'$.status')='superseded' BEGIN SELECT RAISE(ABORT,'boom'); END"
        self.sql(boom)   # the replaced record's status write fails after the new record's link was written
        c, err = self.save("Zebrafish queue ingestion pipeline moves to v3 now", "--replaces", a)
        self.assertIn("link not written", err)
        self.assertEqual(self.cli("settle", b, a, "--replaces", ok=False).returncode, 1)
        for new in (b, c):
            self.assertFalse(self.payload(new).get("supersedes"), new)   # the link side did not stay
        old = self.payload(a)
        self.assertEqual((old["status"], old.get("valid_to"), old.get("superseded_by")), ("active", None, None))
        self.assertIn((c, a, "open"), self.pairs())   # the booked question is left, no link
        self.sql("DROP TRIGGER boom")
        self.cli("settle", b, a, "--replaces")
        self.assertEqual((self.status(a), self.payload(b)["supersedes"]), ("superseded", [a]))

    def test_bad_replaces_target_is_a_loud_error_and_the_save_stays(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        n = self.sql("SELECT count(*) FROM records")[0][0]
        for target, reason in (("obs-nosuchrecord", "no unique record"), (a, "already replaced by")):
            new, err = self.save(f"Unrelated note about paint {target}", "--replaces", target)
            self.assertIn(f"error: --replaces {target}: ", err)
            self.assertIn(reason, err)
            self.assertEqual(self.sql("SELECT count(*) FROM records WHERE id=?", new)[0][0], 1)   # the record is stored
            self.assertFalse(self.payload(new).get("supersedes"))
            self.assertEqual(self.sql("SELECT count(*) FROM suspense WHERE new_id=? AND state IN ('replaces','rests_on')", new)[0][0], 0)   # no link row
        self.assertEqual(self.sql("SELECT count(*) FROM records")[0][0], n + 2)
        x, _ = self.save("Cobalt turbine maintenance schedule for winter")
        bad = self.cli("settle", x, a, "--replaces", ok=False)   # settle: same refusal, nothing written
        self.assertEqual(bad.returncode, 1)
        self.assertIn("already replaced by", bad.stderr)
        self.assertFalse(self.payload(x).get("supersedes"))
        self.assertEqual(self.payload(a)["superseded_by"], [b])

    def test_card_counters_each_alone_and_silent_on_a_clean_store(self):
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.assertEqual(self.cli("card", "--session", "s1").stdout.strip().count("\n"), 0)   # a clean replace prints no counter
        c, _ = self.save("Paint the bike shed green with weatherproof paint", "--rests-on", a, "--why-link", "built on the old queue")
        lost, _ = self.save("Quantum banana sandwich architecture review")
        y, _ = self.save("Cobalt turbine maintenance schedule for winter")
        z, _ = self.save("Violet harbour crane inspection checklist weekly")
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','superseded') WHERE id=?", lost)
        self.force_link(y, z)
        self.force_link(z, y)
        line = self.cli("card", "--session", "s1").stdout.strip().splitlines()[1]
        self.assertIn(f"replaced, no live replacer: 1 [{lost}]", line)
        self.assertIn("replace cycle: 2", line)
        self.assertTrue(y in line and z in line)
        self.assertIn(f"basis changed: 1 [{c}]", line)

    def note(self, rid):
        return self.vault / self.payload(rid)["source"]

    def test_replace_link_survives_a_hand_edit_and_a_rebuild_and_undo_still_works(self):   # D1
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.assertIn(a, self.note(b).read_text())   # the link is in B's note file
        self.note(b).write_text(self.note(b).read_text() + "\nA hand-written line.\n")
        self.cli("sync")
        self.assertEqual(self.payload(b)["supersedes"], [a])
        for f in self.state.glob("memory.sqlite3*"):   # a store rebuilt from the vault
            f.unlink()
        self.cli("sync")
        self.assertEqual(self.payload(b)["supersedes"], [a])
        self.assertEqual(self.cli("settle", b, a, "--undo").returncode, 0)
        self.assertIn(a, self.recalled())
        self.assertNotIn(a, self.note(b).read_text())   # the undo removes it from the file too, so a rebuild does not bring it back

    def test_card_replaced_no_live_replacer_ignores_chains_and_other_projects_but_finds_an_orphan(self):   # D2
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        c, _ = self.save("Zebrafish queue ingestion pipeline moves to v3 now", "--replaces", b)
        x, _ = self.save("Cobalt turbine maintenance schedule for winter")
        other = self.proj.parent / "other"
        (other / ".git").mkdir(parents=True)
        p = subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), "remember",
                            "Cobalt turbine winter maintenance schedule moved to spring", "--kind", "decision", "--replaces", x, "--cwd", str(other)], env=self.env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.status(x), "superseded")
        self.assertEqual(self.cli("card", "--session", "s1").stdout.count("no live replacer"), 0)   # chain A<-B<-C and a replace across projects: nothing is wrong
        orphan, _ = self.save("Violet harbour crane inspection checklist weekly")
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','superseded') WHERE id=?", orphan)
        self.assertIn(f"replaced, no live replacer: 1 [{orphan}]", self.cli("card", "--session", "s1").stdout)

    def test_card_counts_a_record_whose_only_replacer_is_retired(self):   # P1: recall still hides it (the retired record's supersedes), the card was silent
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','active') WHERE id=?", a)   # a pre-1a replace: the link without the status
        self.assertEqual(self.cli("card", "--session", "s1").stdout.count("no live replacer"), 0)   # b is live: nothing is wrong
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','retired') WHERE id=?", b)
        self.assertIn(f"replaced, no live replacer: 1 [{a}]", self.cli("card", "--session", "s1").stdout)
        self.assertNotIn(a, self.recalled())   # what recall hides is unchanged

    def test_relate_never_proposes_the_record_the_same_save_replaces(self):   # D10: relate runs inside ingest, before the replace link exists
        a, _ = self.save(A_TEXT)
        c, _ = self.save("Zebrafish queue ingestion pipeline notes from the March review")
        self.sql("UPDATE records SET payload=json_set(payload,'$.valid_from','2026-01-01')")   # a supersedes proposal needs a strictly older record
        launch = ("import runpy, sys; sys.path.insert(0, %r); import memrove_core as C; "
                  "C.Core.relate_decider = staticmethod(lambda old, new: {'relation': 'supersedes', 'why': 'stub', 'scores': {}}); "   # no model call
                  "sys.argv = sys.argv[1:]; runpy.run_path(%r, run_name='__main__')" % (str(ROOT / "template/.claude/scripts"), str(ROOT / "scripts/memrove_v3.py")))
        p = subprocess.run([sys.executable, "-c", launch, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), "remember", B_TEXT,
                            "--kind", "decision", "--replaces", a[:12], "--cwd", str(self.proj)], env={**self.env, "MEMROVE_RELATE": "1"}, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        proposals = [l for l in p.stderr.splitlines() if l.startswith("relate proposal:")]
        self.assertTrue(any(c in l for l in proposals), p.stderr)   # relate still runs for the other neighbour
        self.assertFalse(any(a in l for l in proposals), p.stderr)   # a prefix of the id is enough to name it

    def test_a_model_inferred_supersedes_edge_never_prints_the_replaced_line(self):   # D3
        a, _ = self.save(A_TEXT)
        b, _ = self.save("Paint the bike shed green with weatherproof paint")
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_graph as G
        import memrove_v3_neighbours as nb
        import memrove_v3_project as project
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            G.ensure(db)
            G.add_edge(db, b, a, "supersedes", "the linker thinks so", G.provenance("job:j5", "model", model="haiku", job="j5"))
        db = sqlite3.connect(self.state / "memory.sqlite3")
        self.addCleanup(db.close)
        shown = project.entry(db, self.vault, "myproj", state=self.state)
        self.assertIn(a, shown)
        self.assertNotIn("REPLACED", shown + self.cli("show", a).stdout + self.cli("show", b).stdout)
        self.assertIn("proposal", shown)
        self.assertEqual(nb.replaced(self.state, a), "")
        self.assertEqual(nb.replaced(self.state, "rec:" + a), "")   # `search` takes its REPLACED line from here
        c, _ = self.save("Zebrafish queue ingestion pipeline moves to v3 now", "--replaces", a)   # an explicit link does print
        shown = project.entry(db, self.vault, "myproj", state=self.state)
        self.assertRegex(shown, rf"rec:{a} REPLACED by rec:{c} on \d{{4}}-\d\d-\d\d: history, do not use as current")
        self.assertTrue(self.cli("show", a).stdout.startswith(f"REPLACED by {c} on"))
        self.assertTrue(nb.replaced(self.state, "rec:" + a).startswith(f"REPLACED by {c} on"))

    def test_no_boundary_line_on_a_record_without_a_live_replacer(self):   # anti-goal 9, on the cases that have a replacer which is not live
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_project as project
        db = sqlite3.connect(self.state / "memory.sqlite3")
        self.addCleanup(db.close)
        self.assertIn(f"rec:{a} REPLACED by rec:{b}", project.entry(db, self.vault, "myproj", state=self.state))   # with a live replacer the line is there
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','retired') WHERE id=?", b)
        self.assertNotIn("REPLACED", self.cli("show", a).stdout)
        self.assertNotIn("REPLACED", project.entry(db, self.vault, "myproj", state=self.state))

    def test_a_replace_that_lost_the_race_between_the_check_and_the_write_is_refused(self):   # D8
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        c, _ = self.save("Zebrafish queue ingestion pipeline moves to v3 now")
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_suspense as suspense
        from memrove_v3_sync import SyncEngine
        engine, calls = SyncEngine(self.vault, self.state), []
        connect = engine.store._connect
        def racing():
            calls.append(1)
            if len(calls) == 2:   # between commit 1 and commit 2 another save links `a`
                self.force_link(c, a)
            return connect()
        engine.store._connect = racing
        with self.assertRaises(ValueError):
            suspense.answer(engine, b, [a], [])
        self.assertFalse(self.payload(b).get("supersedes"))   # b's link was not written: two live replacers cannot arise

    def test_a_pair_already_answered_is_not_reopened_by_a_later_replace_call(self):   # D9
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT)
        self.cli("settle", b, a, "--unrelated")
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_suspense as suspense
        from memrove_v3_sync import SyncEngine
        engine, calls = SyncEngine(self.vault, self.state), []
        connect = engine.store._connect
        def failing():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("commit 2 fails")
            return connect()
        engine.store._connect = failing
        with self.assertRaises(RuntimeError):
            suspense.answer(engine, b, [a], [])
        self.assertIn((b, a, "unrelated"), self.pairs())   # commit 1 left the answered pair alone

    def test_reversed_settle_is_refused_and_both_records_stay_visible(self):   # D5
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        bad = self.cli("settle", a, b, "--replaces", ok=False)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("not live", bad.stderr)
        self.assertEqual((self.payload(a).get("supersedes") or [], self.status(b)), ([], "active"))
        self.assertIn(b, self.recalled())
        self.cli("settle", b, a, "--undo")
        self.assertEqual({a, b} - self.recalled(), set())

    def hand_note(self, name, text):
        (self.vault / "knowledge").mkdir(exist_ok=True)
        f = self.vault / "knowledge" / name
        f.write_text(f"---\n# my own comment\ntitle: {name}\nkind: decision\nproject: myproj\ntags: [infra]\n---\n{text}\n")
        self.cli("sync")
        return f, self.sql("SELECT id FROM records WHERE json_extract(payload,'$.source')=?", f"knowledge/{name}")[0][0]

    def test_a_hand_written_note_is_never_rewritten_by_a_replace_and_undo_restores_recall(self):   # N1
        for how in ("remember", "settle"):
            f, h = self.hand_note(f"{how}.md", A_TEXT + f" by hand {how}")
            before = f.read_bytes()
            if how == "remember":
                b, _ = self.save(B_TEXT, "--replaces", h)
            else:
                b, _ = self.save("Cobalt turbine maintenance schedule for winter")
                self.cli("settle", b, h, "--replaces")
            self.assertEqual(f.read_bytes(), before, how)   # comment and YAML untouched
            self.assertEqual((self.payload(b)["supersedes"], self.status(h), self.payload(h)["superseded_by"]), ([h], "superseded", [b]))
            self.assertNotIn(h, self.recalled())
            self.assertTrue(self.cli("show", h).stdout.startswith(f"REPLACED by {b}"), how)
            self.assertEqual(self.cli("settle", b, h, "--undo").returncode, 0)
            self.assertIn(h, self.recalled())
            self.assertEqual((f.read_bytes(), self.status(h)), (before, "active"))

    def test_reversed_settle_on_a_pair_replaced_before_the_status_was_stored_is_refused(self):   # D5, pre-branch pair
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.sql("UPDATE records SET payload=json_set(payload,'$.status','active') WHERE id=?", a)   # the shape of an old replace: closed, but still `active`
        self.sql("DELETE FROM suspense")
        bad = self.cli("settle", a, b, "--replaces", ok=False)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("already replaces", bad.stderr)
        self.assertFalse(self.payload(a).get("supersedes"))
        self.assertEqual(self.status(b), "active")
        self.assertIn(b, self.recalled())   # a is closed by b's link, as before the settle

    def test_boundary_line_survives_a_store_rebuilt_from_a_vault_without_the_old_records_fields(self):   # N3
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_neighbours as nb
        a, _ = self.save(A_TEXT)
        b, _ = self.save(B_TEXT, "--replaces", a)
        self.sql("UPDATE records SET payload=json_remove(json_set(payload,'$.status','active'),'$.valid_to','$.superseded_by') WHERE id=?", a)
        self.assertTrue(nb._run(self.state, lambda db: nb._node(db, a))[1].startswith(f"REPLACED by {b}"))

    def test_a_replace_rolled_back_on_a_later_target_leaves_the_note_files_as_they_were(self):   # N4
        a, _ = self.save(A_TEXT)
        x, _ = self.save("Cobalt turbine maintenance schedule for winter")
        y, _ = self.save("Violet harbour crane inspection checklist weekly")
        c, _ = self.save("Zebrafish queue v3 and the turbine schedule both change now")
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_suspense as suspense
        from memrove_v3_sync import SyncEngine
        files = {r: self.note(r) for r in (a, x, c)}
        before = {r: f.read_bytes() for r, f in files.items()}
        engine, calls = SyncEngine(self.vault, self.state), []
        connect = engine.store._connect
        def racing():
            calls.append(1)
            if len(calls) == 2:   # another save links x between the two commits: the second target fails after the first was written
                self.force_link(y, x)
            return connect()
        engine.store._connect = racing
        with self.assertRaises(ValueError):
            suspense.answer(engine, c, [a, x], [])
        self.assertEqual({r: f.read_bytes() for r, f in files.items()}, before)
        self.cli("sync")
        self.assertFalse(self.payload(c).get("supersedes"))
        self.assertIn(a, self.recalled())

    def test_the_precedence_line_is_in_the_session_start_note_within_budget(self):   # design change 2
        sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
        import memrove_v3_chattree as ct
        (self.state / "chatlog").mkdir(parents=True)
        (self.state / "chatlog" / "view.json").write_text(json.dumps({"T": 1234567}))
        note = ct.instruction(self.state)
        self.assertIn("REPLACED is history; a newer record or the code beats it; name the conflict", note)
        self.assertLessEqual(len(note.encode()), 500)

    def test_neighbour_step_error_still_saves_and_exits_zero(self):
        a, _ = self.save(A_TEXT)
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            db.executescript("DROP TABLE suspense; CREATE TABLE suspense(wrong TEXT)")   # booking will fail
        b, err = self.save(B_TEXT)
        self.assertEqual(self.sql("SELECT count(*) FROM records WHERE id=?", b)[0][0], 1)


if __name__ == "__main__":
    unittest.main()
