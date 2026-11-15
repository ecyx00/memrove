#!/usr/bin/env python3
"""The code that writes replace links into note files of the Vault (review 3 of step 1b, D1-D7): a journal on disk survives a kill, a hand edit is
never lost, only the touched lines of a note change (comments, CRLF, BOM), the restore runs under the write lock and restores every file.
Real CLI on a temporary Vault and store; no model."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "template/.claude/scripts"
A_TEXT = "Use zebrafish queue for ingestion pipeline"
B_TEXT = "Switch ingestion pipeline to the zebrafish queue v2"
# a launcher that runs the real CLI and kills the process (os._exit, as SIGKILL would) at one point of a replace or an undo
LAUNCH = """
import os, runpy, sys
sys.path.insert(0, %r)
import memrove_v3_suspense as S
kill, n, real_ff, real_mark = os.environ["VWP_KILL"], [0], S._file_fields, S._mark
def ff(*a, **k):
    r = real_ff(*a, **k)
    if r:
        n[0] += 1
        if kill == "write:%%d" %% n[0]:
            os._exit(9)
    return r
def mark(db, new, nb, state, keep=False):
    real_mark(db, new, nb, state, keep)
    if kill == "precommit" and not keep:
        os._exit(9)
S._file_fields, S._mark = ff, mark
sys.argv = sys.argv[1:]
runpy.run_path(%r, run_name="__main__")
""" % (str(SCRIPTS), str(ROOT / "scripts/memrove_v3.py"))


class VaultWritePath(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp = Path(tmp.name).resolve()
        self.vault, self.state, self.proj = tmp / "Vault", tmp / "state", tmp / "work" / "myproj"
        self.vault.mkdir()
        (self.proj / ".git").mkdir(parents=True)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}
        self.env.update(HOME=str(tmp), CLAUDE_CODE_SESSION_ID="s1", MEMROVE_NO_LLM="1", MEMROVE_READER="local")

    def argv(self, cmd, *a):
        extra = ["--cwd", str(self.proj)] if cmd in ("remember", "card") else []
        return [str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), cmd, *a, *extra]

    def cli(self, cmd, *a, kill=None, ok=True):
        argv = [sys.executable, "-c", LAUNCH, *self.argv(cmd, *a)] if kill else [sys.executable, *self.argv(cmd, *a)]
        p = subprocess.run(argv, env=dict(self.env, **({"VWP_KILL": kill} if kill else {})), capture_output=True, text=True)
        if ok and not kill:
            self.assertEqual(p.returncode, 0, p.stderr)
        return p

    def save(self, text, *a):
        return self.cli("remember", text, "--kind", "decision", *a).stdout.strip()

    def sql(self, q, *args):
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            return db.execute(q, args).fetchall()

    def payload(self, rid):
        return json.loads(self.sql("SELECT payload FROM records WHERE id=?", rid)[0][0])

    def note(self, rid):
        return self.vault / self.payload(rid)["source"]

    def recalled(self):
        sys.path.insert(0, str(SCRIPTS))
        from memrove_v3_recall import _load
        from memrove_v3_sync import SyncEngine
        return {rid for _, _, rid in _load(SyncEngine(self.vault, self.state).store)}

    def agree(self, *ids):
        """The note file and the store hold the same bytes' hash and the same link fields."""
        for rid in ids:
            data, p = self.note(rid).read_bytes(), self.payload(rid)
            meta = json.loads(data.decode().split("---\n")[1])
            self.assertEqual(hashlib.sha256(data).hexdigest(), p["source_sha256"], rid)
            for k in ("supersedes", "valid_to", "superseded_by"):
                self.assertEqual(meta.get(k) or None, p.get(k) or None, f"{rid} {k}")

    # D1: kill points of replace and undo ------------------------------------------------------------------------------

    def kill_replace(self, point, next_cmd):
        a, b = self.save(A_TEXT), self.save(B_TEXT)
        before = {r: self.note(r).read_bytes() for r in (a, b)}
        self.assertNotEqual(self.cli("settle", b, a, "--replaces", kill=point).returncode, 0)   # killed
        self.assertNotEqual({r: self.note(r).read_bytes() for r in (a, b)}, before, "the kill point was not reached")
        self.cli(next_cmd, *(["--session", "s1"] if next_cmd == "card" else []))
        self.assertEqual({r: self.note(r).read_bytes() for r in (a, b)}, before)   # the notes are back, as the store is
        self.agree(a, b)
        self.assertIn(a, self.recalled())
        self.cli("settle", b, a, "--replaces")
        self.assertNotIn(a, self.recalled())
        self.assertEqual(self.cli("settle", b, a, "--undo").returncode, 0)
        self.assertIn(a, self.recalled())

    def test_d1_replace_killed_after_the_first_note_write(self):
        self.kill_replace("write:1", "sync")

    def test_d1_replace_killed_after_the_second_note_write(self):
        self.kill_replace("write:2", "sync")

    def test_d1_replace_killed_before_the_commit(self):
        self.kill_replace("precommit", "card")

    def kill_undo(self, point, next_cmd):
        a, b = self.save(A_TEXT), self.save(B_TEXT)
        self.cli("settle", b, a, "--replaces")
        before = {r: self.note(r).read_bytes() for r in (a, b)}
        self.assertNotEqual(self.cli("settle", b, a, "--undo", kill=point).returncode, 0)
        self.assertNotEqual({r: self.note(r).read_bytes() for r in (a, b)}, before, "the kill point was not reached")
        self.cli(next_cmd, *(["--session", "s1"] if next_cmd == "card" else []))
        self.assertEqual({r: self.note(r).read_bytes() for r in (a, b)}, before)   # still the replaced state, in files and store
        self.agree(a, b)
        self.assertNotIn(a, self.recalled())
        self.assertEqual(self.cli("settle", b, a, "--undo").returncode, 0)   # e222edc: exit 1, A hidden for good
        self.assertIn(a, self.recalled())
        self.agree(a, b)

    def test_d1_undo_killed_after_the_first_note_write(self):
        self.kill_undo("write:1", "sync")

    def test_d1_undo_killed_after_the_second_note_write(self):
        self.kill_undo("write:2", "sync")

    def test_d1_undo_killed_before_the_commit(self):
        self.kill_undo("precommit", "card")

    def test_d1_remember_replaces_killed_after_the_new_notes_write_keeps_the_save(self):
        a = self.save(A_TEXT)
        self.assertNotEqual(self.cli("remember", B_TEXT, "--kind", "decision", "--replaces", a, kill="write:1").returncode, 0)
        b = self.sql("SELECT id FROM records WHERE json_extract(payload,'$.text') LIKE ?", B_TEXT + "%")[0][0]   # the save is kept
        self.cli("sync")
        self.assertFalse(self.payload(b).get("supersedes"))
        self.agree(a, b)
        self.assertIn(a, self.recalled())
        self.assertEqual(self.cli("settle", b, a, "--replaces").returncode, 0)

    # D2: a hand edit of an OWNED note ---------------------------------------------------------------------------------

    def test_d2_a_hand_added_line_in_a_projected_note_survives_settle_and_undo(self):
        cap = self.vault / "notes/captured/cap.md"
        cap.parent.mkdir(parents=True)
        cap.write_text('---\n{"id": "q-cap1", "kind": "decision", "project": "myproj"}\n---\nAmber kiln firing temperature log daily\n')
        self.cli("sync")
        b = self.save("Kiln temperature log is now weekly")
        cap.write_text(cap.read_text() + "\nOwner added this line by hand.\n")   # no sync has read it
        self.cli("settle", b, "q-cap1", "--replaces")
        self.assertIn("Owner added this line by hand.", cap.read_text())
        self.assertIn("Owner added this line by hand.", self.payload("q-cap1")["user_note"])
        self.cli("settle", b, "q-cap1", "--undo")
        self.assertIn("Owner added this line by hand.", cap.read_text())

    # D3, D6, D7: only the lines of the changed keys differ -----------------------------------------------------------

    def hand_edit(self, rid, bom=True):
        """The note as an owner's editor leaves it: YAML, a comment, an unknown key, CRLF, a BOM, no final newline."""
        f = self.note(rid)
        meta = json.loads(f.read_text().split("---\n")[1])
        rows = ["---", "# owner comment: keep this", "tags: [infra, \"queue\"]", "mood: ünïcödé ✓", f"id: {meta['id']}", f"kind: {meta['kind']}"]
        rows += [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items() if k not in ("id", "kind")]
        rows += ["---", A_TEXT, "", "Hand line one - çay", "  indented line   ", "last line no newline"]
        f.write_bytes((b"\xef\xbb\xbf" if bom else b"") + "\r\n".join(rows).encode("utf-8"))
        return f

    def test_d3_d6_d7_a_replace_changes_only_the_lines_of_its_keys_and_undo_gives_the_bytes_back(self):
        a, b = self.save(A_TEXT), self.save(B_TEXT)
        f = self.hand_edit(a)
        start = f.read_bytes()
        self.cli("settle", b, a, "--replaces")
        after = f.read_bytes()
        was, now = start.splitlines(keepends=True), after.splitlines(keepends=True)
        gone, new = [l for l in was if l not in now], [l for l in now if l not in was]
        print("\nbyte diff of the hand-edited note (BOM, YAML comment, CRLF):\n" + "".join(f"  - {l!r}\n" for l in gone) + "".join(f"  + {l!r}\n" for l in new))
        self.assertEqual(gone, [])   # nothing the owner wrote was removed or changed
        self.assertEqual(sorted(l.split(b":")[0] for l in new), [b"superseded_by", b"valid_to"])
        self.assertTrue(all(l.endswith(b"\r\n") for l in new))
        self.assertTrue(after.startswith(b"\xef\xbb\xbf") and b"# owner comment" in after and after.count(b"\r") == start.count(b"\r") + len(new))
        self.assertEqual(self.payload(a)["superseded_by"], [b])
        self.assertNotIn(a, self.recalled())
        self.cli("settle", b, a, "--undo")
        self.assertEqual(f.read_bytes(), start)   # byte-identical, not only the same values
        self.assertIn(a, self.recalled())
        self.cli("settle", b, a, "--replaces")
        self.cli("sync")
        self.assertEqual(self.payload(a)["source"], f.relative_to(self.vault).as_posix())   # the record is kept, with its id
        self.assertEqual(self.sql("SELECT count(*) FROM records WHERE id LIKE 'md-%'")[0][0], 0)
        self.assertEqual((self.payload(a)["superseded_by"], self.payload(b)["supersedes"]), ([b], [a]))
        self.assertNotIn(a, self.recalled())

    def test_d6_a_note_without_editable_front_matter_is_not_written_and_stderr_says_so(self):
        a, b = self.save(A_TEXT), self.save(B_TEXT)
        f = self.note(a)
        f.write_bytes(b"Just a body with no front matter\r\n")
        p = self.cli("settle", b, a, "--replaces")
        self.assertEqual(f.read_bytes(), b"Just a body with no front matter\r\n")
        self.assertIn("kept in the store only", p.stderr)
        self.assertEqual(self.payload(a)["superseded_by"], [b])   # the link is in the store

    # D4, D5 and the Ctrl-C path ---------------------------------------------------------------------------------------

    def staged(self):
        """c replaces [a, x]; the second target fails under the write lock because another save links x in between (as the N4 test)."""
        a, x, y = self.save(A_TEXT), self.save("Cobalt turbine maintenance schedule for winter"), self.save("Violet harbour crane inspection checklist weekly")
        return a, x, y, self.save("Zebrafish queue v3 and the turbine schedule both change now")

    def test_d4_a_failed_replace_does_not_write_stale_text_over_another_commit(self):
        a, x, y, c = self.staged()
        d = self.save("Pelican ferry timetable revision for spring")
        marker = self.state.parent / "restoring"
        code = f"""
import sys, time, sqlite3
sys.path.insert(0, {str(SCRIPTS)!r})
import memrove_v3_suspense as s, memrove_v3_sync as sync
from memrove_v3_sync import SyncEngine
e = SyncEngine({str(self.vault)!r}, {str(self.state)!r})
orig, calls, connect, real = open({str(self.note(a))!r}, 'rb').read(), [], e.store._connect, sync.atomic
def racing():
    calls.append(1)
    if len(calls) == 2:
        with sqlite3.connect({str(self.state / 'memory.sqlite3')!r}) as db:
            db.execute("UPDATE records SET payload=json_set(payload,'$.supersedes',json_array(?)) WHERE id=?", ({x!r}, {y!r}))
    return connect()
def slow(path, text):   # the write that puts a's original text back is slow: the window another process could use
    if str(path) == {str(self.note(a))!r} and text.encode() == orig:
        open({str(marker)!r}, 'w').close(); time.sleep(3)
    return real(path, text)
e.store._connect, sync.atomic = racing, slow
try:
    s.answer(e, {c!r}, [{a!r}, {x!r}], [])
except ValueError as exc:
    print('P1 raised ValueError')
"""
        p1 = subprocess.Popen([sys.executable, "-c", code], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.1)
        self.assertTrue(marker.exists(), "P1 never reached its restore")
        self.assertEqual(self.cli("settle", d, a, "--replaces").returncode, 0)   # P2 commits while P1 restores (it waits for the lock, or runs after)
        out, err = p1.communicate(timeout=60)
        self.assertIn("P1 raised ValueError", out + err)
        self.agree(a, d)   # e222edc: a's file holds P1's stale text, not P2's superseded_by
        self.assertEqual(self.payload(a)["superseded_by"], [d])
        self.assertIn(d, self.note(a).read_text())

    def test_d5_one_failing_restore_does_not_stop_the_others_and_the_original_error_is_raised(self):
        a, x, y, c = self.staged()
        sys.path.insert(0, str(SCRIPTS))
        import memrove_v3_suspense as suspense
        import memrove_v3_sync
        from memrove_v3_sync import SyncEngine
        before = {r: self.note(r).read_bytes() for r in (a, c)}
        engine, calls = SyncEngine(self.vault, self.state), []
        connect = engine.store._connect
        def racing():
            calls.append(1)
            if len(calls) == 2:
                self.sql("UPDATE records SET payload=json_set(payload,'$.supersedes',json_array(?)) WHERE id=?", x, y)
            return connect()
        engine.store._connect = racing
        real = memrove_v3_sync.atomic
        def failing(path, text):   # the restore of a's note fails (disk full, a permission)
            if Path(path) == self.note(a) and text.encode() == before[a]:
                raise OSError(28, "No space left on device (injected)")
            return real(path, text)
        memrove_v3_sync.atomic = failing
        try:
            with self.assertRaises(ValueError) as caught:   # the original error, not the OSError
                suspense.answer(engine, c, [a, x], [])
        finally:
            memrove_v3_sync.atomic = real
        self.assertIn("no link written", str(caught.exception))
        self.assertEqual(self.note(c).read_bytes(), before[c])   # the file after the failing one is restored
        self.cli("sync")   # the one that failed is finished by the journal on disk
        self.assertEqual({r: self.note(r).read_bytes() for r in (a, c)}, before)
        self.assertIn(a, self.recalled())

    def test_a_ctrl_c_inside_the_transaction_puts_the_notes_back(self):
        a, b = self.save(A_TEXT), self.save(B_TEXT)
        sys.path.insert(0, str(SCRIPTS))
        import memrove_v3_suspense as suspense
        from memrove_v3_sync import SyncEngine
        before = {r: self.note(r).read_bytes() for r in (a, b)}
        real = suspense._mark
        def mark(db, new, nb, state, keep=False):
            if state == "replaces":
                raise KeyboardInterrupt
            return real(db, new, nb, state, keep)
        suspense._mark = mark
        try:
            with self.assertRaises(KeyboardInterrupt):
                suspense.settle(SyncEngine(self.vault, self.state), b, a, "replaces")
        finally:
            suspense._mark = real
        self.assertEqual({r: self.note(r).read_bytes() for r in (a, b)}, before)


    # final review fixes: M1, L1, m2, m1, m3 ---------------------------------------------------------------------------

    def test_m1_a_bad_file_in_the_rollback_folder_is_moved_to_bad_and_stops_nothing(self):
        self.save(A_TEXT)
        rb = self.state / "markdown-rollback"
        rb.mkdir(exist_ok=True)
        engine = "import sys; sys.path.insert(0, %r); from memrove_v3_sync import SyncEngine; SyncEngine(%r, %r)" % (str(SCRIPTS), str(self.vault), str(self.state))   # the hook drain's start
        runs = {"search": lambda: self.cli("search", "--json", "zebrafish"), "card": lambda: self.cli("card", "--session", "s1"),
                "remember": lambda: self.cli("remember", "Unrelated note about kiln schedules", "--kind", "decision"),
                "engine": lambda: subprocess.run([sys.executable, "-c", engine], env=self.env, capture_output=True, text=True)}
        variants = {"foreign.json": b"null", "list.json": b"[1]", "obj.json": b"{}", "empty.json": b"", "bytes.json": os.urandom(64) + b"\xff\xfe",
                    "other-name.json": b'[{"source": "x.md", "old": 5, "sha": "0"}]'}
        for name, data in variants.items():
            for cmd, run in runs.items():
                (rb / name).write_bytes(data)
                p = run()
                self.assertEqual(p.returncode, 0, f"{cmd} with {name}: {p.stderr}")
                self.assertTrue((rb / "bad" / name).is_file() and not (rb / name).exists(), f"{cmd} with {name}")
                self.assertEqual(len([l for l in p.stderr.splitlines() if "rollback journal" in l]), 1, p.stderr)
        (rb / "notes.txt").write_text("not a journal")   # only *.json is a journal; any other file is left alone
        self.assertEqual(self.cli("card", "--session", "s1").returncode, 0)
        self.assertTrue((rb / "notes.txt").is_file())

    def test_n5_a_journal_that_cannot_be_resolved_stays_quietly_and_resolves_when_the_folder_is_writable(self):
        a = self.save(A_TEXT)
        note, before = self.note(a), self.note(a).read_bytes()
        ahead = before.replace(A_TEXT.encode(), b"an ahead text that was never committed")
        rb = self.state / "markdown-rollback"
        rb.mkdir(exist_ok=True)
        journal = rb / "1-abc.json"
        journal.write_text(json.dumps([{"source": str(note.relative_to(self.vault)), "old": before.decode(), "sha": hashlib.sha256(ahead).hexdigest()}]))
        note.write_bytes(ahead)   # the store still holds `before`: the write was not committed
        engine = "import sys; sys.path.insert(0, %r); from memrove_v3_sync import SyncEngine; SyncEngine(%r, %r)" % (str(SCRIPTS), str(self.vault), str(self.state))
        runs = [("card", lambda: self.cli("card", "--session", "s1")), ("search", lambda: self.cli("search", "--json", "zebrafish")),
                ("engine", lambda: subprocess.run([sys.executable, "-c", engine], env=self.env, capture_output=True, text=True))]
        os.chmod(note.parent, 0o555)   # a read-only folder: the temp file of the restore cannot be made
        self.addCleanup(os.chmod, note.parent, 0o755)
        try:
            for cmd, run in runs + [("sync-last", lambda: self.cli("sync"))]:   # sync scans the file as it is, so it goes last
                p = run()
                self.assertEqual(p.returncode, 0, f"{cmd}: {p.stderr}")
                self.assertEqual(len([l for l in p.stderr.splitlines() if "rollback journal" in l and "not resolved" in l]), 1, f"{cmd}: {p.stderr}")   # once per command, not once per sync inside it
                self.assertTrue(journal.is_file() and not (rb / "bad").exists(), cmd)
                if cmd != "sync-last":
                    self.assertEqual(note.read_bytes(), ahead)
        finally:
            os.chmod(note.parent, 0o755)
        note.write_bytes(ahead)   # the sync above took the file into the store: make the write uncommitted again
        with sqlite3.connect(self.state / "memory.sqlite3") as db:
            db.execute("UPDATE records SET payload=json_set(payload,'$.source_sha256',?) WHERE id=?", (hashlib.sha256(before).hexdigest(), a))
        p = self.cli("card", "--session", "s1")
        self.assertEqual(p.stderr, "")
        self.assertEqual(note.read_bytes(), before)   # resolved once the folder is writable
        self.assertFalse(journal.exists())

    def test_l1_an_engine_built_before_a_kill_does_not_sync_the_uncommitted_note(self):
        sys.path.insert(0, str(SCRIPTS))
        from memrove_v3_sync import SyncEngine
        a, b = self.save(A_TEXT), self.save(B_TEXT)
        before = {r: self.note(r).read_bytes() for r in (a, b)}
        early = SyncEngine(self.vault, self.state)   # the hook drain builds its engine here and syncs later
        self.assertNotEqual(self.cli("settle", b, a, "--replaces", kill="precommit").returncode, 0)
        self.assertNotEqual(self.note(a).read_bytes(), before[a], "the kill point was not reached")
        early.sync(full=False)   # without the journal check in sync() this reads the uncommitted note into the store
        self.assertEqual({r: self.note(r).read_bytes() for r in (a, b)}, before)
        self.agree(a, b)
        self.assertIn(a, self.recalled())
        self.assertEqual(self.sql("SELECT count(*) FROM records WHERE json_extract(payload,'$.superseded_by') IS NOT NULL")[0][0], 0)

    def test_m2_the_owned_note_of_a_killed_replace_is_healed_by_the_next_command_not_only_by_sync(self):
        cap = self.vault / "notes/captured/cap.md"
        cap.parent.mkdir(parents=True)
        cap.write_text('---\n{"id": "q-cap1", "kind": "decision", "project": "myproj"}\n---\nAmber kiln firing temperature log daily\n')
        self.cli("sync")
        b = self.save("Kiln temperature log is now weekly")
        before = cap.read_bytes()
        self.assertNotEqual(self.cli("settle", b, "q-cap1", "--replaces", kill="precommit").returncode, 0)
        self.assertIn(b"superseded_by", cap.read_bytes())   # the kill point was reached: the note is ahead of the store
        self.cli("card", "--session", "s1")
        self.assertEqual(cap.read_bytes(), before)
        self.assertEqual(hashlib.sha256(cap.read_bytes()).hexdigest(), self.payload("q-cap1")["source_sha256"])
        self.assertEqual(self.cli("settle", b, "q-cap1", "--replaces").returncode, 0)   # and the replace still works afterwards

    # m1, m3: edit_front keeps every byte outside the lines of the edited keys -----------------------------------------

    def edit(self, text, **fields):
        sys.path.insert(0, str(SCRIPTS))
        from memrove_v3_sync import edit_front
        return edit_front(text, fields)

    def test_m1_an_indented_comment_or_blank_line_under_an_edited_scalar_key_stays(self):
        out = self.edit("---\nvalid_to: old\n  # set by hand\n   \nkind: note\n---\nB\n", valid_to="2026-11-15", superseded_by=["obs-b"])
        self.assertEqual(out, '---\nvalid_to: "2026-11-15"\n  # set by hand\n   \nkind: note\nsuperseded_by: ["obs-b"]\n---\nB\n')
        self.assertEqual(self.edit("---\nkind: note\nsuperseded_by: [a]\n  # keep: reason\n---\nB\n", superseded_by=[]), "---\nkind: note\n  # keep: reason\n---\nB\n")
        # a block value still owns its indented lines
        self.assertEqual(self.edit("---\nsuperseded_by:\n  - a\n  - b\nkind: note\n---\nB\n", superseded_by=[]), "---\nkind: note\n---\nB\n")
        self.assertEqual(self.edit("---\nvalid_to: |\n  x\n  y\nkind: note\n---\nB\n", valid_to=""), "---\nkind: note\n---\nB\n")

    def test_m3_a_json_form_note_gets_new_keys_without_changing_any_existing_line(self):
        head = '---\n{\n  "kind": "note",\n  "valid_to": "old"\n}\n---\nB\n'
        out = self.edit(head, superseded_by=["obs-b"])
        self.assertEqual(out, '---\n{\n  "superseded_by": [\n    "obs-b"\n  ],\n  "kind": "note",\n  "valid_to": "old"\n}\n---\nB\n')   # lines only added
        self.assertIsNotNone(self.edit(out, valid_to="2026-11-15"))   # and it stays an editable render form
        # pinned: dropping the LAST key must change the comma of the line before it (JSON has no trailing comma), nothing else
        self.assertEqual(self.edit(head, valid_to=""), '---\n{\n  "kind": "note"\n}\n---\nB\n')


if __name__ == "__main__":
    unittest.main()
