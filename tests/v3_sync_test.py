"""Markdown <-> store reconciliation: the source files stay authoritative."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
from memrove_v3_sync import SyncEngine, parse, render  # noqa: E402


class SyncTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-sync-")
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.vault, self.state = base / "vault", base / "state"
        self.vault.mkdir()
        self.engine = SyncEngine(self.vault, self.state)

    def put(self, name, body="Nebula calibration waits for review.\n", **meta):
        path = self.vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(dict({"id": Path(name).stem, "kind": "fact"}, **meta), body), encoding="utf-8")
        return path

    def record(self, rid):
        with self.engine.store._connect() as db:
            row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None

    def test_add_edit_delete_follow_the_file(self):
        path = self.put("notes/a.md")
        self.assertEqual(self.engine.sync()["indexed"], 1)
        first = self.record("a")
        self.assertIn("Nebula", first["text"])
        path.write_text(render({"id": "a", "kind": "fact"}, "Changed text.\n"), encoding="utf-8")
        self.engine.sync()
        self.assertIn("Changed", self.record("a")["text"])
        path.unlink()
        self.assertEqual(self.engine.sync()["deleted"], 1)
        self.assertIsNone(self.record("a"))

    def test_incremental_sync_reads_only_changed_files(self):
        self.put("notes/a.md")
        self.put("notes/b.md")
        self.engine.sync()
        self.put("notes/b.md", "Second body.\n")
        result = self.engine.sync(full=False)
        self.assertEqual((result["mode"], result["indexed"]), ("incremental", 1))

    def test_bad_file_is_a_warning_not_a_crash(self):
        self.put("notes/ok.md")
        (self.vault / "notes/bad.md").write_text("---\n{not json\n---\nx\n", encoding="utf-8")
        result = self.engine.sync()
        self.assertEqual((result["status"], result["indexed"]), ("degraded", 1))
        self.assertEqual(result["warnings"][0]["source"], "notes/bad.md")

    def test_duplicate_ids_are_quarantined(self):
        self.put("notes/x.md", id="same")
        self.put("notes/y.md", id="same")
        result = self.engine.sync()
        self.assertEqual((result["status"], result["indexed"]), ("conflict", 0))

    def test_parse_reads_json_and_flat_yaml(self):
        self.assertEqual(parse('---\n{"a": 1}\n---\nbody')[0], {"a": 1})
        meta, body = parse("---\ntitle: Plan\ntags: [x, y]\nfacts:\n  owner: me\n---\ntext\n")
        self.assertEqual((meta["title"], meta["tags"], meta["facts"], body), ("Plan", ["x", "y"], {"owner": "me"}, "text\n"))
        with self.assertRaises(ValueError):
            parse("---\nkey: |\n  block\n---\n")
        self.assertEqual(parse("no front matter"), ({}, "no front matter"))

    def test_note_create_refuses_overwrite_and_bad_paths(self):
        out = self.engine.note_create("notes/new.md", "A lasting fact.", {"kind": "fact"})
        self.assertEqual(out["status"], "succeeded")
        for source in ("notes/new.md", "elsewhere/x.md", "../notes/x.md"):
            with self.assertRaises(ValueError):
                self.engine.note_create(source, "text")
        with self.assertRaises(ValueError):
            self.engine.note_create("notes/t.md", "x", {"kind": "task"})

    def test_task_create_needs_owner_and_status(self):
        meta = {"id": "t1", "owner": "me", "status": "active"}
        record = self.engine.task_create("tasks/t1.md", "Ship it.", meta)
        self.assertEqual((record["kind"], record["status"]), ("task", "active"))
        with self.assertRaises(ValueError):
            self.engine.task_create("tasks/t2.md", "x", dict(meta, id="t2", owner=""))
        with self.assertRaises(ValueError):
            self.engine.task_create("tasks/t1.md", "again", meta)

    def test_secrets_are_redacted_in_created_notes(self):
        key = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
        self.engine.note_create("notes/s.md", f"token {key} here", {"kind": "fact"})
        self.assertNotIn(key, (self.vault / "notes/s.md").read_text(encoding="utf-8"))

    def test_interrupted_write_is_finished_from_the_journal(self):
        content = render({"id": "j", "kind": "fact"}, "Journaled.\n")
        (self.engine.journal / "j.json").write_text(
            json.dumps({"source": "notes/j.md", "old_hash": None, "content": content, "kind": "note"}), encoding="utf-8")
        self.engine.sync()
        self.assertEqual((self.vault / "notes/j.md").read_text(encoding="utf-8"), content)
        self.assertEqual(list(self.engine.journal.glob("*.json")), [])

    def test_deleted_owned_note_leaves_a_tombstone(self):
        self.put("notes/captured/c1.md", id="c1", kind="decision", status="active")
        with self.engine.store._connect() as db:
            db.execute("INSERT INTO records VALUES ('c1', ?)", (json.dumps({"id": "c1", "revision": 1, "status": "active"}),))
            db.execute("INSERT INTO markdown_sources VALUES ('c1', 'notes/captured/c1.md')")
        (self.vault / "notes/captured/c1.md").unlink()
        self.engine.sync()
        self.assertEqual(self.record("c1")["status"], "deleted_by_user")


if __name__ == "__main__":
    unittest.main()
