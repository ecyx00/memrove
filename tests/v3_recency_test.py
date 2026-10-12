"""A note's date comes from updated_at, or from the first usable alias, and the file is never rewritten."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
from memrove_v3_sync import SyncEngine  # noqa: E402


class RecencyAliasTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-recency-")
        self.addCleanup(tmp.cleanup)
        self.vault, self.state = Path(tmp.name, "vault"), Path(tmp.name, "state")
        self.vault.mkdir()
        self.engine = SyncEngine(self.vault, self.state)

    def dated(self, header):
        path = self.vault / "notes/n.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"---\nid: n\nkind: fact\n{header}\n---\nBody text.\n", encoding="utf-8")
        before = path.read_bytes()
        self.engine.sync()
        self.assertEqual(path.read_bytes(), before)  # sync never rewrites a source
        with self.engine.store._connect() as db:
            import json
            return json.loads(db.execute("SELECT payload FROM records WHERE id='n'").fetchone()[0]).get("updated_at")

    def test_alias_names_give_the_date(self):
        for alias in ("updated", "modified", "last_modified", "date_modified"):
            self.assertEqual(self.dated(f"{alias}: 2026-03-04"), "2026-03-04", alias)

    def test_explicit_updated_at_wins(self):
        self.assertEqual(self.dated("updated_at: 2026-01-01\nupdated: 2026-09-09"), "2026-01-01")

    def test_alias_order_is_updated_then_modified(self):
        self.assertEqual(self.dated("modified: 2026-02-02\nupdated: 2026-05-05"), "2026-05-05")

    def test_values_that_are_not_iso_dates_are_ignored(self):
        self.assertIsNone(self.dated("updated: yesterday\nmodified: 3"))
        self.assertEqual(self.dated("updated: yesterday\nmodified: 2026-06-06"), "2026-06-06")


if __name__ == "__main__":
    unittest.main()
