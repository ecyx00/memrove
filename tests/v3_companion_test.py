"""Companion folder: where it is, starter notes, size limits."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_companion as companion  # noqa: E402


class CompanionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-companion-")
        self.addCleanup(tmp.cleanup)
        self.vault, self.state = Path(tmp.name, "vault"), Path(tmp.name, "state")
        self.vault.mkdir()

    def test_starter_notes_are_created_once_and_never_overwritten(self):
        self.assertEqual(companion.initialize(self.vault, self.state)["status"], "initialized")
        folder = companion.directory(self.vault)
        for name in companion.STARTERS:
            self.assertTrue((folder / name).is_file(), name)
        (folder / "Core.md").write_text("mine", encoding="utf-8")
        (folder / "Threads.md").unlink()
        self.assertEqual(companion.initialize(self.vault, self.state)["status"], "existing")
        self.assertEqual((folder / "Core.md").read_text(encoding="utf-8"), "mine")
        self.assertFalse((folder / "Threads.md").exists())  # a deleted note stays deleted

    def test_an_existing_folder_is_reused_and_a_soul_replaces_core(self):
        custom = self.vault / "My Echo"
        custom.mkdir()
        (custom / "Soul.md").write_text("identity", encoding="utf-8")
        self.assertEqual(companion.directory(self.vault), custom.resolve())
        companion.initialize(self.vault, self.state)
        self.assertFalse((custom / "Core.md").exists())
        self.assertTrue((custom / "Rules.md").exists())

    def test_two_candidate_folders_are_ambiguous(self):
        for name in ("A companion", "B echo"):
            (self.vault / name).mkdir()
        self.assertIsNone(companion.directory(self.vault))
        self.assertEqual(companion.initialize(self.vault, self.state)["status"], "needs_attention")
        self.assertEqual(companion.hygiene(self.vault, self.state)["status"], "ambiguous_directory")

    def test_archive_folders_are_not_companions(self):
        (self.vault / "Companion Archive").mkdir()
        self.assertEqual(companion.directory(self.vault).name, companion.DEFAULT_DIRECTORY)

    def test_limits_validate_and_persist(self):
        self.assertEqual(companion.read_limits(self.state), (companion.LIMITS, True))
        self.assertEqual(companion.save_limits(self.state, {"Threads.md": 0})["Threads.md"], 0)
        self.assertEqual(companion.read_limits(self.state)[0]["Threads.md"], 0)
        for bad in ({"Threads.md": 5}, {"Core.md": 3000}, {"Threads.md": "9000"}):
            with self.assertRaises(ValueError):
                companion.save_limits(self.state, bad)
        (self.state / "companion-limits.json").write_text("{broken", encoding="utf-8")
        self.assertEqual(companion.read_limits(self.state)[1], False)
        with self.assertRaises(ValueError):
            companion.save_limits(self.state, {})

    def test_hygiene_reports_only_limited_files_as_over(self):
        companion.initialize(self.vault, self.state)
        folder = companion.directory(self.vault)
        (folder / "Last-Session.md").write_text("x" * 3001, encoding="utf-8")
        (folder / "Core.md").write_text("y" * 50000, encoding="utf-8")
        report = companion.hygiene(self.vault, self.state)
        self.assertEqual((report["status"], report["over_limit"]), ("over_limit", ["Last-Session.md"]))
        self.assertEqual(report["files"]["Core.md"], {"chars": 50000})

    def test_symlinked_companion_files_are_ignored(self):
        companion.initialize(self.vault, self.state)
        folder = companion.directory(self.vault)
        outside = self.vault.parent / "outside.md"
        outside.write_text("secret", encoding="utf-8")
        (folder / "Journal.md").unlink()
        (folder / "Journal.md").symlink_to(outside)
        self.assertNotIn("Journal.md", companion.hygiene(self.vault, self.state)["files"])


if __name__ == "__main__":
    unittest.main()
