"""Vault preferences and the check cadence that decides when the background worker may start."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from v3_package_helpers import inherited_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_preferences as prefs  # noqa: E402


class PreferencesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-prefs-")
        self.addCleanup(tmp.cleanup)
        self.vault, self.state = Path(tmp.name, "vault"), Path(tmp.name, "state")
        self.vault.mkdir()

    def hook(self, event, *extra):
        out = subprocess.run([sys.executable, str(ROOT / "template/.claude/scripts/memrove_v3_hook.py"), "--vault", str(self.vault),
                              "--state", str(self.state), "--harness", "codex", *extra],
                             input=json.dumps({"hook_event_name": event, "session_id": "s", "event_id": event}),
                             capture_output=True, text=True, encoding="utf-8", env=inherited_env(MEMROVE_V3_NO_SPAWN="1"), timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_defaults_and_roundtrip(self):
        self.assertEqual(prefs.read(self.vault), prefs.DEFAULTS)
        saved = prefs.save(self.vault, {"interval_minutes": 30})
        self.assertEqual(prefs.read(self.vault), saved)
        self.assertEqual(saved["interval_minutes"], 30)

    def test_profiles_never_flip_the_secret_filter(self):
        prefs.save(self.vault, {"secret_filter": False})
        self.assertEqual(prefs.save(self.vault, {}, "manual"), dict(auto_sync=False, interval_minutes=15, secret_filter=False))

    def test_invalid_input_never_replaces_the_file(self):
        prefs.save(self.vault, {"interval_minutes": 5})
        for bad in ({"interval_minutes": -1}, {"interval_minutes": "5"}, {"auto_sync": "yes"}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                prefs.save(self.vault, bad)
        self.assertEqual(prefs.read(self.vault)["interval_minutes"], 5)
        (self.vault / prefs.FILE).write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            prefs.save(self.vault, {})
        self.assertEqual((self.vault / prefs.FILE).read_text(encoding="utf-8"), "{broken")

    def test_retired_keys_from_older_releases_are_accepted(self):
        (self.vault / prefs.FILE).write_text(json.dumps({"context_mode": "off", "context_chars": 100}), encoding="utf-8")
        self.assertEqual(prefs.read(self.vault), prefs.DEFAULTS)

    def test_cadence_waits_between_events_but_not_at_session_start(self):
        settings = prefs.save(self.vault, {}, "economical")
        self.assertTrue(prefs.claim_check(self.state, settings, "Stop", now=1000))
        self.assertFalse(prefs.claim_check(self.state, settings, "Stop", now=1100))
        self.assertTrue(prefs.claim_check(self.state, settings, "SessionStart", now=1200))
        self.assertTrue(prefs.claim_check(self.state, settings, "Stop", now=1200 + 15 * 60))

    def test_a_pending_hook_error_lifts_the_wait(self):
        settings = prefs.save(self.vault, {}, "economical")
        prefs.claim_check(self.state, settings, "Stop", now=1000)
        (self.state / "hook-error.json").write_text("{}")
        self.assertTrue(prefs.claim_check(self.state, settings, "Stop", now=1010))

    def test_manual_profile_never_starts_work(self):
        self.assertFalse(prefs.claim_check(self.state, prefs.save(self.vault, {}, "manual"), "SessionStart"))

    def test_a_started_worker_respects_manual_but_explicit_drain_still_works(self):
        self.hook("Stop")
        prefs.save(self.vault, {}, "manual")
        self.assertTrue(self.hook("Stop", "--worker")["paused"])
        self.assertEqual(len(list((self.state / "hook-queue").glob("*.json"))), 1)
        self.assertEqual(self.hook("Stop", "--drain-queue")["pending"], 0)


if __name__ == "__main__":
    unittest.main()
