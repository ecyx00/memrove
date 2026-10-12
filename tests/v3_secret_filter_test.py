"""The secret filter: patterns, custom literals, and the write paths that use it."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
from memrove_v3_preferences import save  # noqa: E402
from memrove_v3_secrets import MASK, health, redact  # noqa: E402
from memrove_v3_sync import SyncEngine  # noqa: E402

# Built from pieces so this file never holds a complete token.
TOKENS = {
    "openai": "sk-" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7",
    "github": "gh" + "p_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
    "aws": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
    "pem": "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
    "url": "postgres://admin:hunter22pw@db.example.com/app",
}


class RedactTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-secrets-")
        self.addCleanup(tmp.cleanup)
        self.vault, self.state = Path(tmp.name, "vault"), Path(tmp.name, "state")
        self.vault.mkdir()

    def test_token_families_are_masked(self):
        for name, token in TOKENS.items():
            text, count = redact(f"before {token} after", self.state)
            self.assertEqual(count >= 1, True, name)
            self.assertNotIn(token.split("@")[0].split(":")[-1], text, name)
        self.assertIn("db.example.com", redact(TOKENS["url"], self.state)[0])

    def test_named_values_are_masked_in_text_json_and_query(self):
        for sample in ("password = correct-horse-battery", '{"api_key": "abcdef123456"}', "https://x.test/?token=abcdefgh1234&a=1",
                       'escaped {\\"secret\\": \\"abcdef123456\\"}'):
            text, count = redact(sample, self.state)
            self.assertGreaterEqual(count, 1, sample)
            self.assertNotRegex(text, r"correct-horse|abcdef123456|abcdefgh1234")

    def test_ordinary_text_is_left_alone(self):
        for sample in ("The password policy is documented.", "skip-this-ordinary-hyphenated-identifier-name",
                       "token budget is 4000", "commit 0123456789abcdef0123456789abcdef01234567"):
            self.assertEqual(redact(sample, self.state), (sample, 0), sample)

    def test_custom_literals_are_masked(self):
        self.state.mkdir()
        (self.state / "secret-patterns.txt").write_text("# mine\nproject-codename-x\n", encoding="utf-8")
        self.assertEqual(redact("see project-codename-x twice project-codename-x", self.state),
                         (f"see {MASK} twice {MASK}", 2))
        (self.state / "secret-patterns.txt").write_text("abc\n", encoding="utf-8")  # too short
        with self.assertRaises(ValueError):
            redact("x", self.state)

    def test_long_pathological_input_stays_fast(self):
        start = time.perf_counter()
        redact("password=" + "a" * 200000 + " " + "-----BEGIN PRIVATE KEY-----" * 2000, self.state)
        self.assertLess(time.perf_counter() - start, 5)

    def test_created_notes_are_redacted_and_counted_unless_disabled(self):
        engine = SyncEngine(self.vault, self.state)
        engine.note_create("notes/a.md", f"key {TOKENS['openai']}", {"kind": "fact", "title": f"t {TOKENS['github']}"})
        written = (self.vault / "notes/a.md").read_text(encoding="utf-8")
        self.assertNotIn(TOKENS["openai"], written)
        self.assertNotIn(TOKENS["github"], written)
        self.assertGreaterEqual(health(self.state)["total"], 2)
        save(self.vault, {"secret_filter": False})
        engine.note_create("notes/b.md", f"key {TOKENS['openai']}", {"kind": "fact"})
        self.assertIn(TOKENS["openai"], (self.vault / "notes/b.md").read_text(encoding="utf-8"))

    def test_health_keeps_counts_not_matches(self):
        engine = SyncEngine(self.vault, self.state)
        engine.note_create("notes/c.md", f"key {TOKENS['aws']}", {"kind": "fact"})
        self.assertNotIn(TOKENS["aws"], repr(health(self.state)))


if __name__ == "__main__":
    unittest.main()
