#!/usr/bin/env python3
"""Executable memory: --check dry-run refusal, allowlist, stale line + doctor after a later failure, scope in search."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.vault, self.state, self.proj = self.tmp / "Vault", self.tmp / "state", self.tmp / "work" / "myproj"
        self.vault.mkdir(); (self.proj / ".git").mkdir(parents=True)
        self.flag = self.tmp / "flag"
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "MEMROVE"))}, "MEMROVE_READER": "local"}   # the owner, without a harness env
        self.env.update(HOME=str(self.tmp), CLAUDE_CODE_SESSION_ID="s1", MEMROVE_NO_LLM="1")

    def cli(self, *a):
        return subprocess.run([sys.executable, str(ROOT / "scripts/memrove_v3.py"), "--vault", str(self.vault), "--state", str(self.state), *a],
                              env=self.env, capture_output=True, text=True, cwd=self.proj)

    def remember(self, text, *a):
        return self.cli("remember", text, "--kind", "decision", "--cwd", str(self.proj), *a)

    def test_refusals(self):
        bad = self.remember("Failing check record", "--check", f"test -e {self.flag}")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("check refused", bad.stderr + bad.stdout)
        self.assertEqual(self.cli("search", "Failing check record", "--records").stdout.strip() in ("", "[]"), True)
        rm = self.remember("Bad command record", "--check", "rm -rf /tmp/x")
        self.assertNotEqual(rm.returncode, 0)
        self.assertIn("not allowed", rm.stderr + rm.stdout)
        self.assertNotEqual(self.remember("Redirect record", "--check", "cat /etc/hosts > /tmp/x").returncode, 0)
        self.assertNotEqual(self.remember("Writer sqlite", "--check", "sqlite3 x.db 'select 1'").returncode, 0)

    def test_validate_quoting(self):
        import memrove_v3_checks as c
        ok = ["python3 -c \"import datetime,sys;sys.exit(datetime.date.today()>=datetime.date(2026,10,31))\"",
              "python3 -c 'print(1>=0); print(\"a|b\")' && test -e /tmp", "grep '>' /etc/hosts | head -1"]
        for cmd in ok:
            self.assertIsNone(c.validate(cmd), cmd)
        bad = ["cat /etc/hosts > /tmp/x", "cat x >> y", "echo $(id)", "python3 -c \"$(id)\"", "echo `id`", "ls; rm x",
               "ls & rm x", "python3 -c 'x' ; 'memrove' sync", "ls \\\" ; rm x \\\"", "python3 -c 'x' 2>&1"]
        for cmd in bad:
            self.assertIsNotNone(c.validate(cmd), cmd)

    def test_stale_flow_and_scope(self):
        self.flag.write_text("x")
        ok = self.remember("Zebrafish queue stays on", "--check", f"test -e {self.flag}", "--scope", "memrove repo, hook p95, n=60")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        found = self.cli("search", "zebrafish queue", "--records", "--compact")
        self.assertIn("[measured: memrove repo, hook p95, n=60]", found.stdout)
        shown = self.cli("show", ok.stdout.strip())
        self.assertIn("memrove repo, hook p95, n=60", shown.stdout)
        import memrove_v3_checks as c
        sync = self._sync()
        c.run_all(sync.store, self.state)
        self.assertEqual(c.notice(self.state), "")
        self.flag.unlink()  # the world changes later
        c.run_all(sync.store, self.state)
        self.assertEqual(c.notice(self.state), "1 records may be stale: `memrove doctor`\n")
        doc = json.loads(self.cli("doctor").stdout)
        self.assertEqual([r["id"] for r in doc["may_be_stale"]], ["rec:" + ok.stdout.strip()])

    def test_superseded_is_marked_and_skipped(self):
        self.flag.write_text("x")
        old = self.remember("Quokka cache is enabled", "--check", f"test -e {self.flag}").stdout.strip()
        import memrove_v3_checks as c
        sync = self._sync()
        c.run_all(sync.store, self.state)
        self.flag.unlink()
        c.run_all(sync.store, self.state)
        self.assertEqual(c.notice(self.state), "1 records may be stale: `memrove doctor`\n")
        new = self.remember("Quokka cache is disabled now", "--supersedes", old).stdout.strip()
        shown = json.loads(self.cli("show", old).stdout.split("\n", 1)[1])   # the first line is the REPLACED boundary line
        self.assertEqual((shown["status"], shown["superseded_by"]), ("superseded", [new]))
        self.assertEqual(c.notice(self.state), "")  # cached failure dropped on supersede
        self.assertNotIn(old, c.run_all(self._sync().store, self.state))  # and never re-checked

    def _sync(self):
        from memrove_v3_sync import SyncEngine
        return SyncEngine(self.vault, self.state)


if __name__ == "__main__":
    unittest.main()
