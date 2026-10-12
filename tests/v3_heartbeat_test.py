#!/usr/bin/env python3
"""Writer heartbeat: a silent writer with new owner messages raises one SessionStart line; healthy and idle states add nothing."""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'template/.claude/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import memrove_v3_heartbeat as hb  # noqa: E402

NOW = datetime(2026, 10, 12, 12, 0).timestamp()


class HeartbeatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name) / 'state'

    def message(self, when):
        day = self.state / 'chatlog/main' / (when.date().isoformat() + '.jsonl')
        day.parent.mkdir(parents=True, exist_ok=True)
        with open(day, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'i': 1, 'kind': 'user', 'text': 'merhaba', 'date': when.astimezone().isoformat()}) + '\n')

    def test_silent_writer_with_new_messages_gets_one_line(self):
        hb.stamp(self.state, 'j2_threads', now=NOW - 50 * 3600)
        hb.stamp(self.state, 'chatlog', now=NOW - 1 * 3600)
        self.message(datetime.fromtimestamp(NOW - 3600))
        text = hb.line(self.state, NOW)
        self.assertEqual(text.count('\n'), 1)
        self.assertIn('j2_threads', text)
        self.assertNotIn('chatlog', text)

    def test_healthy_state_adds_nothing(self):
        for w in hb.WRITERS:
            hb.stamp(self.state, w, now=NOW - 3600)
        self.message(datetime.fromtimestamp(NOW - 600))
        self.assertEqual(hb.line(self.state, NOW), '')
        self.assertEqual(hb.line(Path(self.tmp.name) / 'never-ran', NOW), '')

    def test_idle_writer_without_new_messages_is_healthy(self):
        hb.stamp(self.state, 'j2_threads', now=NOW - 50 * 3600)
        self.message(datetime.fromtimestamp(NOW - 60 * 3600))  # older than its last success
        self.assertEqual(hb.line(self.state, NOW), '')
        self.assertEqual({r['writer']: r['status'] for r in hb.check(self.state, NOW)}['j2_threads'], 'idle')

    def test_writer_that_never_ran_is_flagged_after_24h_with_new_messages(self):
        (self.state / 'heartbeat').mkdir(parents=True)
        (self.state / 'heartbeat/.since').write_text(str(NOW - 30 * 3600))
        hb.stamp(self.state, 'chatlog', now=NOW - 600)  # the others never stamped
        self.message(datetime.fromtimestamp(NOW - 3600))
        text = hb.line(self.state, NOW)
        self.assertEqual(text.count('\n'), 1)
        self.assertIn('chattree', text)
        self.assertIn('j2_threads', text)
        self.assertNotIn('chatlog', text)
        self.assertNotIn('grab_process', text)
        self.assertEqual({r['writer']: r['status'] for r in hb.check(self.state, NOW)}['j2_threads'], 'never ran')

    def test_never_ran_is_quiet_before_24h_or_without_new_messages(self):
        (self.state / 'heartbeat').mkdir(parents=True)
        (self.state / 'heartbeat/.since').write_text(str(NOW - 30 * 3600))
        self.assertEqual(hb.line(self.state, NOW), '')  # no messages after .since
        (self.state / 'heartbeat/.since').write_text(str(NOW - 2 * 3600))
        self.message(datetime.fromtimestamp(NOW - 3600))
        self.assertEqual(hb.line(self.state, NOW), '')  # under 24 h

    def test_failed_run_alone_does_not_refresh_last_ok(self):
        hb.stamp(self.state, 'chattree', now=NOW - 50 * 3600)
        hb.stamp(self.state, 'chattree', ValueError('boom'), now=NOW - 60)
        self.message(datetime.fromtimestamp(NOW - 3600))
        self.assertIn('chattree', hb.line(self.state, NOW))

    def test_session_start_hook_prints_the_line_only_when_unhealthy(self):
        hook = SCRIPTS / 'memrove_v3_hook.py'
        vault = Path(self.tmp.name) / 'vault'
        vault.mkdir()
        env = {'HOME': self.tmp.name, 'PATH': os.defpath, 'MEMROVE_V3_NO_SPAWN': '1', 'PYTHONDONTWRITEBYTECODE': '1'}

        def start():
            r = subprocess.run([sys.executable, str(hook), '--vault', str(vault), '--state', str(self.state), '--harness', 'codex'],
                               input=json.dumps({'hook_event_name': 'SessionStart', 'session_id': 's', 'event_id': 'e', 'cwd': str(vault)}),
                               text=True, capture_output=True, cwd=vault, env=env, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout
        self.assertNotIn('yazici sessiz', start())
        hb.stamp(self.state, 'chatlog', now=datetime.now().timestamp() - 50 * 3600)
        self.message(datetime.now() - timedelta(hours=1))
        self.assertIn('yazici sessiz: chatlog', start())


if __name__ == '__main__':
    unittest.main()
