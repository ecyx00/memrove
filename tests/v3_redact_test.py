#!/usr/bin/env python3
"""Display-time secret redaction: search hits and zoom hide secrets; the stored log keeps them."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_chattree as ct  # noqa: E402

DAY = '2026-03-%02dT10:00:00+00:00'
KEY = 'KGAT_' + 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6'
GH = 'ghp_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8'
PW = 'hunter2Hunter2hunter2xyz'


class RedactTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='v3-redact-')
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        chatlog.append(self.state, [
            {'kind': 'user', 'text': f'deploy credentials: DATASET_KEY = "{KEY}" and token {GH} for the uploader', 'date': DAY % 1},
            {'kind': 'user', 'text': f'deploy config has password={PW} in it, see m12 and id abc123', 'date': DAY % 2}])

    @mock.patch.object(ct, '_mem', lambda state: types.SimpleNamespace(T=2))
    def test_search_and_zoom_hide_secrets(self):
        out = cs.search(self.state, 'deploy') + ct.zoom(self.state, 0, 1) + ct.zoom(self.state, 1, 1)
        for secret in (KEY, GH, PW):
            self.assertNotIn(secret, out)
        self.assertIn('[redacted secret]', out)
        self.assertIn('for the uploader', out)
        self.assertEqual(next(chatlog.messages(self.state, 0, 1))['text'].count(KEY), 1)   # the log is untouched

    def test_ordinary_text_is_kept(self):
        for s in ('the api_key parameter is documented in the uploader section', 'token: some_function_name_without_digits', 'see m1542 and id abc123 sk-short'):
            self.assertEqual(cs.redact(s), s)


if __name__ == '__main__':
    unittest.main()
