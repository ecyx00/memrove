#!/usr/bin/env python3
"""Sync keeps store-only fields (assessments, status, rule) of records whose Markdown did not change; a real edit still bumps revision."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_llm_test import Base  # noqa: E402
from memrove_v3_context import remember  # noqa: E402
from memrove_v3_projections import update_record  # noqa: E402


class SyncKeepsStoreFieldsTest(Base):
    def setUp(self):
        super().setUp()
        rid = remember(self.engine, 'note', 'Observed note that sync must not flatten.')['id']
        self.rid = rid
        self.assertTrue(update_record(self.engine, rid, {'status': 'dropped_llm', 'assessment_state': 'assessed',
                                                          'assessments': {'j1': {'verdict': 'drop'}}, 'rule': 'x'}))
        self.before = self.records()[rid]
        self.path = next(p for p in self.vault.rglob('*.md') if 'Observed note' in p.read_text(encoding='utf-8'))

    def check_kept(self):
        after = self.records()[self.rid]
        for k in ('status', 'assessment_state', 'assessments', 'rule', 'revision'):
            self.assertEqual(after.get(k), self.before.get(k), k)

    def test_full_sync_keeps_fields(self):
        self.engine.sync(full=True)
        self.check_kept()

    def test_incremental_sync_after_touch_keeps_fields(self):
        st = self.path.stat()
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        self.engine.sync(full=False)
        self.check_kept()

    def test_moved_file_follows_new_path(self):
        new = self.path.with_name('moved-' + self.path.name)
        self.path.rename(new)
        self.engine.sync(full=True)
        self.engine.sync(full=True)  # a stale source would make this second pass drop the record
        after = self.records()[self.rid]
        self.assertEqual(after['source'], new.relative_to(self.vault).as_posix())
        self.assertEqual(after.get('status'), 'dropped_llm')

    def test_real_edit_bumps_revision(self):
        self.path.write_text(self.path.read_text(encoding='utf-8') + '\nAn extra line.\n', encoding='utf-8')
        self.engine.sync(full=True)
        self.assertGreater(self.records()[self.rid]['revision'], self.before['revision'])


if __name__ == '__main__':
    unittest.main()
