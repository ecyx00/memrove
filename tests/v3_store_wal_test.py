#!/usr/bin/env python3
"""The store runs in WAL mode, so an open reader (a hook, a long search) does not lock out a writer (remember, extract)."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
from memrove_v3_sync import SyncEngine  # noqa: E402


class StoreWalTest(unittest.TestCase):
    def test_reader_does_not_block_writer(self):
        with tempfile.TemporaryDirectory(prefix='v3-wal-') as tmp:
            vault, state = Path(tmp) / 'vault', Path(tmp) / 'state'
            vault.mkdir()
            state.mkdir()
            engine = SyncEngine(vault, state)
            with engine.store._connect() as db:
                self.assertEqual(db.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
                db.execute('CREATE TABLE IF NOT EXISTS t (x)')
            reader = sqlite3.connect(engine.store.database, timeout=0)
            reader.execute('BEGIN')
            reader.execute('SELECT count(*) FROM t').fetchone()  # holds a read snapshot
            try:
                with engine.store._connect() as db:
                    db.execute('INSERT INTO t VALUES (1)')
            finally:
                reader.close()


if __name__ == '__main__':
    unittest.main()
