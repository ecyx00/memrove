"""Single-flight hook worker and lock-tolerant remember, under real concurrency."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor

from v3_package_helpers import isolated_env

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "template/.claude/scripts"
HOOK = SCRIPTS / "memrove_v3_hook.py"


class FlightTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="memrove-flight-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.vault, self.state = self.root / "vault", self.root / "state"
        self.vault.mkdir()
        self.env = isolated_env(self.root / "home")
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))

    def test_twenty_hooks_one_heavy_worker_no_event_lost(self):
        env = dict(self.env)
        env.pop("MEMROVE_V3_NO_SPAWN")
        env["MEMROVE_V3_TEST_HOLD"] = "3"  # old code ignores this; its workers still overlap for seconds under 20 starts
        peak, stop = [0], threading.Event()

        def watch():  # heavy = a --worker alive for 2 s or more; a loser exits in well under that
            while not stop.is_set():
                out = subprocess.run(["ps", "-axo", "etime=,command="], capture_output=True, text=True).stdout
                n = 0
                for line in out.splitlines():
                    t, _, cmd = line.strip().partition(" ")
                    if str(self.state) in cmd and "--worker" in cmd:
                        secs = [int(x) for x in t.replace("-", ":").split(":")]
                        n += sum(v * m for v, m in zip(reversed(secs), (1, 60, 3600, 86400))) >= 2
                peak[0] = max(peak[0], n)
                time.sleep(0.05)
        watcher = threading.Thread(target=watch)
        watcher.start()

        def fire(i):
            payload = {"hook_event_name": "Stop", "session_id": f"s{i}", "event_id": f"e{i}", "cwd": str(self.vault)}
            r = subprocess.run([sys.executable, str(HOOK), "--vault", str(self.vault), "--state", str(self.state),
                                "--harness", "claude"], input=json.dumps(payload), text=True, capture_output=True,
                               cwd=self.vault, env=env, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
        with ThreadPoolExecutor(20) as pool:
            list(pool.map(fire, range(20)))
        deadline = time.time() + 90
        while time.time() < deadline:  # workers are detached: wait until none is left
            out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
            if not any(str(self.state) in l and "--worker" in l for l in out.splitlines()):
                break
            time.sleep(0.2)
        stop.set()
        watcher.join()
        queue = list((self.state / "hook-queue").glob("*.json"))
        done = list((self.state / "hook-done").glob("*.json"))
        self.assertEqual((len(queue), len(done)), (0, 20))
        self.assertLessEqual(peak[0], 2, peak[0])

    def test_remember_survives_concurrent_writers(self):
        import memrove_v3 as core
        from memrove_v3_sync import SyncEngine
        from memrove_v3_context import remember
        engine = SyncEngine(self.vault, self.state)
        engine.sync()
        db_path = str(engine.store.database)
        real = sqlite3.connect  # the live 30 s wait was outlasted; force 1 s so the test is quick and fails on old code too
        patcher = unittest.mock.patch.object(core.sqlite3, "connect", lambda *a, **k: real(*a, **{**k, "timeout": 1}) if a and a[0] == db_path else real(*a, **k))
        patcher.start()
        self.addCleanup(patcher.stop)

        def writer(i):
            time.sleep(i * 2)
            db = real(db_path, timeout=30)
            db.execute("BEGIN IMMEDIATE")
            time.sleep(3)  # lock held 0-3, 2-5, 4-7, 6-9 s: continuously busy for ~9 s
            db.rollback()
            db.close()
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(writer, i) for i in range(4)]
            time.sleep(0.5)
            result = remember(engine, "note", "memrove lock fix concurrency test", user=True)
            for f in futures:
                f.result()
        self.assertEqual(result["status"], "created")

    def test_remember_after_locked_write_ends_created_with_decide_log(self):
        """Lock hits after the file landed; the between-try sync is locked too, then succeeds: still "created" + decide-log."""
        import memrove_v3_context as ctx
        import memrove_v3_decide_log as dl
        from memrove_v3_sync import SyncEngine
        engine = SyncEngine(self.vault, self.state)
        engine.sync()
        real_create, real_sync, calls = engine.note_create, engine.sync, {"c": 0, "s": 0}

        def create(*a, **k):
            calls["c"] += 1
            out = real_create(*a, **k)
            if calls["c"] == 1:
                raise sqlite3.OperationalError("database is locked")  # file written, readback lost
            return out

        def sync(*a, **k):
            calls["s"] += 1
            if calls["s"] == 1:
                raise sqlite3.OperationalError("database is locked")
            if calls["s"] == 2:
                raise RuntimeError("not an OperationalError")
            return real_sync(*a, **k)
        engine.note_create, engine.sync = create, sync
        logged = []
        with unittest.mock.patch("time.sleep", lambda s: None), \
                unittest.mock.patch.object(dl, "log_capture_kind", lambda *a, **k: logged.append(a)):
            result = ctx.remember(engine, "note", "retried remember lands created", user=True)
        self.assertEqual(result["status"], "created")
        self.assertEqual(len(logged), 1)

    def test_retry_budget_checked_before_each_try(self):
        import memrove_v3_context as ctx
        clock, calls = [0.0], [0]

        def fn():
            calls[0] += 1
            raise sqlite3.OperationalError("database is locked")

        class S:
            def sync(self, full=False):
                clock[0] += 500  # a slow between-try sync outlasts the budget
        with unittest.mock.patch("time.monotonic", lambda: clock[0]), unittest.mock.patch("time.sleep", lambda s: None):
            with self.assertRaises(sqlite3.OperationalError):
                ctx._retry_locked(S(), fn, budget=120.0)
        self.assertEqual(calls[0], 1)

    def test_worker_quarantines_poison_event_and_drains_the_rest(self):
        import memrove_v3_hook as hook
        q = self.state / "hook-queue"
        q.mkdir(parents=True)
        (q / "a-bad.json").write_text("{not json")
        for i in range(3):
            (q / f"b{i}.json").write_text(json.dumps({"event": "Stop", "session": f"s{i}", "cwd": str(self.vault), "at": i}))
        out = hook.run_worker(self.vault, self.state)
        self.assertEqual(list(q.glob("*.json")), [], out)
        self.assertEqual(len(list((self.state / "hook-done").glob("*.json"))), 3)
        self.assertEqual([p.name for p in (self.state / "hook-bad").glob("*.json")], ["a-bad.json"])


if __name__ == "__main__":
    unittest.main()
