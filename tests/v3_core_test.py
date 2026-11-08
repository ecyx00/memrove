#!/usr/bin/env python3
"""core facade: bi-temporal migration, as-of, update closes (never deletes), hard forget survives re-import, feedback row, recall == chatsearch.merged."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
os.environ.setdefault("MEMROVE_NO_LLM", "1")
import memrove_core as core  # noqa: E402
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402
import memrove_v3_graph as graph  # noqa: E402
import memrove_v3_llm as llm  # noqa: E402


def drop_store(state):
    """Delete the store as a user would (rebuild test); leaked connections are closed first, an unlinked WAL db breaks the next connections of this process."""
    import gc
    gc.collect()
    for suffix in ("", "-wal", "-shm"):
        (state / f"memory.sqlite3{suffix}").unlink(missing_ok=True)


class CoreTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = self.t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.state = t / "s"
        self.c = core.Core(t / "v", self.state)

    def test_migration_idempotent_and_backfills(self):
        db = sqlite3.connect(self.state / "memory.sqlite3")
        db.execute("DROP TABLE graph_edges")
        db.execute("CREATE TABLE graph_edges(src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL, why TEXT NOT NULL, prov TEXT NOT NULL, ts TEXT NOT NULL, PRIMARY KEY(src,dst,type))")
        db.execute("INSERT INTO graph_edges VALUES ('a','b','relates','w','{}','2026-01-02')")
        graph.migrate(db)
        one = db.execute("SELECT * FROM graph_edges").fetchall()
        graph.migrate(db)
        self.assertEqual(one, db.execute("SELECT * FROM graph_edges").fetchall())
        self.assertEqual(one[0][6:], ("2026-01-02", None, "2026-01-02"))  # valid_from, valid_to, recorded_at... column order below
        db.close()

    def test_update_closes_old_and_as_of(self):
        old = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        res = self.c.update(old, {"text": "use sqlite for the store", "why": "w", "valid_from": "2026-06-01"})
        self.assertEqual(res["closed"], old)
        self.assertEqual([i for i, _ in self.c.as_of("2026-03-01")], [old])
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [res["id"]])
        with self.c.store._connect() as db:   # nothing deleted
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], 2)
        self.assertEqual(self.c.expand(old)["record"]["valid_to"], "2026-06-01")

    def _edit_and_sync(self, rid):
        with self.c.store._connect() as db:
            src = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])["source"]
        f = self.t / "v" / src
        f.write_text(f.read_text() + "\nedited by hand\n")
        self.c.sync.sync()

    def test_valid_to_survives_file_edit(self):
        old = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        new = self.c.update(old, {"text": "use sqlite for the store", "why": "w", "valid_from": "2026-06-01"})["id"]
        self._edit_and_sync(old)
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [new])
        self.assertEqual(self.c.expand(old)["record"]["valid_to"], "2026-06-01")

    def test_open_record_stays_open_after_file_edit(self):
        rid = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        self._edit_and_sync(rid)
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [rid])
        self.assertIsNone(self.c.expand(rid)["record"].get("valid_to"))

    def test_hard_forget_removes_and_stays_gone(self):
        rid = self.c.ingest({"kind": "lesson", "text": "secret lesson about tokens", "why": "w"})[0]["id"]
        other = self.c.ingest({"kind": "lesson", "text": "other lesson", "why": "w"})[0]["id"]
        with self.c.store._connect() as db:
            db.execute("INSERT INTO graph_edges(src,dst,type,why,prov,ts) VALUES (?,?,?,?,?,?)", (rid, other, "relates", "w", "{}", "2026-01-01"))
        v = sqlite3.connect(self.state / "vectors.sqlite3")
        v.execute("CREATE TABLE vec(h TEXT, model TEXT, v BLOB, PRIMARY KEY(h, model)) WITHOUT ROWID")
        v.execute("CREATE TABLE link(id TEXT, model TEXT, h TEXT, PRIMARY KEY(id, model)) WITHOUT ROWID")
        v.execute("INSERT INTO vec VALUES ('h1','m',x'00')")
        v.execute("INSERT INTO link VALUES (?,?,?)", (rid, "m", "h1"))
        v.commit()
        v.close()
        r = self.c.forget(rid, "hard")
        self.assertEqual(r["deleted"]["graph_edges"], 1)
        self.assertEqual(r["deleted"]["vectors"], 2)
        self.c.ingest({"kind": "lesson", "text": "secret lesson about tokens", "why": "w"})   # re-import of the same content
        with self.c.store._connect() as db:
            self.assertEqual([x[0] for x in db.execute("SELECT id FROM records")], [other])
            self.assertEqual(db.execute("SELECT count(*) FROM forgotten").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT count(*) FROM events WHERE record_id=?", (rid,)).fetchone()[0], 0)

    def test_update_same_text_does_not_close_itself(self):   # B1
        a = self.c.ingest({"kind": "decision", "text": "use postgres for store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        self.c.update(a, {"kind": "decision", "text": "use postgres for store", "why": "w", "valid_from": "2026-06-01"})
        self.assertEqual([i for i, _ in self.c.as_of("2026-07-01")], [a])

    def test_valid_to_survives_store_rebuild(self):   # B2
        old = self.c.ingest({"kind": "decision", "text": "use postgres for the store", "why": "w", "valid_from": "2026-01-01"})[0]["id"]
        new = self.c.update(old, {"text": "use sqlite for the store", "why": "w", "valid_from": "2026-06-01"})["id"]
        self.c.sync.sync()   # the file now carries the close: the hash in the store matches, no hand edit seen
        self.assertEqual(self.c.expand(old)["record"]["revision"], 1)
        drop_store(self.state)
        c2 = core.Core(self.t / "v", self.state)
        c2.sync.sync()
        self.assertEqual(c2.expand(old)["record"]["valid_to"], "2026-06-01")
        self.assertEqual([i for i, _ in c2.as_of("2026-07-01")], [new])

    def test_edge_readd_keeps_closed_interval(self):   # B3
        prov = graph.provenance("user", "user")
        with self.c.store._connect() as db:
            graph.add_edge(db, "a", "b", "relates", "w", prov)
            db.execute("UPDATE graph_edges SET valid_to='2026-03-01'")
            graph.add_edge(db, "a", "b", "relates", "again", prov)
            self.assertEqual(db.execute("SELECT valid_to, why FROM graph_edges").fetchall(), [("2026-03-01", "again")])

    def test_hard_forget_fails_closed_when_forgotten_json_fails(self):
        rid = self.c.ingest({"kind": "lesson", "text": "keep me until the list is written", "why": "w"})[0]["id"]
        with mock.patch.object(core, "keep_forgotten", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.c.forget(rid, "hard")
        with self.c.store._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records WHERE id=?", (rid,)).fetchone()[0], 1)

    def test_malformed_forgotten_json_does_not_stop_open(self):
        (self.state / "forgotten.json").write_text("{}", encoding="utf-8")
        core.Core(self.t / "v", self.state)
        rid = self.c.ingest({"kind": "lesson", "text": "forget me after a bad list", "why": "w"})[0]["id"]
        self.c.forget(rid, "hard")   # keep_forgotten also reads the bad shape as empty
        self.assertIsInstance(json.loads((self.state / "forgotten.json").read_text(encoding="utf-8")), list)

    def test_junk_rows_in_forgotten_json_do_not_block_forget(self):
        (self.state / "forgotten.json").write_text("[1, [], [\"x\"]]", encoding="utf-8")
        core.Core(self.t / "v", self.state)
        rid = self.c.ingest({"kind": "lesson", "text": "forget me after junk rows", "why": "w"})[0]["id"]
        self.c.forget(rid, "hard")
        rows = json.loads((self.state / "forgotten.json").read_text(encoding="utf-8"))
        self.assertTrue(rows and all(isinstance(r, list) and len(r) == 4 for r in rows))

    def test_hard_forget_survives_store_rebuild(self):
        rid = self.c.ingest({"kind": "lesson", "text": "secret lesson about tokens", "why": "w"})[0]["id"]
        self.c.forget(rid, "hard")
        drop_store(self.state)
        c2 = core.Core(self.t / "v", self.state)
        c2.sync.sync()
        with c2.store._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records WHERE id=?", (rid,)).fetchone()[0], 0)

    def test_hard_forget_hides_text_from_every_reader(self):   # B4
        text = "the vault password is swordfish-42"
        rid = self.c.ingest({"kind": "lesson", "text": text.title(), "why": "w"})[0]["id"]   # other case than the chat copies
        ids = [m["i"] for m in chatlog.append(self.state, [{"kind": "user", "text": text, "date": "2026-10-07T10:00:00+00:00"},
                                                           {"kind": "user", "text": "unrelated banana plan", "date": "2026-10-07T10:00:00+00:00"}])]
        cs.sync(str(self.state))
        import memrove_v3_ideacards as ic
        cdb = ic.connect(self.state)
        cdb.execute("INSERT INTO cards(id,card) VALUES (?,?)", (ids[0], "card"))
        cdb.execute("INSERT INTO cards_fts(rowid,card,topics,project) VALUES (?,?,?,?)", (ids[0], "card", "", ""))
        cdb.commit()
        cdb.close()
        import memrove_v3_imports as imports
        imp = imports._db(self.state)
        imp.execute("INSERT INTO x(id,date,kind,text) VALUES ('x-1','2026-10-07','user',?)", (text,))
        imp.execute("INSERT INTO xfts(rowid,folded) VALUES (1,?)", (cs.index_text(text),))
        imp.commit()
        imp.close()
        r = self.c.forget(rid, "hard")
        self.assertEqual((r["deleted"]["hidden"]["chat"], r["deleted"]["hidden"]["imports"]), (1, 1))
        self.assertEqual([m["i"] for m in chatlog.messages(self.state)], [ids[1]])
        self.assertEqual(cs.merged(str(self.state), "swordfish", top=5), [])
        self.assertIsNone(imports.get(self.state, "x-1"))
        self.assertEqual(imports.scored(self.state, None, None, None, 5), [])
        self.assertEqual(self.c.recall("swordfish", k=5), [])
        sdb = sqlite3.connect(self.state / "chatlog" / "search.db")
        self.assertEqual(sdb.execute("SELECT count(*) FROM m WHERE id=?", (ids[0],)).fetchone()[0], 0)
        self.assertEqual(sdb.execute("SELECT count(*) FROM fts WHERE rowid=?", (ids[0],)).fetchone()[0], 0)
        sdb.close()
        self.assertEqual(r["deleted"]["hidden"]["cards"], 1)

    def test_feedback_one_row(self):
        with mock.patch.dict(os.environ, {"DECIDE_HOME": str(self.t / "d")}):
            os.environ.pop("MEMROVE_NO_DECIDE_LOG", None)
            self.c.feedback("obs-1", "wrong", "stale fact")
            lines = (self.t / "d/log/decisions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["answer"], "wrong")
            with self.assertRaises(ValueError):
                self.c.feedback("obs-1", "bogus")

    def test_recall_matches_chatsearch(self):
        chatlog.append(self.state, [{"kind": "user", "text": f"we talked about banana plan {i} and the kiwi budget", "date": "2026-10-07T10:00:00+00:00"} for i in range(5)])
        want = [r[0] for r in cs.merged(str(self.state), "banana plan", top=3)]
        got = self.c.recall("banana plan", k=3)
        self.assertEqual([h["id"] for h in got], want)
        self.assertEqual(got[0]["evidence"][0]["kind"], "msg")


class ReflectTest(unittest.TestCase):
    DAY = "2026-10-07"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = self.t = Path(tmp.name).resolve()
        (t / "v").mkdir()
        self.state = t / "s"
        self.c = core.Core(t / "v", self.state)
        ts = self.DAY + "T10:00:00+00:00"
        self.msgs = [m["i"] for m in chatlog.append(self.state, [
            {"kind": "user", "text": "we use sqlite for the store from now on", "date": ts, "src": "claude"},
            {"kind": "agent", "text": "ok, sqlite then", "date": ts, "src": "claude"},
            {"kind": "user", "text": "codex only text, must never leave the machine", "date": ts, "src": "codex"}])]
        self.sent = []
        patches = [mock.patch.object(llm, "disabled", return_value=False), mock.patch.object(llm, "RUNNER", self.runner)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.answer = {}

    def runner(self, system, payload, vault=None):
        self.sent.append(payload)
        return json.dumps(self.answer), 0.0, None

    def rec(self, text, kind="decision", egress="external_ok", day=None, **kw):
        rid = self.c.ingest({"kind": kind, "text": text, "why": "w", "valid_from": day or self.DAY, **kw})[0]["id"]
        self.c._patch(rid, "test", egress=egress)
        return rid

    def get(self, rid):
        return self.c.expand(rid)["record"]

    def test_owner_preference_is_never_closed_only_proposed(self):   # (a)
        pref = self.rec("subagents on sonnet", kind="preference")
        usr = self.rec("never use tabs", day="2026-09-02", user=True)
        dup = self.rec("never use tabs please")
        opus = chatlog.append(self.state, [{"kind": "user", "text": "subagents on opus for the hard design work", "date": self.DAY + "T11:00:00+00:00", "src": "claude"}])[0]["i"]
        self.answer = {"stale": [{"id": pref, "by": opus, "quote": "subagents on opus for the hard", "why": "opus asked"}],
                       "duplicates": [{"ids": [usr, dup], "text": "never use tabs"}], "unsupported": [{"id": pref}]}
        res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual((res["closed"], res["merged"], res["marked"]), ([], [], []))
        self.assertEqual({p["verdict"] for p in res["proposals"]}, {"stale", "duplicates", "unsupported"})
        for r in (pref, usr):
            self.assertIsNone(self.get(r).get("valid_to"))
            self.assertNotIn("unverified", self.get(r))
        self.assertEqual(len((self.state / "reflect-proposals.jsonl").read_text().splitlines()), len(res["proposals"]))

    def _owner_variants(self):
        origin = self.rec("owner origin only", user=True)
        self.c._patch(origin, "test", verified_by=None)   # a day record is audited unless it is a verbatim quote
        quote = self.rec("use postgres for the store", day="2026-09-01", user=True)
        self.c._patch(quote, "test", origin_type="agent_inference")
        return origin, quote

    def test_every_owner_clause_is_proposal_only(self):
        origin, quote = self._owner_variants()
        self.answer = {"unsupported": [{"id": origin}], "stale": [{"id": quote, "by": self.msgs[0], "quote": "sqlite for the store"}]}
        res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual((res["marked"], res["closed"]), ([], []))
        self.assertEqual({p["id"] for p in res["proposals"]}, {origin, quote})

    def test_duplicate_with_owner_on_either_side_is_a_proposal(self):
        usr, dup = self.rec("never use tabs", day="2026-09-02", user=True), self.rec("never use tabs please")
        for ids in ([usr, dup], [dup, usr]):
            self.answer = {"duplicates": [{"ids": ids, "text": "never use tabs"}]}
            res = self.c.reflect(kind="dream", day=self.DAY)
            self.assertEqual((res["merged"], len(res["proposals"])), ([], 1))
            (self.state / "reflect-usage.json").unlink(missing_ok=True)

    def test_merge_guards(self):
        a, b = self.rec("the store is sqlite"), self.rec("we use sqlite for the store")
        k1, k2 = self.rec("deploy on fridays", kind="lesson"), self.rec("deploy on fridays too")
        p1, p2 = self.rec("ship weekly", project="x"), self.rec("ship weekly please", project="y")
        self.answer = {"duplicates": [{"ids": [a, b], "text": "x" * 500}, {"ids": [k1, k2], "text": "deploy on fridays"}, {"ids": [p1, p2], "text": "ship weekly"}]}
        res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual(res["merged"], [])
        closed, unv = self.rec("old closed note"), self.rec("old unverified note")
        self.c._patch(closed, "test", valid_to=self.DAY)
        self.c._patch(unv, "test", unverified=True)
        self.answer = {"duplicates": [{"ids": [closed, a], "text": "old closed note"}, {"ids": [unv, b], "text": "old unverified note"}]}
        (self.state / "reflect-usage.json").unlink(missing_ok=True)
        self.assertEqual(self.c.reflect(kind="dream", day=self.DAY)["merged"], [])   # a closed or unverified record is not a merge input
        self.answer = {"duplicates": [{"ids": [a, b], "text": "the store is sqlite"}]}
        self.assertEqual(len(self.c.reflect(kind="dream", day=self.DAY)["merged"]), 1)

    def test_token_use_accumulates_and_persists_per_local_day(self):
        self.rec("use sqlite for the store")
        full = self.c.reflect(kind="dream", day=self.DAY, dry=True)["tokens"]
        (self.state / "reflect-usage.json").unlink(missing_ok=True)
        cap = full * 3 // 4   # one call fits, two do not
        res = self.c.reflect(kind="dream", day=self.DAY, cap_tokens=cap)
        self.assertEqual((res["calls"], res["stop"]), (1, "token-cap"))
        self.assertEqual(self.c.reflect(kind="dream", day=self.DAY, cap_tokens=cap)["calls"], 0)   # the usage file remembers
        other = self.c.reflect(kind="dream", day="2026-10-06", cap_tokens=cap)   # another data day, the same local day: same bucket
        self.assertNotEqual(other.get("stop"), None)

    def test_malformed_answer_skips_the_batch(self):
        ok = self.rec("always deploy on fridays")
        for bad in ({"unsupported": [{"id": ["x"]}]}, {"duplicates": [{"ids": [["a"], ["b"]]}]}, {"stale": [{"id": ok, "by": [1]}]}, ["x"]):
            self.answer = bad
            res = self.c.reflect(kind="dream", day=self.DAY)
            self.assertEqual((res["marked"], res["merged"], res["closed"]), ([], [], []))
            self.assertTrue(res["errors"])
            (self.state / "reflect-usage.json").unlink(missing_ok=True)

    def test_one_pass_alone_only_proposes(self):
        bad = self.rec("always deploy on fridays")
        answers = [{"unsupported": [{"id": bad, "why": "x"}]}, {}]
        with mock.patch.object(llm, "RUNNER", lambda system, payload, vault=None: (json.dumps(answers.pop(0)), 0.0, None)):
            res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual((res["marked"], [p["agreed"] for p in res["proposals"]]), ([], [False]))
        self.assertNotIn("unverified", self.get(bad))

    def test_branch_call_has_full_messages_second_pass_reversed_and_model_check(self):
        import memrove_v3_reflect as dream
        long = "we use sqlite " + "x" * 3000
        chatlog.append(self.state, [{"kind": "user", "text": long, "date": self.DAY + "T12:00:00+00:00", "src": "claude"}])
        a, b = self.rec("always deploy on fridays"), self.rec("never deploy on mondays")
        seen = []

        def runner(system, payload, vault=None, model=None):
            seen.append((model, json.loads(payload)))
            return json.dumps({"unsupported": [{"id": a}]} if model != "sonnet" else {}), 0.0, None
        with mock.patch.object(llm, "RUNNER", runner):
            res = self.c.reflect(kind="dream", day=self.DAY, check_model="sonnet")
        self.assertEqual([m for m, _ in seen], [None, None, "sonnet"])
        self.assertIn(long, [m["text"] for m in seen[0][1]["MESSAGES"]])   # no cut
        ids = lambda k: [r["id"] for r in seen[k][1]["RECORDS"]]
        self.assertEqual(ids(0), ids(1)[::-1])   # pass B sees the other order
        self.assertEqual(res["marked"], [a])
        self.assertEqual(res["agree"]["unsupported"], {"AB": [1, 0, 0], "AC": [0, 1, 0], "BC": [0, 1, 0]})
        self.assertEqual(len(res["branches"]), 1)
        with mock.patch.object(dream, "MAX_PAYLOAD", 100), mock.patch.object(llm, "RUNNER", runner):
            self.assertIn("nothing cut", self.c.reflect(kind="dream", day=self.DAY)["errors"][0])   # a too-big branch is skipped, never trimmed

    def test_assign_links_a_record_to_the_branch_it_shares_words_with(self):
        import memrove_v3_reflect as dream
        with mock.patch.object(dream, "BRANCH", 2):
            ts = self.DAY + "T13:00:00+00:00"
            chatlog.append(self.state, [{"kind": "user", "text": "kubernetes ingress certificates expire", "date": ts, "src": "claude"},
                                        {"kind": "agent", "text": "renew them", "date": ts, "src": "claude"}])
            msgs = [m for f in chatlog._files(chatlog.root(self.state)) for m in chatlog.read_file(f) if m["src"] == "claude"]
            tree = dream.branches(msgs)
            k8s, other = self.rec("kubernetes ingress certificates need renewal"), self.rec("zzzz qqqq unrelated words")
            link = dream.assign([(k8s, self.get(k8s)), (other, self.get(other))], tree)
            self.assertEqual(link[k8s], [m for m in msgs if "kubernetes" in m["text"]][0]["i"] // 2)
            self.assertNotIn(other, link)

    def test_settled_branch_is_one_short_line_and_still_zoomable(self):
        import memrove_v3_chattree as tree
        import memrove_v3_reflect as dream
        mem = mock.Mock()
        mem.view.parts = [(1, 0), (1, 1)]
        mem.tree.line = lambda l, i: f"{i << l}+{1 << l}|" + "word " * 80
        (self.state / "reflect-settled.json").write_text(json.dumps({"blocks": [0]}))
        lines = tree.map_lines(self.state, mem)
        self.assertEqual([len(x) < 120 for x in lines], [True, True])   # parts 0..3 sit in block 0
        self.assertIn("zoom 0 2", lines[0])
        mem.view.parts = [(1, 0), (1, 64)]   # the second part lies in block 1, not settled
        self.assertEqual([len(x) < 120 for x in tree.map_lines(self.state, mem)], [True, False])
        res = dream.settle(self.c, "2030-01-01", 3)   # the test day is long past and holds no open linked record
        self.assertIn(0, res["blocks"])
        self.rec("always deploy on fridays")
        res = self.c.reflect(kind="settle", day="2030-01-01")
        self.assertEqual((0 in res["pinned"], 0 in res["blocks"]), (True, False))   # an open linked record keeps the branch in the map

    def test_stale_close_is_dated_by_the_message_not_the_range_end(self):
        rid = self.rec("we plan on postgres for the store")
        mid = chatlog.append(self.state, [{"kind": "user", "text": "forget postgres, sqlite for the store", "date": "2026-10-08T09:00:00+00:00", "src": "claude"}])[0]["i"]
        self.answer = {"stale": [{"id": rid, "by": mid, "quote": "forget postgres, sqlite for the store"}]}
        res = self.c.reflect(kind="dream", day=self.DAY, until="2026-10-09")
        self.assertEqual(res["closed"], [rid])
        self.assertEqual(self.get(rid)["valid_to"], "2026-10-08")

    def test_dry_run_changes_nothing(self):
        bad = self.rec("always deploy on fridays")
        self.answer = {"unsupported": [{"id": bad}]}
        res = self.c.reflect(kind="dream", day=self.DAY, dry=True)
        self.assertEqual(res["marked"], [bad])
        self.assertNotIn("unverified", self.get(bad))

    def test_stale_needs_a_real_contradiction(self):
        old = self.rec("use postgres for the store", day="2026-09-01")
        self.rec("something said today")
        for quote in ("we never said this at all", "from now on", "sqlite for the store"):   # not in the message | no word shared with the record | real
            self.answer = {"stale": [{"id": old, "by": self.msgs[0], "quote": quote}]}
            res = self.c.reflect(kind="dream", day=self.DAY)
            self.assertEqual(res["closed"], [old] if quote == "sqlite for the store" else [])
            (self.state / "reflect-usage.json").unlink(missing_ok=True)

    def test_marks_survive_rebuild_and_sync_keeps_store_state(self):
        bad = self.rec("always deploy on fridays")
        self.answer = {"unsupported": [{"id": bad, "why": "w"}]}
        self.c.reflect(kind="dream", day=self.DAY)
        old = self.rec("use postgres for the store", day="2026-09-01")
        self.c.update(old, {"text": "use sqlite for the store", "valid_from": "2026-09-05"})
        old2, new2 = self.rec("use mysql for the store", day="2026-09-01"), self.rec("use redis for the cache", day="2026-09-01")
        drop_store(self.state)
        c2 = core.Core(self.t / "v", self.state)
        c2.sync.sync()
        self.assertTrue(c2.expand(bad)["record"]["unverified"])
        self.assertEqual(c2.expand(bad)["record"]["reflect"]["verdict"], "unsupported")
        self.assertEqual(c2.expand(old)["record"]["valid_to"], "2026-09-05")
        # the sync keep-lines matter when the file could not carry the field
        with mock.patch.object(core.Core, "_fields_to_file"):
            c2._patch(old2, "close", valid_to="2026-09-06")
            c2._patch(new2, "reflect", unverified=True, reflect={"day": "x"})
        with c2.store._connect() as db:
            srcs = [json.loads(db.execute("SELECT payload FROM records WHERE id=?", (i,)).fetchone()[0])["source"] for i in (old2, new2)]
        for f in srcs:
            (self.t / "v" / f).write_text((self.t / "v" / f).read_text() + "\nedited\n")
        c2.sync.sync()
        self.assertEqual(c2.expand(old2)["record"]["valid_to"], "2026-09-06")
        self.assertTrue(c2.expand(new2)["record"]["unverified"])

    def test_unsupported_is_marked_not_deleted_and_never_a_rule(self):
        bad, ok = self.rec("always deploy on fridays"), self.rec("use sqlite for the store")
        self.answer = {"unsupported": [{"id": bad, "why": "no message says it"}, {"id": "rec:ghost"}]}
        res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual(res["marked"], [bad])
        self.assertTrue(self.get(bad)["unverified"])
        self.assertNotIn("unverified", self.get(ok))
        import memrove_v3_rules as rules
        self.assertFalse(rules._live(self.get(bad)))
        self.assertTrue(rules._live(self.get(ok)))

    def test_duplicates_merge_stale_closes_nothing_deleted(self):
        a, b = self.rec("the store is sqlite", day="2026-10-01"), self.rec("we use sqlite for the store")
        old = self.rec("use postgres for the store", day="2026-09-01")
        keep = self.rec("prefer tabs", day="2026-09-02")
        with self.c.store._connect() as db:
            before = db.execute("SELECT count(*) FROM records").fetchone()[0]
        self.answer = {"duplicates": [{"ids": [a, b], "text": "the store is sqlite"}, {"ids": [a, old], "text": "x" * 500}],
                       "stale": [{"id": old, "by": self.msgs[0], "quote": "sqlite for the store", "why": "replaced"},
                                 {"id": keep, "by": self.msgs[1], "quote": "ok, sqlite then prefer tabs"}]}  # msgs[1] is an agent message: refused
        res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual(len(res["merged"]), 1)
        self.assertEqual(res["closed"], [old])
        with self.c.store._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], before + 1)   # the merged record is new; nothing was removed
        self.assertEqual(self.get(a)["valid_to"], self.DAY)
        self.assertEqual(self.get(b)["valid_to"], self.DAY)
        self.assertEqual(self.get(old)["valid_to"], self.DAY)
        self.assertEqual({i for i, _ in self.c.as_of("2026-10-08")}, {res["merged"][0]["id"], keep})   # keep: the agent message could not close it
        self.assertEqual(self.get(old)["text"].strip(), "use postgres for the store")

    def test_token_cap_stops_and_egress_gate(self):
        self.rec("use sqlite for the store")
        hidden = self.rec("device only note", egress="device_only")
        self.assertEqual(self.c.reflect(kind="dream", day=self.DAY, cap_tokens=5)["stop"], "token-cap")
        self.assertEqual(self.sent, [])
        res = self.c.reflect(kind="dream", day=self.DAY)
        self.assertEqual(res["calls"], 2)   # two passes
        self.assertNotIn(hidden, self.sent[0])
        self.assertNotIn("codex only", self.sent[0])
        self.assertIn("sqlite", self.sent[0])

    def test_agent_filter_and_migration_keeps_counts(self):
        mine = self.rec("use sqlite for the store")
        cx = self.rec("codex decided to use sqlite")
        self.c._patch(cx, "test", imported_from="codex-session")
        with self.c.store._connect() as db:   # a store from before the agent field
            n = db.execute("SELECT count(*) FROM records").fetchone()[0]
            for rid in (mine, cx):
                p = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])
                p.pop("agent", None)
                db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(p), rid))
        for _ in range(2):   # idempotent
            c = core.Core(self.t / "v", self.state)
        with c.store._connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], n)
            self.assertEqual(db.execute("SELECT count(*) FROM records WHERE json_extract(payload,'$.agent') IS NULL").fetchone()[0], 0)
        self.assertEqual([i for i, _ in c.as_of(self.DAY, agent="codex")], [cx])
        self.assertEqual([i for i, _ in c.as_of(self.DAY, agent="claude-code")], [mine])
        hits = c.recall("sqlite store", agent="codex")
        self.assertTrue(all(h["origin"] == "codex" for h in hits))
        self.assertNotIn(str(self.msgs[0]), {str(h["id"]) for h in hits})  # a claude message is not a codex hit


if __name__ == "__main__":
    unittest.main()
