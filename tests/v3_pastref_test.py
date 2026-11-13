#!/usr/bin/env python3
"""Past-reference trigger (flag-gated hook line) and the re-brief detector, on synthetic data."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "template/.claude/scripts"
sys.path.insert(0, str(SCRIPTS))
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_hook as hook  # noqa: E402
import memrove_v3_pastref as pr  # noqa: E402

os.environ["MEMROVE_DOC_REPOS"] = ""   # the line alone; pointers have their own test


class TriggerTest(unittest.TestCase):
    def test_flags_past_references_in_turkish_and_english(self):
        for t in ("daha önce ne demiştik bu konuda?", "geçen sefer neyi seçmiştik", "neredeyiz şimdi", "hani şu graph işi",
                  "What did we decide about the cache?", "Where were we?", "taska devam et.", "devam", "P0-3 taskına devam et", "kaldığın yerden sürdür", "Parser'ı yazmıştık. Nerede o silindi mi?",
                  "dün konuştuğumuz indexer kararı neydi", "acme bunu yazdırıyordu değil mi?", "P0-7 taskına devam et",
                  "hani bu cache modu gibi projeler inject ediyorlar ya", "daha önce ne karar vermiştik",
                  "geçen hafta yaptığımız ölçüm", "önceki tasklara bak"):
            self.assertTrue(pr.is_past_reference(t), t)

    def test_ignores_plain_tasks(self):
        for t in ("testleri çalıştır ve hataları düzelt", "add a retry to the fetcher", "bu fonksiyonu yeniden adlandır"):
            self.assertFalse(pr.is_past_reference(t), t)

    def test_hook_line_only_with_a_cue_and_once_per_session(self):
        import os
        os.environ.pop("MEMROVE_PASTREF", None)
        payload = {"prompt": "geçen sefer ne demiştik?", "session_id": "s1"}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hook.pastref_line({"prompt": "testleri çalıştır", "session_id": "s1"}, d), "")   # no cue: silent
            first = hook.pastref_line(payload, d)
            self.assertTrue(first.startswith(pr.LINE))
            self.assertLessEqual(len(first), 300)
            self.assertEqual(hook.pastref_line(payload, d), "")                                      # same session: never again
            self.assertEqual(hook.pastref_line({**payload, "session_id": "s2"}, d), first)           # new session: once more
        with patch.dict("os.environ", {"MEMROVE_PASTREF": "0"}):
            self.assertEqual(hook.pastref_line(payload), "")


class RebriefTest(unittest.TestCase):
    def test_finds_planted_repeat_across_sessions_not_within(self):
        ask = "chatsearch passage index için bm25 sıralaması ekle ve zoom ile doğrula lütfen"
        with tempfile.TemporaryDirectory() as d:
            items = [
                {"kind": "user", "text": ask, "date": "2026-10-05T10:00:00+03:00", "src": "claude", "k": "c:A1"},
                {"kind": "user", "text": ask, "date": "2026-10-05T10:05:00+03:00", "src": "claude", "k": "c:A2"},  # same session: not a re-brief
                {"kind": "user", "text": "tamamen alakasız bir istek: fiş görsellerini sentetik üret ve etiketle", "date": "2026-10-06T10:00:00+03:00", "src": "claude", "k": "c:B"},
                {"kind": "user", "text": "chatsearch passage index için bm25 sıralaması ekle, zoom ile doğrula", "date": "2026-10-12T09:00:00+03:00", "src": "claude", "k": "c:C"},
            ]
            chatlog.append(d, items, today="2026-10-12")
            out = Path(d) / "pairs.jsonl"
            res = json.loads(pr.rebrief_command(d, str(out)))
            pairs = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual([(p["new"], p["earlier"]) for p in pairs], [(3, 0)])
        self.assertEqual(res["weekly"]["2026-W42"][1], 1)

    def test_in_session_corrections(self):
        ask = "chatsearch passage index için bm25 sıralaması ekle ve zoom ile doğrula lütfen"
        rows = [
            ("kök nedeni bulup düzeltir misin lütfen bu testi", "2026-10-05T10:00:00+03:00", "c:A"),
            ("hayır yanlış anladın, ben kategori filtresini istemiştim", "2026-10-05T10:02:00+03:00", "c:B"),  # marker
            ("Here is a note offered by a side agent: hayır yanlış, demiştim ben sana", "2026-10-05T10:03:00+03:00", "c:C"),  # not the owner
            ("[Subagent hand-back] yanlış sonuç, hayır", "2026-10-05T10:03:30+03:00", "c:C2"),
            (ask, "2026-10-05T10:04:00+03:00", "c:D"),
            (ask, "2026-10-05T10:06:00+03:00", "c:E"),  # same-session near-repeat
            (ask, "2026-10-05T14:00:00+03:00", "c:F"),  # other session: old pairs, not an in-session correction
        ]
        with tempfile.TemporaryDirectory() as d:
            chatlog.append(d, [{"kind": "user", "text": t, "date": dt, "src": "claude", "k": k} for t, dt, k in rows], today="2026-10-05")
            out = Path(d) / "pairs.jsonl"
            res = json.loads(pr.rebrief_command(d, str(out)))
            sample = [json.loads(l) for l in Path(res["corrections_file"]).read_text().splitlines()]
            new = [json.loads(l)["new"] for l in out.read_text().splitlines()]
        self.assertEqual({(r["how"], r["text"][:5]) for r in sample}, {("marker", "hayır"), ("repeat", ask[:5])})
        self.assertEqual(res["corrections"], 2)
        self.assertEqual(res["corrections_weekly"]["2026-W41"], [5, 1, 0.2, 1])
        self.assertEqual(new, [6])  # old cross-session pair kept

    def test_owner_prompt_filters_and_markers(self):
        self.assertTrue(pr.is_correction("again? why"))
        self.assertFalse(pr.is_correction("run it again please"))
        self.assertFalse(pr.is_correction("try again later"))
        rows = ["wrong file, bunu düzelt lütfen şimdi", "System One modelini kontrol et lütfen", "hook çalışmıyor neden, bakar mısın",
                "nonce kontrolü nasıl çalışıyor anlat", "/Users/ann/x.md bunu oku lütfen", "/clear", "/model opus lütfen", "[System note] hello there"]
        with tempfile.TemporaryDirectory() as d:
            chatlog.append(d, [{"kind": "user", "text": t, "date": "2026-10-05T10:0%d:00+03:00" % n, "src": "claude", "k": "c:%d" % n} for n, t in enumerate(rows)], today="2026-10-05")
            got = pr.owner_prompts(d)
        self.assertEqual([x[3][:6] for x in got], ["wrong ", "System", "hook ç", "nonce ", "/Users"])
        self.assertEqual(pr.corrections(got), {})  # the session-opening "wrong file" is not counted

    def test_probe_sessions_are_not_rebriefs(self):
        ask = "reply with only your exact model id and nothing else please now"
        with tempfile.TemporaryDirectory() as d:
            chatlog.append(d, [
                {"kind": "user", "text": ask, "date": "2026-10-09T10:00:00+03:00", "src": "claude", "k": "c:A", "probe": True},
                {"kind": "user", "text": ask, "date": "2026-10-10T10:00:00+03:00", "src": "claude", "k": "c:B", "probe": True},
            ], today="2026-10-10")
            out = Path(d) / "pairs.jsonl"
            self.assertEqual(json.loads(pr.rebrief_command(d, str(out)))["pairs"], 0)

    def test_is_probe(self):
        self.assertTrue(chatlog.is_probe({"entrypoint": "sdk-cli", "cwd": "/srv/x"}))
        self.assertTrue(chatlog.is_probe({"entrypoint": "cli", "cwd": "/private/tmp/e39-cwd"}))
        self.assertFalse(chatlog.is_probe({"entrypoint": "cli", "cwd": "/srv/x/tmpdir"}))


if __name__ == "__main__":
    unittest.main()
