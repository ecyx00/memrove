#!/usr/bin/env python3
"""Past-reference trigger (flag-gated hook line) and the re-brief detector, on synthetic data."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import json
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

    def test_hook_line_only_when_flag_on(self):
        payload = {"prompt": "geçen sefer ne demiştik?", "session_id": "s1"}
        with patch.dict("os.environ", {"MEMROVE_PASTREF": "1"}):
            self.assertEqual(hook.pastref_line(payload), pr.LINE)
            self.assertEqual(hook.pastref_line({"prompt": "testleri çalıştır"}), "")
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEMROVE_PASTREF", None)
            self.assertEqual(hook.pastref_line(payload), "")

    def test_flag_file_enables_line(self):
        payload = {"prompt": "geçen sefer ne demiştik?", "session_id": "s1"}
        import os
        os.environ.pop("MEMROVE_PASTREF", None)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hook.pastref_line(payload, d), "")
            (Path(d) / "pastref.on").write_text("")
            self.assertEqual(hook.pastref_line(payload, d), pr.LINE)
            self.assertEqual(hook.pastref_line({"prompt": "testleri çalıştır"}, d), "")


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
