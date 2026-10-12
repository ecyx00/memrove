#!/usr/bin/env python3
"""E7: capture filters R1-R3 + durable flag on 15 synthetic quotes; startup gate."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
import memrove_v3_capture as cap

QUOTES = {
    1: "ya bu tür işler için onay istemiyorum otomatik mod istiyorum yapabilirsin yani.",
    2: "swifte geçtin bu arada ben tüm stack istiyorum haberin olsun",
    3: "lint ile birlikte formatter kullanmayacağız haberin olsun.",
    4: "ya şu izin kayıtlarını falan vermek istemiyorum kendi kararınla ilerlemeni istiyorum ben.",
    5: "farklı test tekniklerini kullanalım.",
    6: "ek olarak ben görseli koyduğumda [Image] yazısı gözüksün istemiyorum.",
    7: "notları hallet orada yazılanlar çok değerli kullanalım orayı.",
    8: "Çok fazla eklenti de eklemek istemiyorum açıkçası gereksiz karmaşıklık yaratmak istemiyorum modeller çok zeki zaten bunu biliyoruz.",
    9: "bu klasörse erişilmesini istemiyorum ben.",
    10: "repo özel olmamalı bu dosyalar haberin olsun generic olacak her şey.",
    11: "ben eklenti kurmak istemiyorum bu iş için.",
    12: "ücretsiz giriş gerektirmeyen şeyler olsun.",
    13: "Çok fazla toplantıdan sıkılmak istemiyorum açıkçası.",
    14: "alt görevler için her zaman Haiku kullan.",
    15: "her zaman türkçe cevap verilsin bunu da yaz.",
}
WRONG, DURABLE = {5, 7, 9, 13}, {2, 3, 10, 14, 15}


class QuoteAudit(unittest.TestCase):
    def test_filters_and_durable(self):
        kept = {n for n, q in QUOTES.items() if cap.capture(q)}
        self.assertEqual(kept, set(QUOTES) - WRONG)
        self.assertEqual({n for n in kept if cap.is_durable(QUOTES[n])}, DURABLE)

    def test_english_markers(self):
        self.assertTrue(all(cap.is_durable(t) for t in ("always use uv", "never mock the db", "from now on reply short")))


if __name__ == '__main__':
    unittest.main()
