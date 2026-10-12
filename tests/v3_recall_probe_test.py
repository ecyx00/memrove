#!/usr/bin/env python3
"""E15 recall gate: prompt-side IDF coverage + F5 stemming, replayed on Turkish probes against records mirroring the live store."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import unittest
from v3_recall_test import Base, recall

AI = dict(origin_type='agent_inference', verified_by='llm', confidence=0.5, visibility='internal', updated_at='2026-10-04T00:00:00Z')
LESSONS = {
    'mcp': dict(kind='lesson', project='dev', result='worked', assessment_state='assessed',
                assessments={'lesson_reusable': {'state': 'assessed', 'answers': {'reusable': 'yes'}}},
                title="Sunucu maliyetinin yüksek olması ve gereksiz cron işlerinin kapatılması gerektiği",
                topics=['cost-optimization', 'cron-configuration', 'jobs-management', 'config-optimization'],
                approach="config.yaml'dan 4 cron işini kapatıp log seviyesini düşürme",
                situation="Sunucu maliyetinin yüksek olması ve gereksiz cron işlerinin kapatılması gerektiği",
                why="Maliyeti azaltmak ve logları temizlemek için", text='cron maliyet config'),
    'ok': dict(kind='lesson', project='dev', result='worked', assessment_state='assessed',
               assessments={'lesson_reusable': {'state': 'assessed', 'answers': {'reusable': 'yes'}}},
               title='Sorgu çalıştığında ne kadar süre harcanıp nereye gittiğini bilmeme',
               approach="Boş veritabanında 'select 1' ile ölçüm yapıp profil çıktısını analiz etme",
               situation='Sorgu çalıştığında ne kadar süre harcanıp nereye gittiğini bilmeme',
               why='Bir sorgu çalıştırdığınızda boş bir bağlantı yaklaşık 15 ms harcıyor', text='sorgu ölçüm'),
}
ITEMS = {
    'redis': ('decision', 'Önbellek: varsayılan olarak Redis, bellek içi sadece testler için'),
    'windows': ('rejected', 'Native Windows desteği (macOS merkezli)'),
    'mermaid': ('decision', "Diyagram: README'de Mermaid; kalıcı mimari dokümanlar için PlantUML"),
    'paid': ('rejected', 'Ücretli veya login gerektiren araştırma kaynakları'),
}
FILLER = ['Stripe webhook retry idempotency anahtarları', 'Postgres indeks stratejisi ve migration planı',
          'Dashboard için koyu tema renk paleti', 'Fatura PDF üretimi için şablon motoru seçimi',
          'Docker imajı küçültme ve katman önbelleği', 'Git dal isimlendirme kuralları', 'Haftalık rapor e-postası zamanlaması',
          'Kullanıcı girişi için OAuth akışı', 'Test verisi üretimi için fixture yöntemi']
PROBES = [  # (prompt, cwd project, expected record id or None)
    ('cron işlerini kapatıp maliyet kullanımını azaltmak istiyorum config ayarlarına bakar mısın', 'acme-api', 'mcp'),
    ('önbellek için hangi yöntemi kullanmalıyız redis mi bellek mi', 'Billing v2', 'redis'),
    ('windows desteği ekleyelim mi uygulamaya', 'acme-web', 'windows'),
    ('sorgu süresini ölçmek için boş veritabanında bir test yapalım', 'indexer', 'ok'),
    ('araştırma için ücretli kaynak kullanalım mı yoksa login gerektirmeyenler yeter mi', 'indexer', 'paid'),
    ('lint kullandığım repolar var karşılaştırabiliriz. diğer projelerimde kullandım lint.', 'indexer', None),  # E19: generic words only
    ('bugün hava nasıl', 'indexer', None),
    ('yeni bir özellik için en hızlı nasıl ilerleriz, ekip kuralım mı', 'Billing v2', None),  # generic stems only: never Mermaid
    ('fatura numarası çıkarımında parser retry hatası var bakalım', 'Billing v2', None),
]


class ProbeTest(Base):
    def setUp(self):
        super().setUp()
        for i, f in LESSONS.items():
            self.add(i, **dict(AI, **f))
        for i, (kind, text) in ITEMS.items():
            rec = dict(AI, kind=kind, project='notes-hub', text=text, title=text)
            if i in ('mermaid', 'windows'):  # dateless in the live store: only valid_from
                rec.pop('updated_at')
                rec['valid_from'] = '2026-10-04'
            self.add(i, **rec)
        for n, text in enumerate(FILLER):
            self.add(f'f{n}', **dict(AI, kind='decision', project='other', text=text, title=text))

    def test_probes(self):
        for n, (prompt, project, want) in enumerate(PROBES):
            (self.root / project).mkdir(exist_ok=True)
            (self.root / project / '.git').mkdir(exist_ok=True)
            out = recall.recall(self.store, prompt, str(self.root / project), f'probe{n}')
            if want is None:
                self.assertEqual(out, '', prompt)
            else:
                self.assertIn(self.hint(want), out, prompt)
            self.assertNotIn('Mermaid', out, prompt)

    def test_date_falls_back_and_hint_shows_prompt_words(self):
        out = recall.recall(self.store, 'windows desteği ekleyelim mi uygulamaya', str(self.repo), 's')
        self.assertIn('· 2026-10-04 ·', out)
        self.assertIn('memrove show <id>', out)
        self.assertIn('[windows] rejected: Native Windows desteği (macOS merkezli) (notes-hub · 2026-10-04 · inferred)', out)
        self.assertNotIn('more', out)
        self.assertIn('· 2026-10-04 ·', recall._line({'r': {'kind': 'decision', 'text': 'x', 'valid_from': '2026-10-04'}, 'user': False}, 300))
        self.assertNotIn('·  ·', recall._line({'r': {'kind': 'decision', 'text': 'x', 'last_seen': '2026-10-03'}, 'user': False}, 300))

    def test_cwd_subdirectory_resolves_to_nearest_marker(self):
        from memrove_v3_project import project_root
        pi = self.root / 'notes-hub'
        (pi / 'docs').mkdir(parents=True)
        self.assertEqual(project_root(pi / 'docs'), (pi / 'docs').resolve())  # unmarked: not a project, and no name-match guess
        (pi / '.git').mkdir()
        self.assertEqual(project_root(pi / 'docs'), pi.resolve())
        (pi / 'repo' / '.git').mkdir(parents=True)  # a nearer git root wins
        self.assertEqual(project_root(pi / 'repo'), (pi / 'repo').resolve())
        # recall excludes the record's own project when the shell sits in docs/
        self.assertEqual(recall.recall(self.store, 'windows desteği ekleyelim mi uygulamaya', str(pi / 'docs'), 'd'), '')

    @staticmethod
    def hint(i):
        return f'[{i}] '  # pointer card: the model opens it with `memrove show <id>`

    def test_stemming_matches_turkish_suffixes(self):
        self.assertEqual(recall.terms('serverlarını kullanımını ayarlarına'), ['serve', 'kulla', 'ayarl'])

    def test_prompt_side_gate_blocks_long_record_short_prompt_inverse(self):
        # prompt is mostly unexplained by the record: two words match, six do not
        self.assertEqual(recall.recall(self.store, 'windows uygulama için kahve makinesi bütçe planı toplantı notları', str(self.repo), 's'), '')


if __name__ == '__main__':
    unittest.main()
