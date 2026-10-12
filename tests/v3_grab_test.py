#!/usr/bin/env python3
"""Grab: card path and id, URL scrub, never-fetch guards, process into snapshot + catalog + note, [grab:] tag, pinned Defuddle."""
# Note: the Turkish strings in this file are deliberate test inputs for the Turkish-text detectors (cues, stemming, case folding).
import json
import os
from datetime import timedelta
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'template/.claude/scripts'))
sys.dont_write_bytecode = True
os.environ['MEMROVE_GRAB_HEADLESS'] = '1'  # popup tests never show a window on the screen
TK_SKIP = unittest.skipIf(sys.platform == 'darwin' and not os.environ.get('MEMROVE_GRAB_TK_MAC'),
                          'Tk makes Python a Dock app on macOS; the hotkey path never uses it. MEMROVE_GRAB_TK_MAC=1 runs these')
os.environ['MEMROVE_GRAB_NO_PROCESS'] = '1'  # popup saves never start a real background process
import memrove_v3_research as research  # noqa: E402
import memrove_v3_sources as sources  # noqa: E402
import memrove_v3_wiki as wiki  # noqa: E402
import memrove_v3_grab as grab  # noqa: E402

CARDS = 'wiki/_sources/grab'
ID = re.compile(r'^\d{4}-\d{2}-\d{2}-[a-z0-9]+(-[a-z0-9]+)*$')
ARTICLE = 'This is an article text. ' * 20


def fake_fetch(vault, kind, ident, slug_name):
    dest = sources.snap_dir(vault, slug_name)
    dest.mkdir(parents=True)
    (dest / 'README.md').write_text('# ' + ident + '\n', encoding='utf-8')
    return sources._finish(dest, {'kind': kind, 'url': 'https://github.com/' + ident, 'sha': 'a' * 40})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v3-grab-')
        self.addCleanup(self.tmp.cleanup)
        self.vault = Path(self.tmp.name) / 'Sample Memrové'
        self.vault.mkdir()


class CardTest(Base):
    def test_card_path_and_id_format(self):
        first = grab.capture(self.vault, url='https://example.com/a-post', why='reason one')
        self.assertEqual(first['path'], CARDS + '/' + first['id'] + '.md')
        self.assertTrue((self.vault / first['path']).is_file())
        self.assertRegex(first['id'], ID)
        self.assertTrue(first['id'].startswith(grab.now().strftime('%Y-%m-%d') + '-'))
        # same slug twice: the duplicate suffix stays inside the slug part, never after a time or extension
        a = grab.capture(self.vault, text='same title', title='Same title')
        b = grab.capture(self.vault, text='other text', title='Same title')
        self.assertNotEqual(a['id'], b['id'])
        for card_id in (a['id'], b['id']):
            self.assertRegex(card_id, ID)
        self.assertTrue(b['id'].endswith('-2'))
        for card in grab.cards(self.vault):
            self.assertRegex(card['id'], ID)

    def test_capture_dedupes_and_requires_something(self):
        first = grab.capture(self.vault, url='https://www.youtube.com/watch?v=abc', why='limit observation')
        again = grab.capture(self.vault, url='https://youtu.be/abc', why='second look')
        self.assertEqual((first['status'], again['status'], again['id']), ('captured', 'existing_card_added', first['id']))
        card = grab.cards(self.vault)[0]
        self.assertIn('second look', card['body'])
        self.assertEqual(grab.pending(self.vault), 1)
        with self.assertRaises(ValueError):
            grab.capture(self.vault, why='yalniz reason')

    def test_private_kinds_and_files(self):
        source = Path(self.tmp.name) / 'not.txt'
        source.write_text('lesson in the file', encoding='utf-8')
        result = grab.capture(self.vault, files=[source], why='belge')
        self.assertEqual(grab.find_card(self.vault, result['id'])['meta']['visibility'], 'private')
        self.assertTrue((self.vault / CARDS / 'files/not.txt').is_file())
        text = grab.capture(self.vault, text='pano metni')
        self.assertEqual(grab.find_card(self.vault, text['id'])['meta']['visibility'], 'private')

    def test_canonical_kind_slug_vtt(self):
        self.assertEqual(grab.canonical('https://youtu.be/abc123?t=40'), grab.canonical('https://www.youtube.com/watch?v=abc123&list=x'))
        self.assertEqual(grab.kind_of('https://x.com/a/status/1'), 'x')
        self.assertEqual(grab.kind_of(None, app='Mail'), 'mail')
        self.assertEqual(grab.slug('Kısa tut, İyi düşün: ağ'), 'kisa-tut-iyi-dusun-ag')
        text = grab.vtt_text('WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello <c>world</c>\n\n00:00:40.000 --> 00:00:42.000\nnew\n')
        self.assertIn('**0:00** hello world', text)
        self.assertIn('**0:40** new', text)

    def test_hotkey_and_cli_plumbing(self):
        self.assertEqual(grab.parse_hotkey('ctrl+alt+b'), (['ctrl', 'alt'], 'b'))
        self.assertEqual(grab.parse_hotkey('cmd+"'), (['cmd'], '"'))
        for bad in ('', 'b', 'hyper+b', 'cmd+'):
            with self.assertRaises(ValueError):
                grab.parse_hotkey(bad)
        if sys.platform == 'darwin':
            self.assertEqual(grab.mac_hotkey('ctrl+alt+b')[:2], (11, 0x1800))
        grab.capture(self.vault, url='https://ornek.com/a')
        for argv in (['--vault', str(self.vault), '--json', 'list'], ['list', '--vault', str(self.vault), '--json']):
            with patch('sys.stdout', new_callable=__import__('io').StringIO) as out:
                grab.main(argv)
            self.assertEqual(json.loads(out.getvalue())['pending'], 1)

    def test_bad_hotkey_writes_nothing_and_foreign_listener_stays(self):
        state = Path(self.tmp.name) / 'state'
        with patch.object(grab.sys, 'platform', 'darwin'), self.assertRaises(ValueError):
            grab.install(self.vault, state, spec='hyper+b')
        self.assertFalse((self.vault / CARDS).exists() or state.exists())
        import plistlib
        agent = Path(self.tmp.name) / 'agent.plist'
        agent.write_bytes(plistlib.dumps({'ProgramArguments': ['python3', 'x.py', 'listen', '--vault', '/other/vault']}))
        with patch.object(grab, '_launch_agent', return_value=agent), patch.object(grab.sys, 'platform', 'darwin'), \
                patch.object(grab.subprocess, 'run') as run:
            result = grab.uninstall(self.vault, state)
        run.assert_not_called()
        self.assertTrue(agent.exists())
        self.assertNotIn('LaunchAgent', result['removed'])


class PrivacyTest(Base):
    def test_scrub_url_keeps_only_allow_listed_keys(self):
        dirty = 'https://user:pw@Example.com:8443/post/1?token=SECRET&utm_source=x&session=abc#frag'
        self.assertEqual(grab.scrub_url(dirty), 'https://example.com:8443/post/1')
        self.assertEqual(grab.scrub_url('https://www.youtube.com/watch?v=abc&t=30&si=TRACK&list=x&token=S'),
                         'https://www.youtube.com/watch?v=abc&t=30')
        self.assertEqual(grab.scrub_url('https://youtu.be/abc?t=9&si=TRACK'), 'https://youtu.be/abc?t=9')
        self.assertEqual(grab.scrub_url('https://arxiv.org/abs/2405.15793?context=cs'), 'https://arxiv.org/abs/2405.15793')
        self.assertEqual(grab.scrub_url('file:///etc/passwd'), '')

    def test_capture_stores_the_clean_url_only(self):
        result = grab.capture(self.vault, url='https://example.com/p?access_token=SECRET123&x=1', why='n')
        text = (self.vault / result['path']).read_text(encoding='utf-8')
        self.assertNotIn('SECRET123', text)
        self.assertIn('https://example.com/p', text)

    def test_never_fetch_hosts(self):
        for url in ('https://mail.google.com/mail/u/0/#inbox/1', 'https://docs.google.com/document/d/x/edit', 'https://drive.google.com/f',
                    'https://www.notion.so/page-1', 'https://team.slack.com/archives/C1', 'http://localhost:3000/a', 'http://127.0.0.1/a',
                    'http://192.168.1.5/admin', 'http://10.0.0.7/x', 'http://169.254.169.254/latest', 'http://[::1]/a', 'http://nas.local/f',
                    'http://intranet/wiki', 'http://2130706433/', 'ftp://example.com/a'):
            self.assertTrue(grab.never_fetch(url), url)
        for url in ('https://github.com/kepano/defuddle', 'https://arxiv.org/abs/2405.15793', 'https://example.com/a', 'http://8.8.8.8/'):
            self.assertFalse(grab.never_fetch(url), url)
        host_card = grab.capture(self.vault, url='https://docs.google.com/document/d/x/edit', why='n')
        self.assertEqual(grab.find_card(self.vault, host_card['id'])['meta']['visibility'], 'private')

    def _process_with_net_spies(self, card_id):
        spies = {'defuddle': MagicMock(return_value=None), 'stdlib': MagicMock(return_value=None), 'fetch': MagicMock(),
                 'run': MagicMock(return_value=(1, '', '')), 'urlopen': MagicMock(side_effect=AssertionError('network call'))}
        with patch.object(grab, 'via_defuddle', spies['defuddle']), patch.object(grab, 'via_stdlib', spies['stdlib']), \
                patch.object(grab, '_run', spies['run']), patch('urllib.request.urlopen', spies['urlopen']):
            result = grab.process(self.vault, [card_id], fetch=spies['fetch'])
        return result, spies

    def test_never_fetch_host_makes_no_network_call(self):
        card = grab.capture(self.vault, url='http://192.168.1.5/admin?token=T', text='secilen satir', why='n')
        result, spies = self._process_with_net_spies(card['id'])
        for name, spy in spies.items():
            spy.assert_not_called()
        self.assertEqual(result['cards'][0]['method'], 'selection')  # only the owner's selection was used

    def test_private_card_never_fetches_even_on_a_public_host(self):
        card = grab.capture(self.vault, url='https://github.com/kepano/defuddle', why='n')
        found = grab.find_card(self.vault, card['id'])
        meta = dict(found['meta'], visibility='private')
        grab.write_card(found['path'], meta, found['body'] + '\n\n## Selection\n\n> my own words')
        result, spies = self._process_with_net_spies(card['id'])
        for name, spy in spies.items():
            spy.assert_not_called()
        self.assertEqual(result['cards'][0]['status'], 'processed')

    def test_defuddle_is_the_pinned_local_install_never_npx(self):
        found = dict(grab.tools(), defuddle=None, node='/usr/bin/node')
        with patch.object(grab, '_run') as run:
            self.assertIsNone(grab.via_defuddle('https://example.com/a', found))
            run.assert_not_called()
        found['defuddle'] = '/tools/node_modules/.bin/defuddle'
        out = json.dumps({'content': 'x' * 100, 'title': 'T'})
        with patch.object(grab, '_run', return_value=(0, out, '')) as run:
            grab.via_defuddle('https://example.com/a', found)
        self.assertEqual(run.call_args[0][0][:2], ['/tools/node_modules/.bin/defuddle', 'parse'])
        self.assertNotIn('npx', json.dumps(run.call_args[0][0]))
        self.assertEqual(grab.DEFUDDLE_VERSION, '0.19.4')

    def test_install_defuddle_uses_ci_without_scripts(self):
        tools = Path(self.tmp.name) / 'defuddle'
        calls = []
        def fake_run(cmd, timeout, cwd=None):
            calls.append(cmd)
            if cmd[1] == 'install':
                (tools / 'package-lock.json').write_text('{}')
            return 0, '', ''
        with patch.object(grab, '_which', return_value='/usr/bin/npm'), patch.object(grab, '_run', fake_run):
            grab.install_defuddle(tools)
        self.assertEqual(json.loads((tools / 'package.json').read_text())['dependencies'], {'defuddle': '0.19.4'})
        self.assertIn(['/usr/bin/npm', 'ci', '--ignore-scripts'], calls)
        self.assertTrue(all('--ignore-scripts' in c for c in calls))


class IsleTest(Base):
    def test_process_article_makes_snapshot_catalog_and_note(self):
        card = grab.capture(self.vault, url='https://example.com/essay?utm_source=x', why='reason: "full" this, é è')
        notes = []
        info = {'title': 'An Essay'}
        def fake_defuddle(url, found, info_out=None):
            self.assertEqual(url, 'https://example.com/essay')  # scrubbed before the fetch
            info_out.update(info)
            return ARTICLE
        with patch.object(grab, 'via_defuddle', fake_defuddle):
            result = grab.process(self.vault, note=lambda text, why: notes.append((text, why)))
        entry = result['cards'][0]
        self.assertEqual(entry['status'], 'processed')
        snap = sources.snap_dir(self.vault, 'an-essay')
        self.assertIn('This is an article', (snap / 'page.md').read_text(encoding='utf-8'))
        meta = json.loads((snap / sources.SNAP_META).read_text(encoding='utf-8'))
        self.assertEqual((meta['card'], meta['url']), (card['id'], 'https://example.com/essay'))
        cat = json.loads((self.vault / 'wiki/_sources/catalog.json').read_text(encoding='utf-8'))
        item = next(i for i in cat['items'] if i.get('card') == card['id'])
        self.assertEqual((item['origin'], item['url'], item['reason'], item['snapshot']),
                         ('grab', 'https://example.com/essay', 'reason: "full" this, é è', 'an-essay'))
        self.assertEqual(research.check_catalog(cat, {}), [])
        self.assertEqual(notes[0][0], 'owner captured An Essay because reason: "full" this, é è')
        self.assertIn(card['id'], notes[0][1])
        saved = grab.find_card(self.vault, card['id'])
        self.assertEqual((saved['meta']['status'], saved['meta']['snapshot']), ('processed', 'an-essay'))
        self.assertTrue((self.vault / CARDS / '.raw' / (card['id'] + '.md')).is_file())
        # a second run does not touch an processed card
        self.assertEqual(grab.process(self.vault)['processed'], 0)

    def test_process_repo_goes_through_sources_fetch(self):
        card = grab.capture(self.vault, url='https://github.com/kepano/defuddle?tab=readme', why='test capture')
        fetch = MagicMock(side_effect=fake_fetch)
        notes = []
        with patch.object(grab, 'via_defuddle') as defuddle:
            grab.process(self.vault, fetch=fetch, note=lambda t, w: notes.append(t))
            defuddle.assert_not_called()
        fetch.assert_called_once_with(self.vault, 'repo', 'kepano/defuddle', 'defuddle')
        cat = json.loads((self.vault / 'wiki/_sources/catalog.json').read_text(encoding='utf-8'))
        item = cat['items'][0]
        self.assertEqual((item['id'], item['kind'], item['origin'], item['card'], item['reason']),
                         ('github:kepano/defuddle', 'repo', 'grab', card['id'], 'test capture'))
        self.assertEqual(json.loads((sources.snap_dir(self.vault, 'defuddle') / sources.SNAP_META).read_text())['card'], card['id'])
        self.assertEqual(notes, ['owner captured kepano/defuddle because test capture'])

    def test_process_paper_goes_through_sources_fetch(self):
        grab.capture(self.vault, url='https://arxiv.org/pdf/2405.15793v3', why='n')
        fetch = MagicMock(side_effect=lambda v, k, i, s: fake_fetch(v, k, i, s))
        grab.process(self.vault, fetch=fetch)
        self.assertEqual(fetch.call_args[0][1:], ('paper', '2405.15793', '2405-15793'))

    def test_failed_card_is_marked_error_and_queue_continues(self):
        bad = grab.capture(self.vault, url='https://example.com/gone', why='n')
        good = grab.capture(self.vault, text='a paragraph note', why='n')
        with patch.object(grab, 'via_defuddle', return_value=None), patch.object(grab, 'via_stdlib', return_value=None):
            result = grab.process(self.vault)
        states = {e['id']: e['status'] for e in result['cards']}
        self.assertEqual(states, {bad['id']: 'error', good['id']: 'processed'})

    def test_private_text_card_note_omits_the_title(self):
        grab.capture(self.vault, text='mail body', title='Re: salary raise', why='keep', app='Mail')
        notes = []
        grab.process(self.vault, note=lambda t, w: notes.append(t))
        self.assertEqual(notes, ['owner captured a private mail because keep'])

    def test_catalog_rebuild_keeps_captures(self):
        research.add_capture(self.vault, {'id': 'grab:c1', 'kind': 'article', 'title': 'T', 'url': 'https://e.com/a', 'card': '2026-10-10-c1',
                                          'reason': 'n', 'snapshot': 't'})
        research.add_capture(self.vault, {'id': 'github:a/b', 'kind': 'repo', 'title': 'b', 'url': None, 'card': '2026-10-10-b', 'reason': 'm',
                                          'snapshot': 'b'})
        old = json.loads((self.vault / 'wiki/_sources/catalog.json').read_text())['items']
        report_item = {'id': 'github:a/b', 'kind': 'repo', 'title': 'b', 'url': 'https://github.com/a/b', 'mentions': [{'x': 1}]}
        kept = {i['id']: i for i in research.keep_captures(old, [report_item])}
        self.assertEqual(set(kept), {'grab:c1', 'github:a/b'})
        self.assertEqual(kept['github:a/b']['captures'][0]['reason'], 'm')
        self.assertEqual(kept['github:a/b']['mentions'], [{'x': 1}])


FIXTURE_PNG = ROOT / 'tests/fixtures/grab-ocr.png'


class ImageTest(Base):
    def test_hotkey_rejects_shottrs_key(self):
        for spec in ('cmd+"', '⌘"', 'cmd+shift+\''):
            with patch.object(grab.sys, 'platform', 'darwin'), self.assertRaises(ValueError) as caught:
                grab.install(self.vault, Path(self.tmp.name) / 'state', spec=spec, defuddle=False)
            self.assertIn('Shottr', str(caught.exception))
        self.assertEqual(grab.mac_hotkey('ctrl+alt+b')[:2], (11, 0x1800)) if sys.platform == 'darwin' else None
        self.assertEqual(grab.DEFAULT_HOTKEY, 'ctrl+alt+b')
        self.assertFalse((self.vault / CARDS).exists())

    def test_clipboard_image_is_read_through_osascript(self):
        png = FIXTURE_PNG.read_bytes()
        with patch.object(grab.sys, 'platform', 'darwin'), \
                patch.object(grab, '_osascript', return_value='«data PNGf' + png.hex().upper() + '»') as osa:
            self.assertEqual(grab.clipboard_image(), png)
        self.assertIn('«class PNGf»', osa.call_args[0][0])
        with patch.object(grab.sys, 'platform', 'darwin'), patch.object(grab, '_osascript', return_value=None):
            self.assertIsNone(grab.clipboard_image())

    def test_image_card_is_private_image_with_the_png_beside_it(self):
        png = FIXTURE_PNG.read_bytes()
        result = grab.capture(self.vault, image=png, why='screen goruntusu')
        card = grab.find_card(self.vault, result['id'])
        self.assertRegex(result['id'], ID)
        self.assertEqual((card['meta']['source_kind'], card['meta']['visibility']), ('image', 'private'))
        self.assertEqual((self.vault / CARDS / 'files' / (result['id'] + '.png')).read_bytes(), png)
        self.assertIn('[[' + CARDS + '/files/' + result['id'] + '.png]]', card['body'])

    def test_image_card_never_fetches_even_with_urls_in_the_ocr_text(self):
        result = grab.capture(self.vault, image=FIXTURE_PNG.read_bytes(), why='n')
        spies = {'defuddle': MagicMock(), 'stdlib': MagicMock(), 'fetch': MagicMock(), 'urlopen': MagicMock(side_effect=AssertionError('net'))}
        with patch.object(grab, 'ocr_text', return_value='see https://github.com/kepano/defuddle and https://evil.example/x'), \
                patch.object(grab, 'via_defuddle', spies['defuddle']), patch.object(grab, 'via_stdlib', spies['stdlib']), \
                patch('urllib.request.urlopen', spies['urlopen']):
            out = grab.process(self.vault, [result['id']], fetch=spies['fetch'], note=lambda t, w: None)
        for spy in spies.values():
            spy.assert_not_called()
        self.assertEqual(out['cards'][0]['status'], 'processed')

    @unittest.skipUnless(sys.platform == 'darwin' and grab._which('swiftc'), 'needs macOS Vision and swiftc')
    def test_process_reads_a_png_with_vision_and_snapshots_image_plus_text(self):
        binary = Path(self.tmp.name) / 'ocr'
        notes = []
        with patch.object(grab, 'OCR_BIN', binary):
            result = grab.capture(self.vault, image=FIXTURE_PNG.read_bytes(), why='ocr testi')
            out = grab.process(self.vault, note=lambda t, w: notes.append(t))
        self.assertEqual(out['cards'][0]['status'], 'processed')
        snap = sources.snap_dir(self.vault, 'clipboard-image')
        # 'Yakala' is text rendered inside the fixture PNG, not a command name
        page = (snap / 'page.md').read_text(encoding='utf-8')
        self.assertIn('Yakala OCR fixture 4217', page)
        self.assertEqual((snap / 'image.png').read_bytes(), FIXTURE_PNG.read_bytes())
        self.assertIn('Yakala OCR fixture 4217', (self.vault / CARDS / '.raw' / (result['id'] + '.md')).read_text(encoding='utf-8'))
        cat = json.loads((self.vault / 'wiki/_sources/catalog.json').read_text(encoding='utf-8'))
        self.assertEqual((cat['items'][0]['kind'], cat['items'][0]['visibility'], cat['items'][0]['url']), ('image', 'private', None))
        self.assertEqual(notes, ['owner captured a private image because ocr testi'])


@TK_SKIP
class FreshImageTest(Base):
    PAGE = {'url': 'https://example.com/in-front', 'title': 'In front', 'app': 'Google Chrome'}

    def _popup(self, tab_presses=0):
        """Run the real Tk popup headlessly: press Tab N times, then Return."""
        import tkinter as tk
        png = FIXTURE_PNG.read_bytes()

        def drive(root):
            root.update()
            for _ in range(tab_presses):
                root.event_generate('<Tab>')
                root.update()
            root.event_generate('<Return>')
            root.update()
            for pending in root.tk.splitlist(root.tk.call('after', 'info')):
                root.after_cancel(pending)
            root.destroy()
        context = dict(self.PAGE, pano_taze=True)  # a screenshot taken just before must not replace the page
        with patch.object(grab, 'clipboard_image', return_value=png), patch.object(tk.Tk, 'mainloop', drive), \
                patch.object(grab.sys, 'platform', 'darwin'), patch.object(grab, '_style_mac'):
            try:
                return grab.popup(self.vault, context)
            except tk.TclError as exc:
                self.skipTest('no display: ' + str(exc))

    def _kind(self, result):
        return grab.find_card(self.vault, result['id'])['meta']['source_kind']

    def test_choose_source_rules(self):
        self.assertEqual(grab.choose_source(b'png', True), 'other')     # the page in front leads, even with a new screenshot
        self.assertEqual(grab.choose_source(b'png', False), 'image')    # nothing else to save
        self.assertEqual(grab.choose_source(None, True), 'other')

    def test_clipboard_image_does_not_replace_the_browser_page(self):
        self.assertEqual(self._kind(self._popup()), 'article')

    def test_tab_switches_to_the_clipboard_image(self):
        self.assertEqual(self._kind(self._popup(tab_presses=1)), 'image')
        self.assertEqual(self._kind(self._popup(tab_presses=2)), 'article')

    def test_fallback_dialog_follows_the_same_rule(self):
        png = FIXTURE_PNG.read_bytes()
        with patch.object(grab, 'clipboard_image', return_value=png), patch.object(grab, '_osascript', return_value='reason'):
            page = grab._popup_fallback(self.vault, dict(self.PAGE, pano_taze=True))
            alone = grab._popup_fallback(self.vault, {})
        self.assertEqual((self._kind(page), self._kind(alone)), ('article', 'image'))


class WikiTagTest(Base):
    def _errors(self, tag):
        root = Path(self.tmp.name) / 'root'
        root.mkdir(exist_ok=True)
        page = Path(self.tmp.name) / 'p.md'
        page.write_text(f'---\nproject: demo\nroot: {root}\n---\n# K\n- The owner word. {tag}\n', encoding='utf-8')
        return [e for e in wiki.check(page, None, Path(self.tmp.name), self.vault) if 'unresolved tag' in e]

    def test_grab_tag_resolves_only_for_an_existing_card(self):
        card = grab.capture(self.vault, url='https://example.com/a-post', why='n')
        self.assertEqual(self._errors(f"[grab:{card['id']}]"), [])
        self.assertEqual(self._errors('[grab:2026-10-10-missing]'), ['unresolved tag [grab:2026-10-10-missing]'])
        self.assertEqual(self._errors('[grab:../../catalog]'), ['unresolved tag [grab:../../catalog]'])
        self.assertEqual(self._errors(f"[grab:{card['id']}.md]"), [f"unresolved tag [grab:{card['id']}.md]"])


class IntentionTest(Base):
    def processed(self, name, days=0, **kw):
        card = grab.capture(self.vault, text=ARTICLE, title=name, why='reason ' + name)
        with patch.object(grab, 'via_defuddle'):
            grab.process(self.vault, [card['id']], **kw)
        self.age(card['id'], days)
        return card['id']

    def age(self, card_id, days):
        from datetime import timedelta
        card = grab.find_card(self.vault, card_id)
        meta = dict(card['meta'])
        meta['captured'] = (grab.now() - timedelta(days=days)).isoformat(timespec='seconds')
        grab.write_card(card['path'], meta, card['body'])

    def report(self):
        return grab.intention(self.vault)

    def test_target_counts_as_used_and_done_takes_it(self):
        card = grab.capture(self.vault, text='x', title='Goal one', why='w')
        grab.finish(self.vault, card['id'], target='indexer')
        meta = grab.find_card(self.vault, card['id'])['meta']
        self.assertEqual(meta['target'], 'indexer')
        self.age(card['id'], 20)
        self.assertEqual(self.report()['pending_processed'], [])

    def test_wiki_cite_and_record_reference_count_but_self_note_does_not(self):
        a, b, c = (self.processed(n, 20) for n in ('Cited', 'Recorded', 'Unused'))
        page = self.vault / 'wiki/_concepts/x.md'
        page.parent.mkdir(parents=True)
        page.write_text('claim [grab:' + a + ']\n', encoding='utf-8')
        rec = self.vault / 'notes/observed/note/r.md'
        rec.parent.mkdir(parents=True)
        rec.write_text('---\n{}\n---\nsee ' + b + '\ngrab card ' + c + '\n', encoding='utf-8')
        stale = [r['id'] for r in self.report()['pending_processed']]
        self.assertEqual(stale, [c])

    def test_pending_ages(self):
        old = self.processed('Old', 15)
        fresh = self.processed('Fresh', 2)
        raw = grab.capture(self.vault, text='y', title='Raw old', why='w')
        self.age(raw['id'], 4)
        new_raw = grab.capture(self.vault, text='z', title='Raw new', why='w')
        rep = self.report()
        self.assertEqual([(r['id'], r['days']) for r in rep['pending_processed']], [(old, 15)])
        self.assertEqual([r['id'] for r in rep['pending_unprocessed']], [raw['id']])
        self.assertNotIn(fresh, str(rep['pending_processed']))
        self.assertNotIn(new_raw['id'], str(rep))

    def test_archive_needs_one_showing_and_moves_only_unused(self):
        old = self.processed('Old one', 20)
        cited = self.processed('Cited one', 20)
        page = self.vault / 'wiki/_concepts/x.md'
        page.parent.mkdir(parents=True)
        page.write_text('[grab:' + cited + ']', encoding='utf-8')
        self.assertEqual(grab.archive(self.vault)['archived'], [])  # never shown yet
        self.report()  # the showing
        done = grab.archive(self.vault)
        self.assertEqual(done['archived'], [old])
        inbox = self.vault / CARDS
        self.assertFalse((inbox / (old + '.md')).exists())
        self.assertTrue((inbox / 'archive' / (old + '.md')).is_file())
        self.assertTrue((inbox / (cited + '.md')).is_file())
        cat = json.loads((self.vault / 'wiki/_sources/catalog.json').read_text(encoding='utf-8'))
        item = next(i for i in cat['items'] if i.get('card') == old)
        self.assertEqual(item['card_path'], CARDS + '/archive/' + old + '.md')
        self.assertTrue(sources.snap_dir(self.vault, item['snapshot']).is_dir())

    def test_cited_card_is_never_archived_even_after_being_shown(self):
        card = self.processed('Late cite', 30)
        self.report()  # shown while unused
        page = self.vault / 'wiki/_concepts/x.md'
        page.parent.mkdir(parents=True)
        page.write_text('[grab:' + card + ']', encoding='utf-8')
        self.assertEqual(grab.archive(self.vault)['archived'], [])
        self.assertEqual(wiki_errors(self.vault, card), [])

    def test_metric_counts_archived_cards_in_the_denominator(self):
        used = self.processed('Used', 20)
        grab.finish(self.vault, used, target='panel')
        card = grab.find_card(self.vault, used)  # the goal was set on day 5 of 20
        meta = dict(card['meta'])
        meta['used_at'] = (grab.datetime.fromisoformat(meta['captured']) + grab.timedelta(days=5)).isoformat()
        grab.write_card(card['path'], meta, card['body'])
        self.processed('Idle', 20)
        self.report()
        grab.archive(self.vault)
        metric = self.report()['metric']
        self.assertEqual((metric['processed'], metric['used'], metric['rate']), (2, 1, 0.5))

    def test_use_date_comes_from_the_citing_page_not_from_status_runs(self):
        from datetime import timedelta
        cited = self.processed('Cited late', 30)
        stamp = grab.datetime.fromisoformat(grab.find_card(self.vault, cited)['meta']['captured'])
        page = self.vault / 'wiki/_concepts/x.md'
        page.parent.mkdir(parents=True)
        page.write_text('---\nupdated: ' + (stamp + timedelta(days=5)).date().isoformat() + '\n---\n[grab:' + cited + ']', encoding='utf-8')
        metric = self.report()['metric']  # first status run, 30 days after capture
        self.assertEqual((metric['processed'], metric['used']), (1, 1))
        page.write_text('---\nupdated: ' + (stamp + timedelta(days=25)).date().isoformat() + '\n---\n[grab:' + cited + ']', encoding='utf-8')
        self.assertEqual(self.report()['metric']['used'], 0)

    def test_cli_target_and_json(self):
        card = grab.capture(self.vault, text='x', title='Cli goal', why='w')
        with patch.object(grab, 'via_defuddle'), patch('sys.stdout'):
            grab.main(['process', card['id'], '--target', 'memrove', '--vault', str(self.vault)], vault=self.vault)
        self.assertEqual(grab.find_card(self.vault, card['id'])['meta']['target'], 'memrove')


def wiki_errors(vault, card_id):
    page = Path(vault) / 'wiki/_concepts/x.md'
    ok = (Path(vault) / CARDS / (card_id + '.md')).is_file()  # Panel resolves only here
    return [] if ok and ('[grab:' + card_id + ']') in page.read_text(encoding='utf-8') else ['unresolved']


class NativePanelTest(Base):
    """The native panel only draws and asks; Python answers hint/create/save/cancel and does every write."""

    def fake_panel(self, *requests):
        """An executable that speaks the panel protocol: reads the model, sends each request, records model + replies."""
        log = Path(self.tmp.name) / 'panel-log.json'
        script = Path(self.tmp.name) / 'FakePanel'
        script.write_text('#!' + sys.executable + '\nimport json, sys\nmodel = json.loads(sys.stdin.readline())\nreplies = []\n'
                          'for request in ' + repr(list(requests)) + ':\n'
                          '    print(json.dumps(request), flush=True)\n'
                          '    replies.append(json.loads(sys.stdin.readline()))\n'
                          'open(' + repr(str(log)) + ', "w").write(json.dumps({"model": model, "replies": replies}))\n',
                          encoding='utf-8')
        script.chmod(0o755)
        return script, log

    def test_model_offers_the_tweet_with_a_clean_title_and_a_symbol(self):
        ctx = {'url': 'https://x.com/a/status/1', 'title': "(2) X'te Ada: \"kisa not\" / X", 'app': 'Google Chrome'}
        model, raw = grab.panel_model(self.vault, ctx, '', None)
        self.assertEqual(model['source'], 'url')
        self.assertEqual(model['sources'][0]['symbol'], 'at')
        self.assertEqual(model['sources'][0]['title'], 'Ada \u2014 kisa not')
        self.assertEqual(model['sources'][0]['detail'], 'x.com')
        self.assertEqual(raw['url'], 'https://x.com/a/status/1')

    def test_model_without_a_source_is_a_thought(self):
        model, _raw = grab.panel_model(self.vault, {'app': 'Notes'}, '', None)
        self.assertIsNone(model['source'])
        self.assertEqual(model['sources'], [])

    def test_the_link_leads_and_a_screenshot_is_the_other_source(self):
        # pano_taze: the installed listener still reports a screenshot taken in the last 90 s; it must not decide any more.
        ctx = {'url': 'https://example.com/a', 'title': 'A', 'pano_taze': True}
        model, _raw = grab.panel_model(self.vault, ctx, '', b'png' * 100)
        self.assertEqual([item['id'] for item in model['sources']], ['url', 'image'])

    def test_thought_is_saved_verbatim_and_a_new_group_is_created(self):
        typed = '  raw thought: Open Panel search with ⌘K \n'
        binary, log = self.fake_panel({'op': 'create', 'name': 'Fikirler'},
                                      {'op': 'save', 'text': typed, 'groups': ['Fikirler'], 'open': True, 'source': None})
        result = grab.native_popup(self.vault, {'app': 'Notes'}, binary, clip='', image=None)
        card = grab.find_card(self.vault, result['id'])
        self.assertEqual(card['meta']['source_kind'], 'thought')
        self.assertEqual(card['body'].split('\n\n' + grab.LINKS_OPEN)[0], typed.rstrip('\n'))
        self.assertIn('[[memrove/concepts/fikirler|Fikirler]]', card['body'])
        self.assertEqual(card['meta']['groups'], ['Fikirler'])
        self.assertTrue(grab.is_open(card['meta']))
        replies = json.loads(log.read_text())['replies']
        self.assertEqual(replies[0]['name'], 'Fikirler')
        self.assertEqual(replies[1], {'ok': True})

    def test_source_save_uses_the_link_and_the_typed_reason(self):
        binary, _log = self.fake_panel({'op': 'save', 'text': 'okunacak', 'groups': [], 'open': False, 'source': 'url'})
        ctx = {'url': 'https://example.com/yazi', 'title': 'Yazi', 'app': 'Safari'}
        result = grab.native_popup(self.vault, ctx, binary, clip='', image=None)
        card = grab.find_card(self.vault, result['id'])
        self.assertEqual(card['meta']['url'], 'https://example.com/yazi')
        self.assertIn('okunacak', card['body'])

    def test_empty_reason_on_a_source_keeps_the_groups_carried_reason(self):
        grab.capture(self.vault, url='https://example.com/b', why='skill arastirmasi', groups=['skills'])
        binary, _log = self.fake_panel({'op': 'save', 'text': '', 'groups': ['skills'], 'open': False, 'source': 'url'})
        result = grab.native_popup(self.vault, {'url': 'https://example.com/c', 'title': 'C'}, binary, clip='', image=None)
        self.assertIn('skill arastirmasi', grab.find_card(self.vault, result['id'])['body'])

    def test_a_panel_that_dies_before_asking_falls_back(self):
        dead = Path(self.tmp.name) / 'DeadPanel'
        dead.write_text('#!/bin/sh\nexit 2\n', encoding='utf-8')
        dead.chmod(0o755)
        self.assertIsNone(grab.native_popup(self.vault, {}, dead, clip='', image=None))

    def test_cancel_writes_nothing(self):
        binary, _log = self.fake_panel({'op': 'cancel'})
        result = grab.native_popup(self.vault, {}, binary, clip='', image=None)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(grab.cards(self.vault), [])

    def test_empty_thought_is_refused_and_the_panel_stays_open(self):
        binary, log = self.fake_panel({'op': 'save', 'text': '   ', 'groups': [], 'open': False, 'source': None}, {'op': 'cancel'})
        grab.native_popup(self.vault, {}, binary, clip='', image=None)
        self.assertIn('error', json.loads(log.read_text())['replies'][0])
        self.assertEqual(grab.cards(self.vault), [])

    def test_hint_is_the_reason_of_that_group_set(self):
        grab.capture(self.vault, url='https://example.com/b', why='skill arastirmasi', groups=['skills'])
        binary, log = self.fake_panel({'op': 'hint', 'groups': ['skills']}, {'op': 'cancel'})
        grab.native_popup(self.vault, {}, binary, clip='', image=None)
        self.assertEqual(json.loads(log.read_text())['replies'][0]['hint'], 'skill arastirmasi')

    @TK_SKIP
    def test_headless_popup_never_starts_the_native_panel(self):
        with patch.object(grab, 'ensure_panel', side_effect=AssertionError('native panel in headless mode')):
            with patch.object(grab, '_popup_fallback', return_value={'status': 'cancelled'}), patch.dict(sys.modules, {'tkinter': None}):
                grab.popup(self.vault, {})


class MenuBarAgentTest(Base):
    SWIFT = ROOT / 'template/.claude/scripts/memrove_v3_grab_panel.swift'

    def test_launch_agent_runs_the_swift_agent_and_copies_both_sources(self):
        import plistlib
        state, agent = Path(self.tmp.name) / 'state', Path(self.tmp.name) / 'agent.plist'

        def fake_run(command, **kwargs):
            if '-o' in command:
                Path(command[command.index('-o') + 1]).write_bytes(b'bin')
            return types.SimpleNamespace(returncode=0, stdout=b'', stderr='')
        with patch.object(grab.sys, 'platform', 'darwin'), patch.object(grab, '_launch_agent', return_value=agent), \
                patch.object(grab, '_which', return_value='/usr/bin/swiftc'), patch.object(grab, '_listener_ok', return_value=True), \
                patch.object(grab.subprocess, 'run', side_effect=fake_run):
            grab.install(self.vault, state, spec='ctrl+alt+b', defuddle=False)
        plist = plistlib.loads(agent.read_bytes())
        arguments = plist['ProgramArguments']
        self.assertEqual(Path(arguments[0]).name, 'MemroveGrab')
        self.assertEqual(arguments[1], 'agent')
        self.assertNotIn('dinle', arguments)
        self.assertIs(plist['KeepAlive'], True)
        folder = state / 'grab'
        self.assertEqual((folder / grab.PANEL_SOURCE).read_bytes(), self.SWIFT.read_bytes())
        self.assertEqual((folder / 'memrove_v3_grab.py').read_bytes(), Path(grab.__file__).read_bytes())

    def test_swift_agent_is_an_accessory_menu_bar_app_with_the_hotkey(self):
        source = self.SWIFT.read_text(encoding='utf-8')
        self.assertIn('final class Agent', source)
        self.assertIn('NSStatusItem', source)
        self.assertIn('RegisterEventHotKey', source)
        self.assertLess(source.index('app.setActivationPolicy(.accessory)'), source.index('app.run()'))

    def test_hotkey_path_imports_no_tkinter_and_python_has_no_listener(self):
        self.assertFalse(hasattr(grab, 'listen_mac'))
        with patch.dict(os.environ, {}, clear=False) as env:
            for name in ('MEMROVE_GRAB_HEADLESS', 'MEMROVE_GRAB_SNAPSHOT', 'MEMROVE_GRAB_TK'):
                env.pop(name, None)
            with patch.object(grab, 'ensure_panel', return_value=None), patch.object(grab, '_popup_fallback', return_value={'status': 'cancelled'}) as fallback, \
                    patch.dict(sys.modules, {'tkinter': None}):
                self.assertEqual(grab.popup(self.vault, {}), {'status': 'cancelled'})
        fallback.assert_called_once()

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS only')
    def test_mac_never_opens_tk_even_when_headless_envs_are_set(self):
        with patch.dict(os.environ, {'MEMROVE_GRAB_HEADLESS': '1', 'MEMROVE_GRAB_TK': '1'}), \
                patch.object(grab, 'ensure_panel', return_value=None), patch.object(grab, '_popup_fallback', return_value={'status': 'cancelled'}) as fallback, \
                patch.dict(sys.modules, {'tkinter': None}):
            self.assertEqual(grab.popup(self.vault, {}), {'status': 'cancelled'})
        fallback.assert_called_once()



if __name__ == '__main__':
    unittest.main()


@TK_SKIP
class GroupTest(Base):
    PAGE = {'url': 'https://x.com/a/status/1', 'title': 'A tweet', 'app': 'Safari'}

    def _popup(self, script):
        """Open the real Tk popup headlessly and run script(root, api); the script saves with Return itself."""
        import tkinter as tk

        def drive(root):
            root.update()
            script(root, root.grab)
            root.update()
            for pending in root.tk.splitlist(root.tk.call('after', 'info')):
                root.after_cancel(pending)
            root.destroy()
        with patch.object(grab, 'clipboard_image', return_value=getattr(self, 'IMAGE', None)), patch.object(tk.Tk, 'mainloop', drive), \
                patch.object(grab.sys, 'platform', 'darwin'), patch.object(grab, '_style_mac'):
            try:
                return grab.popup(self.vault, dict(self.PAGE))
            except tk.TclError as exc:
                self.skipTest('no display: ' + str(exc))

    def _meta(self, result):
        return grab.find_card(self.vault, result['id'])['meta']

    def _later(self, minutes):
        return patch.object(grab, 'now', return_value=grab.now() + timedelta(minutes=minutes))

    def test_multi_group_front_matter_and_filter(self):
        result = grab.capture(self.vault, text='a note', why='reason', groups=['Rust', 'startup'])
        card = grab.find_card(self.vault, result['id'])
        self.assertEqual(card['meta']['groups'], ['Rust', 'startup'])
        self.assertIn('groups: ["Rust", "startup"]', card['path'].read_text(encoding='utf-8'))
        self.assertEqual(len(grab.listing(self.vault, group='startup')['cards']), 1)
        self.assertEqual(grab.listing(self.vault, group='none')['cards'], [])

    def test_selection_and_reason_carry_over_for_60_minutes_only(self):
        grab.capture(self.vault, text='a note', why='Rust fikirleri', groups=['Rust', 'Startup'])
        with self._later(59):
            self.assertEqual(grab.active_groups(self.vault), ['Rust', 'Startup'])
            self.assertEqual(grab.set_reason(self.vault, ['Startup', 'rust']), 'Rust fikirleri')  # same set, any order
            self.assertEqual(grab.set_reason(self.vault, ['Rust']), '')  # another set
        with self._later(61):
            self.assertEqual(grab.active_groups(self.vault), [])
            self.assertEqual(grab.set_reason(self.vault, ['Rust', 'Startup']), '')

    def test_enter_on_empty_field_saves_the_sets_reason_and_groups(self):
        grab.capture(self.vault, text='first', why='Rust fikirleri', groups=['Rust', 'Startup'])
        result = self._popup(lambda root, api: root.event_generate('<Return>'))
        meta = self._meta(result)
        self.assertEqual(meta['groups'], ['Rust', 'Startup'])
        self.assertIn('Rust fikirleri', grab.find_card(self.vault, result['id'])['body'])

    def test_empty_reason_is_fine_with_groups_and_typed_reason_is_kept_as_written(self):
        grab.capture(self.vault, text='first', why='old reason', groups=['Rust'])
        with self._later(90):  # the reason is stale, the group still gives context
            result = grab.capture(self.vault, url='https://example.com/p', groups=['Rust'])
        self.assertEqual(self._meta(result)['groups'], ['Rust'])
        typed = grab.capture(self.vault, url='https://example.com/q', why='new reason', groups=['Rust'])
        self.assertIn('new reason', grab.find_card(self.vault, typed['id'])['body'])
        self.assertEqual(grab.set_reason(self.vault, ['Rust']), 'new reason')

    def test_token_add_and_remove(self):
        grab.group_create(self.vault, 'Rust')
        grab.group_create(self.vault, 'Design')

        def script(root, api):
            api.toggle('Rust')
            api.toggle('Design')
            self.assertEqual(api.selected, ['Rust', 'Design'])
            api.remove('Rust')
            root.event_generate('<Return>')
        self.assertEqual(self._meta(self._popup(script))['groups'], ['Design'])

    def test_search_filters_as_you_type_and_keyboard_toggles(self):
        for name in ('Rust', 'Design', 'Startup'):
            grab.group_create(self.vault, name)

        def script(root, api):
            api.open()
            root.update()
            api.search_entry.focus_force()
            api.search.set('ru')
            self.assertEqual(api.rows(), [('g', 'Rust')])
            api.search.set('')
            api.search_entry.event_generate('<Down>')
            api.search_entry.event_generate('<space>')  # empty search: space toggles the highlighted row
            self.assertEqual(api.selected, [api.rows()[1][1]])
            api.search.set('ti')
            api.search_entry.event_generate('<Return>')  # Return toggles the highlighted match
            self.assertEqual(len(api.selected), 2)
        self._popup(script)

    def test_backspace_in_empty_search_removes_the_last_token(self):
        grab.group_create(self.vault, 'Rust')

        def script(root, api):
            api.toggle('Rust')
            api.open()
            root.update()
            api.search_entry.focus_force()
            api.search_entry.event_generate('<BackSpace>')
            self.assertEqual(api.selected, [])
        self._popup(script)

    def test_plus_creates_with_return_and_with_the_button(self):
        def script(root, api):
            api.open()
            api.enter_create()
            root.update()
            self.assertTrue(api.hint_shown())  # placeholder 'New group name' while the name is empty
            api.name.set('   ')
            self.assertNotEqual(api.button_fill(), grab.PALETTE['accent'])  # dimmed while empty
            api.name.set('Kitaplar')
            self.assertEqual(api.button_fill(), grab.PALETTE['accent'])
            api.name_entry.focus_force()
            api.name_entry.event_generate('<Return>')
            self.assertEqual(api.selected, ['Kitaplar'])
            api.enter_create()
            api.name.set('Filmler')
            api.create_button.event_generate('<Button-1>')
            self.assertEqual(api.selected, ['Kitaplar', 'Filmler'])
            self.assertEqual(api.known[0], 'Filmler')  # new group on top
        self._popup(script)
        self.assertEqual(set(grab.recent_groups(self.vault)), {'Kitaplar', 'Filmler'})

    def test_duplicate_name_selects_the_existing_group_turkish_aware(self):
        grab.group_create(self.vault, 'ışık')

        def script(root, api):
            api.open()
            api.create('  IŞIK ')  # I -> ı, so this is the same group
            self.assertEqual(api.selected, ['ışık'])
            self.assertEqual(api.msg.get(), 'Already exists, selected')
            api.create('işik')  # i is not ı: a different group
            self.assertEqual(api.selected, ['ışık', 'işik'])
        self._popup(script)

    def test_no_match_row_creates_a_group(self):
        grab.group_create(self.vault, 'Rust')

        def script(root, api):
            api.open()
            root.update()
            api.search_entry.focus_force()
            api.search.set('Kozmoloji')
            self.assertEqual(api.rows(), [('new', 'Kozmoloji')])
            api.search_entry.event_generate('<Return>')
            self.assertEqual(api.selected, ['Kozmoloji'])
        self._popup(script)
        self.assertIn('Kozmoloji', grab.recent_groups(self.vault))

    def test_recent_first_then_alphabetical(self):
        order = ('Z', 'Y', 'X', 'W', 'V', 'U', 'B', 'A')  # A is the most recently used
        for name in order:
            grab.group_create(self.vault, name)
        store = grab._group_store(self.vault)
        for index, name in enumerate(order):
            store['groups'][name] = '2026-01-01T00:%02d:00+00:00' % index
        grab._group_write(self.vault, store)
        self.assertEqual(grab.recent_groups(self.vault), ['A', 'B', 'U', 'V', 'W', 'X', 'Y', 'Z'])

    def test_option_arrows_move_by_word_and_cmd_arrows_to_line_ends(self):
        def script(root, api):
            entry = api.entry
            entry.focus_force()
            root.update()
            entry.event_generate('<Key>', keysym='b')  # typing replaces the placeholder
            entry.delete(0, 'end')
            entry.insert(0, 'bir iki  üç')
            entry.icursor('end')
            entry.event_generate('<Option-Left>')
            self.assertEqual(entry.index('insert'), 9)
            entry.event_generate('<Option-Left>')
            self.assertEqual(entry.index('insert'), 4)
            entry.event_generate('<Option-Right>')
            self.assertEqual(entry.index('insert'), 7)
            entry.event_generate('<Command-Left>')
            self.assertEqual(entry.index('insert'), 0)
            entry.event_generate('<Command-Right>')
            self.assertEqual(entry.index('insert'), len('bir iki  üç'))
            entry.event_generate('<Option-BackSpace>')
            self.assertEqual(entry.get(), 'bir iki  ')
            entry.event_generate('<Shift-Option-Left>')
            self.assertEqual(entry.selection_get(), 'iki  ')
            entry.event_generate('<Command-BackSpace>')  # a selection goes first, as in a native field
            self.assertEqual(entry.get(), 'bir ')
            entry.event_generate('<Command-BackSpace>')
            self.assertEqual(entry.get(), '')
        self._popup(script)


    def test_panel_fits_its_rows_and_scrolls_after_eight(self):
        for index in range(12):
            grab.group_create(self.vault, 'G%02d' % index)

        def script(root, api):
            api.open()
            root.update()
            tall = api.panel.winfo_height()
            api.search.set('G01')  # one row
            root.update()
            self.assertEqual(api.panel.winfo_height(), api.panel.winfo_reqheight())
            one = api.panel.winfo_height()
            self.assertLess(one, tall)
            self.assertLessEqual(tall - one, 8 * grab._u(28) - grab._u(28) + 4)  # 8 rows at most, nothing beyond the rows
            api.search.set('')
            root.update()
            self.assertEqual(api.panel.winfo_height(), tall)  # 12 groups: capped at 8 visible rows
        self._popup(script)

    def test_caret_ends_up_after_programmatic_text(self):
        def script(root, api):
            api.open()
            api.search.set('Kozmoloji')
            self.assertEqual(api.search_entry.index('insert'), len('Kozmoloji'))
            api.enter_create()
            self.assertEqual(api.name_entry.index('insert'), len('Kozmoloji'))  # prefilled from the search text
        self._popup(script)

    def test_no_list_highlight_while_creating(self):
        grab.group_create(self.vault, 'Rust')

        def script(root, api):
            api.open()
            lst = api.panel.nametowidget('list')
            fills = lambda: [lst.itemcget(i, 'fill') for i in lst.find_all() if lst.type(i) == 'polygon']
            self.assertIn(grab.PALETTE['panel_sel'], fills())
            api.enter_create()
            self.assertNotIn(grab.PALETTE['panel_sel'], fills())
        self._popup(script)


class PaletteTest(unittest.TestCase):
    @staticmethod
    def _ratio(first, second):
        def lum(hex_color):
            channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        high, low = sorted((lum(first), lum(second)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    def test_panel_roles_and_contrast(self):
        P = grab.PALETTE
        self.assertEqual((P['bg'], P['card'], P['accent'], P['accent_text']), ('#222221', '#292928', '#F0EFE9', '#222221'))
        for text in ('text', 'muted', 'soft'):
            for surface in ('bg', 'card', 'panel'):
                self.assertGreaterEqual(self._ratio(P[text], P[surface]), 4.5, (text, surface))
        self.assertGreaterEqual(self._ratio(P['text'], P['bg']), 7)  # Panel's textPrimary floor
        self.assertGreaterEqual(self._ratio(P['accent_text'], P['accent']), 4.5)  # button text on accentFill
        self.assertGreaterEqual(self._ratio(P['faint'], P['bg']), 4.5)  # the placeholder is text too
        for signal in ('ok', 'warn', 'error'):
            self.assertGreaterEqual(self._ratio(P[signal], P['bg']), 4.5, signal)
        self.assertGreaterEqual(self._ratio(P['text'], P['panel_sel']), 4.5)  # list row text on the selection wash
        self.assertNotIn(P['accent'].upper(), ('#F5C84C', '#D97757', '#ECEBE6'))


@TK_SKIP
class TypeScaleTest(Base):
    PAGE = GroupTest.PAGE
    _popup = GroupTest._popup

    def setUp(self):
        super().setUp()
        grab.type_s.cache_clear()
        self.addCleanup(grab.type_s.cache_clear)

    def _with_size(self, stored, script, image=None):
        grab.type_s.cache_clear()
        self.IMAGE = image
        with patch.object(grab, '_defaults_read', return_value=stored):
            return self._popup(script)

    @staticmethod
    def _click(canvas, box):
        x1, y1, x2, y2 = box
        canvas.event_generate('<Button-1>', x=int((x1 + x2) / 2), y=int((y1 + y2) / 2))
        canvas.update()

    def test_no_literal_font_sizes_outside_the_table(self):
        source = (ROOT / 'template/.claude/scripts/memrove_v3_grab.py').read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'_(?:font|mono)\(\s*\d+', source), [])
        self.assertNotIn('BASE_SIZE', source)

    def test_scale_is_panels_own_and_never_below_11(self):
        for stored, s in ((None, 13), ('13', 13), ('11', 11), ('16', 16), ('40', 16), ('2', 11)):
            grab.type_s.cache_clear()
            with patch.object(grab, '_defaults_read', return_value=stored):
                self.assertEqual(grab.type_s(), s, stored)
                self.assertEqual(grab._size('composer'), s + 4)
                self.assertEqual(grab._size('body'), s)
                self.assertEqual(grab._size('secondary'), max(11, s - 1))
                self.assertEqual(grab._size('caption'), max(11, s - 2))
                self.assertTrue(all(grab._size(role) >= 11 for role in grab.TYPE_ROLES))

    def test_popup_sizes_follow_the_scale(self):
        for stored, s in (('13', 13), ('16', 16)):
            sizes = {}

            def script(root, api):
                import tkinter.font as tkfont
                for item in api.canvas.find_all():
                    if api.canvas.type(item) == 'text':
                        sizes[api.canvas.itemcget(item, 'text')] = tkfont.Font(root=root, font=api.canvas.itemcget(item, 'font')).actual('size')
                sizes['composer'] = tkfont.Font(root=root, font=api.entry.cget('font')).actual('size')
            self._with_size(stored, script)
            self.assertEqual(sizes['composer'], s + 4)
            self.assertEqual(sizes['Grab · Safari'], max(11, s - 1))
            self.assertEqual(sizes['+ Group'], max(11, s - 1))
            self.assertEqual(sizes['A tweet'], s)
            self.assertGreaterEqual(min(sizes.values()), 11, sizes)

    def test_group_row_is_a_compact_pill_not_a_full_width_field(self):
        seen = {}

        def script(root, api):
            seen['text'] = [api.canvas.itemcget(i, 'text') for i in api.canvas.find_all() if api.canvas.type(i) == 'text']
            seen['tags'] = {tag for i in api.canvas.find_all() for tag in api.canvas.gettags(i)}
            x1, y1, x2, y2 = api.pill()
            seen['pill'] = (x2 - x1, y2 - y1, int(api.canvas['width']))
        self._with_size('13', script)
        self.assertNotIn('Group add', seen['text'])
        self.assertNotIn('tokbg', seen['tags'])
        width, height, full = seen['pill']
        self.assertLess(width, full / 6)
        self.assertLessEqual(height, 26)

    def test_clicking_the_pill_opens_the_popover_without_growing_the_window(self):
        grab.group_create(self.vault, 'skills')
        seen = {}

        def script(root, api):
            before = root.winfo_height()
            self._click(api.canvas, api.pill())
            seen['open'] = api.ui['open']
            seen['mapped'] = api.panel.winfo_ismapped()
            seen['grew'] = root.winfo_height() - before
            seen['below_pill'] = api.panel.winfo_y() >= api.pill()[3]
        self._with_size('13', script)
        self.assertTrue(seen['open'] and seen['mapped'])
        self.assertEqual(seen['grew'], 0)
        self.assertTrue(seen['below_pill'])

    def test_clicking_the_footer_keycap_opens_it_and_keycaps_do_not_overlap(self):
        grab.group_create(self.vault, 'skills')
        seen = {}

        def script(root, api):
            self._click(api.canvas, api.ui['keycaps']['groups'])
            seen['open'] = api.ui['open']
            seen['names'] = sorted(api.ui['keycaps'])
            boxes = sorted(api.ui['keycaps'].values())
            seen['overlap'] = any(a[2] > b[0] for a, b in zip(boxes, boxes[1:]))
        self._with_size('13', script, image=b'x' * 2048)
        self.assertTrue(seen['open'])
        self.assertEqual(seen['names'], ['groups', 'save'])
        self.assertFalse(seen['overlap'])

    def test_x_titles_lose_their_noise(self):
        self.assertEqual(grab.clean_title('(2) X\'te George from prodmgmt.world: "/edge-cases: a simple skill" / X'),
                         'George from prodmgmt.world \u2014 /edge-cases: a simple skill')
        self.assertEqual(grab.clean_title('George on X: "hello\nworld"'), 'George \u2014 hello world')
        self.assertEqual(grab.clean_title('Plain page title'), 'Plain page title')
        self.assertEqual(grab.clean_title(None), '')

    def test_footer_left_is_the_app_and_status_never_touches_an_action(self):
        for stored in ('13', '16'):
            seen = {}

            def script(root, api):
                api.say('Save_failed: ' + 'a very long error message ' * 12, 'red')
                root.update()
                status = api.canvas.bbox(api.status_item)
                seen['gaps'] = [box[0] - status[2] for box in api.ui['keycaps'].values()]
            self._with_size(stored, script, image=b'x' * 2048)
            self.assertTrue(seen['gaps'] and min(seen['gaps']) > 0, (stored, seen))
            self.assertEqual(len(seen['gaps']), 2)


@TK_SKIP
class ThoughtTest(Base):
    """e67: grab is also the "empty your head" door: thought cards, open loops, background process, link block."""
    PAGE = {}
    IMAGE = None
    _popup = GroupTest._popup
    _meta = GroupTest._meta

    def setUp(self):
        super().setUp()
        for patcher in (patch.object(grab, '_clipboard', return_value=''), patch.dict(sys.modules, {'memrove_v3_graph': None})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _type_and_save(self, text, keys=()):
        def script(root, api):
            api.why(text)
            for key in keys:
                root.event_generate(key)
                root.update()
            root.event_generate('<Return>')
        return self._popup(script)

    def test_popup_without_source_saves_a_thought_verbatim(self):
        typed = 'Should I write it in Rust?  Try this too… '
        result = self._type_and_save(typed)
        card = grab.find_card(self.vault, result['id'])
        self.assertEqual(card['meta']['source_kind'], 'thought')
        self.assertEqual(card['meta']['status'], 'thought')
        self.assertEqual(card['body'], typed + '\n')
        self.assertNotIn('open', card['meta'])

    def test_empty_thought_is_not_saved(self):
        self.assertEqual(self._type_and_save('   ').get('status'), 'cancelled')
        self.assertEqual(grab.cards(self.vault), [])

    def test_source_row_reads_thought(self):
        seen = {}

        def script(root, api):
            seen['text'] = [api.canvas.itemcget(i, 'text') for i in api.canvas.find_withtag('prev')]
        self._popup(script)
        self.assertIn('Thought', seen['text'])

    def test_cmd_d_makes_a_thought_even_with_a_source(self):
        self.PAGE = {'url': 'https://example.com/p', 'title': 'P'}
        with patch.object(grab.subprocess, 'Popen') as popen:
            result = self._type_and_save('sadece fikir', ['<Command-d>'])
        self.assertEqual(self._meta(result)['source_kind'], 'thought')
        self.assertFalse([c for c in popen.call_args_list if 'process' in str(c)])

    def test_cmd_t_flags_open_and_open_close_round_trip(self):
        result = self._type_and_save('bunu bitirmeliyim', ['<Command-t>'])
        self.assertTrue(grab.is_open(self._meta(result)))
        self.assertIn('open: true', grab.find_card(self.vault, result['id'])['path'].read_text(encoding='utf-8'))
        quiet = grab.capture(self.vault, text='yalniz not', thought=True)  # never inferred
        self.assertNotIn('open', self._meta(quiet))
        older = grab.capture(self.vault, text='old work', thought=True, open=True)
        listed = grab.open_loops(self.vault)['cards']
        self.assertEqual({r['id'] for r in listed}, {result['id'], older['id']})
        self.assertIn('days', listed[0])
        closed = grab.close_loop(self.vault, result['id'])
        self.assertTrue(self._meta(result)['closed'].startswith(closed['closed'][:10]))
        self.assertEqual([r['id'] for r in grab.open_loops(self.vault)['cards']], [older['id']])
        with self.assertRaises(ValueError):
            grab.close_loop(self.vault, quiet['id'])

    def test_cli_open_and_close(self):
        card = grab.capture(self.vault, text='ac', thought=True, open=True, groups=['Rust'])
        with patch('builtins.print') as out:
            grab.main(['--vault', str(self.vault), '--json', 'open'])
        self.assertIn(card['id'], out.call_args[0][0])
        with patch('builtins.print'):
            grab.main(['--vault', str(self.vault), '--json', 'close', card['id']])
        self.assertTrue(self._meta(card)['closed'])

    def test_source_save_spawns_process_detached_and_does_not_wait(self):
        self.PAGE = {'url': 'https://example.com/p', 'title': 'P'}
        with patch.dict(os.environ), patch.object(grab.subprocess, 'Popen') as popen:
            os.environ.pop('MEMROVE_GRAB_NO_PROCESS')
            result = self._type_and_save('reason')
        spawned = [c for c in popen.call_args_list if c[0] and 'process' in c[0][0]]
        self.assertEqual(len(spawned), 1)
        command, options = spawned[0][0][0], spawned[0][1]
        self.assertEqual(command[-3:], ['process', '--background', result['id']])
        self.assertTrue(options['start_new_session'])
        self.assertEqual(options['stdout'], grab.subprocess.DEVNULL)
        for name in ('wait', 'communicate', 'poll'):
            getattr(popen.return_value, name).assert_not_called()
        self.assertEqual(grab.find_card(self.vault, result['id'])['meta']['status'], 'pending')

    def test_spawn_failure_leaves_pending_with_a_reason_and_popup_returns(self):
        self.PAGE = {'url': 'https://example.com/p', 'title': 'P'}
        with patch.dict(os.environ), patch.object(grab.subprocess, 'Popen', side_effect=OSError('none')):
            os.environ.pop('MEMROVE_GRAB_NO_PROCESS')
            result = self._type_and_save('reason')
        meta = self._meta(result)
        self.assertEqual(meta['status'], 'pending')
        self.assertIn('none', meta['error'])

    def test_background_process_failure_leaves_pending_with_a_reason(self):
        card = grab.capture(self.vault, url='https://example.com/p', why='x')
        with patch.object(grab, 'extract', return_value=(None, '', 'sayfa okunamadi')):
            report = grab.process(self.vault, [card['id']], background=True)
        meta = self._meta(card)
        self.assertEqual((meta['status'], report['cards'][0]['status']), ('pending', 'pending'))
        self.assertIn('sayfa okunamadi', meta['error'])
        with patch.object(grab, 'extract', return_value=(None, '', 'sayfa okunamadi')):
            grab.process(self.vault, [card['id']])  # the foreground CLI keeps its old behaviour
        self.assertEqual(self._meta(card)['status'], 'error')

    def test_thought_card_is_never_processed(self):
        grab.capture(self.vault, text='dusun', thought=True)
        with patch.object(grab, 'extract') as extract:
            self.assertEqual(grab.process(self.vault)['processed'], 0)
        extract.assert_not_called()

    def test_link_block_ends_the_body_when_graph_is_importable(self):
        graph = types.ModuleType('memrove_v3_graph')
        graph.card_links = lambda meta, body: 'Groups: ' + ', '.join('[[g/%s]]' % n for n in meta.get('groups', []))
        with patch.dict(sys.modules, {'memrove_v3_graph': graph}):
            thought = grab.capture(self.vault, text='fikir', thought=True, groups=['Rust'])
            source = grab.capture(self.vault, url='https://example.com/p', why='x', groups=['Rust'])
            with patch.object(grab, 'extract', return_value=('text ' * 30, 'stdlib', None)):
                grab.process(self.vault, [source['id']])
            for res in (thought, source):
                body = grab.find_card(self.vault, res['id'])['body']
                self.assertTrue(body.rstrip().endswith(grab.LINKS_CLOSE), body)
                self.assertEqual(body.count('Groups: [[g/Rust]]'), 1)
            self.assertTrue(grab.find_card(self.vault, thought['id'])['body'].startswith('fikir\n'))
            self.assertIn('## Processed', grab.find_card(self.vault, source['id'])['body'])

    def test_no_link_block_without_graph_or_when_it_raises(self):
        plain = grab.capture(self.vault, text='fikir', thought=True)
        self.assertEqual(grab.find_card(self.vault, plain['id'])['body'], 'fikir\n')
        broken = types.ModuleType('memrove_v3_graph')
        broken.card_links = lambda meta, body: 1 / 0
        with patch.dict(sys.modules, {'memrove_v3_graph': broken}):
            res = grab.capture(self.vault, text='other', thought=True)
        self.assertEqual(grab.find_card(self.vault, res['id'])['body'], 'other\n')


if __name__ == '__main__':
    unittest.main()
