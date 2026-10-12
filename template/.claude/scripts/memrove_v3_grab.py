#!/usr/bin/env python3
# Permission notice: Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files, to deal in the Software without restriction, subject to the above copyright notice and
# this permission notice being included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""One-key capture ("grab"): a popup, a card per capture, and `process`, which turns each card into wiki source evidence.

Capturing never touches the network: a card is one Markdown file in <vault>/wiki/_sources/grab/.
`process` is the only step that fetches, and only through the guards below: the URL is scrubbed to an allow-list of query keys,
never-fetch hosts and `visibility: private` cards are read locally only, and Defuddle runs from a pinned local install.
A processed card becomes a snapshot under wiki/_sources/snap/<slug>/, a catalog entry (origin: grab) and one memrove note.
Card id = file stem = YYYY-MM-DD-slug; `[grab:<id>]` in a wiki page cites it.
"""
import argparse
import functools
from datetime import datetime, timedelta
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unicodedata
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse

sys.dont_write_bytecode = True

INBOX = 'wiki/_sources/grab'
CARD_ID = re.compile(r'^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*$')  # card id = file stem; Panel resolves [grab:<id>] to INBOX/<id>.md
OCR_BIN = Path(os.environ.get('MEMROVE_GRAB_OCR') or Path.home() / '.local/share/memrove/tools/grab-ocr')
TOOLS = Path(os.environ.get('MEMROVE_GRAB_TOOLS') or Path.home() / '.local/share/memrove/tools/defuddle')
RAW = '.raw'
FILES = 'files'
STATE_FILE = 'grab.json'
DEFUDDLE_VERSION = '0.19.4'  # pinned in TOOLS/package.json + lock; installed with `npm ci --ignore-scripts`, never npx -y
LAUNCH_LABEL = 'com.memrove.grab'
MAC_HOTKEY = 'Control+Option+B'  # Option+Command+B is the bookmarks key in Chrome and Safari
WIN_HOTKEY = 'CTRL+ALT+B'
STATUSES = ('pending', 'extracted', 'processed', 'error', 'thought')  # 'thought' = a thought card: nothing to process
PRIVATE_KINDS = ('mail', 'text', 'file', 'image', 'thought')  # a screenshot can hold anything: never fetched, never shared
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


# ---------------------------------------------------------------- cards

def inbox(vault):
    return Path(vault) / INBOX


def now():
    return datetime.now().astimezone()


def slug(text, limit=48):
    # Fold Turkish dotted/dotless i.
    text = str(text or '').replace('ı', 'i').replace('İ', 'I')
    text = unicodedata.normalize('NFKD', str(text or '')).encode('ascii', 'ignore').decode('ascii').lower()
    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return (text[:limit].rstrip('-') or 'note')


def kind_of(url=None, files=(), app=None):
    if files:
        return 'file'
    if not url:
        return 'mail' if app and re.search(r'mail|outlook|spark|thunderbird', app, re.I) else 'text'
    host = (urlparse(url).hostname or '').lower().removeprefix('www.').removeprefix('m.')
    path = urlparse(url).path.lower()
    if host in ('youtube.com', 'youtu.be', 'music.youtube.com'):
        return 'youtube'
    if host in ('x.com', 'twitter.com', 'fxtwitter.com', 'vxtwitter.com'):
        return 'x'
    if host in ('mail.google.com', 'outlook.live.com', 'outlook.office.com', 'mail.proton.me'):
        return 'mail'
    if host == 'github.com':
        return 'github'
    if path.endswith('.pdf'):
        return 'pdf'
    return 'article'


# ---------------------------------------------------------------- privacy guards

ALLOWED_QUERY = {'youtube': ('v', 't'), 'arxiv': ()}  # everything else is dropped: tokens, tracking ids, session keys
NEVER_FETCH_HOSTS = ('mail.google.com', 'outlook.live.com', 'outlook.office.com', 'outlook.office365.com', 'mail.proton.me',
                     'docs.google.com', 'drive.google.com', 'sheets.google.com', 'slides.google.com', 'calendar.google.com',
                     'notion.so', 'notion.site', 'slack.com', 'localhost')
NEVER_FETCH_SUFFIXES = ('.local', '.localhost', '.internal', '.lan', '.home.arpa', '.corp', '.intranet',
                        '.notion.so', '.notion.site', '.slack.com')


def scrub_url(url):
    """Keep scheme, host, path and only the allow-listed query keys (YouTube v and t); no user info, no fragment.
    A URL that is not http(s) is returned as ''."""
    if not url:
        return url
    parts = urlparse(url.strip())
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        return ''
    host = parts.hostname.lower()
    bare = host.removeprefix('www.').removeprefix('m.')
    keys = ALLOWED_QUERY['youtube'] if bare in ('youtube.com', 'music.youtube.com', 'youtu.be') else ALLOWED_QUERY['arxiv']
    query = [(k, v) for k, v in parse_qsl(parts.query) if k in keys]
    netloc = host + (':' + str(parts.port) if parts.port else '')
    return urlunparse((parts.scheme, netloc, parts.path, '', urlencode(query), ''))


def never_fetch(url):
    """True when the URL must never leave this machine: no http(s), mail/docs/notion/slack hosts, localhost, *.local, private IPs."""
    if not url:
        return False
    parts = urlparse(url.strip())
    host = (parts.hostname or '').lower().rstrip('.')
    if parts.scheme not in ('http', 'https') or not host:
        return True
    if host in NEVER_FETCH_HOSTS or host.endswith(NEVER_FETCH_SUFFIXES) or '.' not in host and not host.startswith('['):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if re.fullmatch(r'(0x[0-9a-f]+|\d+)(\.(0x[0-9a-f]+|\d+)){0,3}', host):
            return True  # shorthand or hex IPv4 forms
        return False
    return not ip.is_global


def private_card(meta):
    return meta.get('visibility') == 'private' or meta.get('source_kind') in PRIVATE_KINDS or never_fetch(meta.get('url'))


def canonical(url):
    """Same video or page captured twice must land on one card."""
    if not url:
        return ''
    parts = urlparse(url.strip())
    host = (parts.hostname or '').lower().removeprefix('www.').removeprefix('m.')
    if host == 'youtu.be':
        return 'youtube:' + parts.path.strip('/')
    if host.endswith('youtube.com'):
        video = parse_qs(parts.query).get('v', [''])[0] or parts.path.rsplit('/', 1)[-1]
        return 'youtube:' + video
    if host in ('x.com', 'twitter.com'):
        match = re.search(r'/status/(\d+)', parts.path)
        if match:
            return 'x:' + match.group(1)
    return host + parts.path.rstrip('/') + ('?' + parts.query if parts.query else '')


def _value(raw):
    raw = raw.strip()
    if raw.startswith('"'):
        try:
            return json.loads(raw)
        except ValueError:
            return raw.strip('"')
    if raw.startswith('[') and raw.endswith(']'):
        try:
            return json.loads(raw)
        except ValueError:
            return [part.strip().strip('"\'') for part in raw[1:-1].split(',') if part.strip()]
    if raw.startswith("'") and raw.endswith("'") and len(raw) > 1:
        return raw[1:-1].replace("''", "'")
    return raw


def read_card(path):
    text = Path(path).read_text(encoding='utf-8')
    if not text.startswith('---'):
        return {}, text
    end = text.find('\n---', 3)
    if end < 0:
        return {}, text
    meta = {}
    for line in text[3:end].splitlines():
        match = re.match(r'^([A-Za-z_][\w-]*):\s?(.*)$', line)
        if match:
            meta[match.group(1)] = _value(match.group(2))
    return meta, text[end + 4:].lstrip('\n')


def _dump(meta, body):
    lines = ['---']
    for key, value in meta.items():
        if value is None or value == '':
            continue
        lines.append(key + ': ' + json.dumps(value, ensure_ascii=False))
    return '\n'.join(lines) + '\n---\n\n' + body.rstrip('\n') + '\n'  # only trailing newlines are normalised: a thought stays verbatim


LINKS_OPEN, LINKS_CLOSE = '<!-- links -->', '<!-- /links -->'
LINKS_BLOCK = re.compile(r'\n*' + re.escape(LINKS_OPEN) + r'.*?' + re.escape(LINKS_CLOSE) + r'[ \t]*\n*', re.S)


def with_links(meta, body):
    """The card body ends with the link block of memrove_v3_graph.card_links. Any old block is dropped first, so a rewrite never
    stacks them; with no graph module (or an error in it) the body is left without a block."""
    body = LINKS_BLOCK.sub('\n\n', body).rstrip('\n') if LINKS_OPEN in body else body
    try:
        import memrove_v3_graph
        block = str(memrove_v3_graph.card_links(meta, body) or '').strip()
    except Exception:
        block = ''
    return body.rstrip('\n') + '\n\n' + LINKS_OPEN + '\n' + block + '\n' + LINKS_CLOSE if block else body


def write_card(path, meta, body):
    path = Path(path)
    body = with_links(meta, body)
    fd, temp = tempfile.mkstemp(prefix='.grab-', dir=path.parent)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(_dump(meta, body))
    os.replace(temp, path)


def cards(vault):
    folder = inbox(vault)
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob('*.md')):
        try:
            meta, body = read_card(path)
        except (OSError, UnicodeDecodeError):
            continue
        if meta.get('type') == 'grab':
            found.append({'path': path, 'id': path.stem, 'meta': meta, 'body': body})
    return found


def find_card(vault, card_id):
    for card in cards(vault):
        if card['id'] == card_id or card['path'].name == card_id:
            return card
    raise ValueError('Card not found: ' + str(card_id))


def pending(vault):
    """Cheap for SessionStart: one directory listing, a 600-byte head read per card."""
    folder = inbox(vault)
    count = 0
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return 0
    for entry in entries:
        if entry.name.endswith('.md') and entry.is_file():
            try:
                with open(entry.path, encoding='utf-8', errors='ignore') as handle:
                    head = handle.read(600)
            except OSError:
                continue
            if re.search(r'^status:\s*"?pending"?\s*$', head, re.M) and re.search(r'^type:\s*"?grab"?\s*$', head, re.M):
                count += 1
    return count


def _unique(path):
    path = Path(path)
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(path.stem + '-' + str(number) + path.suffix)
        if not candidate.exists():
            return candidate
    raise ValueError('Too many files with the same name: ' + path.name)


GROUP_FILE = '.groups.json'  # next to the cards: known groups, the last selection, and the reason of each exact group set
GROUP_WINDOW_MIN = 60  # the last selection and a set's reason carry over only when used this recently


def fold(name):
    """Case folding that knows Turkish: İ -> i, I -> ı, so 'IŞIK' and 'ışık' are one group but 'IŞIK' and 'işik' are not."""
    return str(name or '').replace('İ', 'i').replace('I', 'ı').casefold()


def group_name(raw):
    return re.sub(r'\s+', ' ', str(raw or '')).strip()[:40]


def group_list(raw):
    """Clean, de-duplicated (case-insensitive) group names, order kept. Accepts a list or a comma separated string."""
    items = re.split(r'\s*,\s*', raw) if isinstance(raw, str) else (raw or [])
    seen, found = set(), []
    for item in items:
        name = group_name(item)
        if name and fold(name) not in seen:
            seen.add(fold(name))
            found.append(name)
    return found


def _group_store(vault):
    try:
        data = json.loads((inbox(vault) / GROUP_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    return {'groups': data.get('groups') if isinstance(data.get('groups'), dict) else {},
            'selection': data.get('selection') if isinstance(data.get('selection'), dict) else {},
            'sets': data.get('sets') if isinstance(data.get('sets'), dict) else {}}


SUGGESTIONS = 3  # faint group tokens shown in the empty field
RECENT_TOP = 5  # this many most recently used groups lead the list; the rest follow alphabetically


def recent_groups(vault):
    """Every known group: the RECENT_TOP most recently used first, then the others alphabetically (Turkish-aware)."""
    known = _group_store(vault)['groups']
    by_use = sorted(known, key=lambda name: str(known[name]), reverse=True)
    return by_use[:RECENT_TOP] + sorted(by_use[RECENT_TOP:], key=fold)


def group_counts(vault):
    """{group: number of cards that carry it}, for the groups the store knows."""
    counts = {}
    for card in cards(vault):
        for name in group_list(card['meta'].get('groups')):
            counts[fold(name)] = counts.get(fold(name), 0) + 1
    return {name: counts.get(fold(name), 0) for name in _group_store(vault)['groups']}


def group_create(vault, raw):
    """Make a group known now (so it survives a popup that is closed unsaved). Returns the name in its stored spelling."""
    name = group_name(raw)
    if not name:
        raise ValueError('The group name cannot be empty.')
    store = _group_store(vault)
    existing = next((known for known in store['groups'] if fold(known) == fold(name)), None)
    if existing is None:
        store['groups'][name] = now().isoformat(timespec='seconds')
        _group_write(vault, store)
    return existing or name


def _group_write(vault, store):
    folder = inbox(vault)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.groups-', dir=folder)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(store, handle, ensure_ascii=False, indent=1)
    os.replace(temp, folder / GROUP_FILE)


def _fresh(stamp):
    try:
        return timedelta(0) <= now() - datetime.fromisoformat(stamp) <= timedelta(minutes=GROUP_WINDOW_MIN)
    except (ValueError, TypeError):
        return False


def _set_key(names):
    return '|'.join(sorted(fold(name) for name in names))


def active_groups(vault):
    """The groups of the last capture when it was within 60 minutes, else []."""
    chosen = _group_store(vault)['selection']
    return group_list(chosen.get('set')) if _fresh(chosen.get('last')) else []


def set_reason(vault, names):
    """The reason last used with exactly this group set, when that was within 60 minutes, else ''."""
    entry = _group_store(vault)['sets'].get(_set_key(group_list(names))) if group_list(names) else None
    return str(entry.get('reason') or '') if isinstance(entry, dict) and _fresh(entry.get('last')) else ''


def _group_use(vault, names, typed_why):
    """Record a capture with these groups. A typed reason is stored as the set's reason as written; an empty one falls back to
    the set's fresh reason. Returns (names in their known spelling, the reason to use)."""
    store = _group_store(vault)
    spelling = {fold(name): name for name in store['groups']}
    names = [spelling.get(fold(name), name) for name in names]
    key = _set_key(names)
    previous = store['sets'].get(key) if isinstance(store['sets'].get(key), dict) else {}
    reason = typed_why or (str(previous.get('reason') or '') if _stamp_fresh(previous) else '')
    stamp = now().isoformat(timespec='seconds')
    for name in names:
        store['groups'][name] = stamp
    store['selection'] = {'set': names, 'last': stamp}
    store['sets'][key] = {'reason': typed_why or str(previous.get('reason') or ''), 'last': stamp}
    _group_write(vault, store)
    return names, reason


def _stamp_fresh(entry):
    return _fresh(entry.get('last'))


def capture(vault, url=None, text=None, files=(), why='', app=None, title=None, tool='desktop', image=None, groups=None, open=False, thought=False):
    """Write one card; a repeated URL only gains the new reason. No network. `image` = PNG bytes (clipboard screenshot): the card
    is source_kind image, private, and the PNG sits under FILES next to it. `groups` (a list) puts the card in those groups
    (front matter `groups`). An empty `why` falls back to the reason last used with exactly that set within 60 minutes, else stays
    empty; a typed `why` is saved as written and becomes that set's reason."""
    vault = Path(vault)
    raw_text = text or ''
    if thought:  # a thought is the typed text, byte for byte: no URL, no files, no reason, nothing to fetch
        url, files, image, why = None, (), None, ''
        if not raw_text.strip():
            raise ValueError('An empty thought is not saved.')
    url = scrub_url((url or '').strip()) or None  # a token in the query never reaches the card
    text = (text or '').strip() or None
    why = (why or '').strip()
    files = [Path(f).expanduser() for f in files]
    if not (url or text or files or image):
        raise ValueError('Nothing to capture: give a URL, text or a file.')
    folder = inbox(vault)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now()
    names = group_list(groups)
    if names:
        names, shared = _group_use(vault, names, why)
        why = why or shared
    if url:
        key = canonical(url)
        for card in cards(vault):
            if card['meta'].get('url') and canonical(card['meta']['url']) == key:
                body = card['body'].rstrip()
                addition = '\n\n## Captured again (' + stamp.strftime('%Y-%m-%d %H:%M') + ')\n' + (why or '_(no reason given)_')
                if text:
                    addition += '\n\n' + _quote(text)
                if names:
                    card['meta']['groups'] = group_list(list(card['meta'].get('groups') or []) + names)
                write_card(card['path'], card['meta'], body + addition)
                return {'status': 'existing_card_added', 'id': card['id'], 'path': str(card['path'].relative_to(vault))}
    kind = 'thought' if thought else 'image' if image else kind_of(url, files, app)
    if thought:
        title = _first_line(raw_text, 160)
    if image and not title:
        title = 'Clipboard image'
    if not title:
        title = (files[0].name if files else None) or (url if url else _first_line(text))
    copied = []
    name = stamp.strftime('%Y-%m-%d') + '-' + (url_slug(url) if url and title == url else slug(title))
    path = _unique(folder / (name + '.md'))
    if image:
        target = folder / FILES / (path.stem + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image)
        copied.append(target.relative_to(vault).as_posix())
    for source in files:
        if not source.is_file():
            raise ValueError('File not found: ' + str(source))
        target = _unique(folder / FILES / source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target.relative_to(vault).as_posix())
    meta = {'type': 'grab', 'status': 'pending', 'source_kind': kind, 'title': _first_line(title, 160),
            'url': url, 'captured': stamp.isoformat(timespec='seconds'), 'tool': tool, 'app': app,
            'visibility': 'private' if kind in PRIVATE_KINDS or never_fetch(url) else 'internal'}
    if thought:
        meta['status'] = 'thought'
    if names:
        meta['groups'] = names
    if open:
        meta['open'] = True  # explicit only: never inferred from the text
    if thought:
        write_card(path, meta, raw_text)
        return {'status': 'captured', 'id': path.stem, 'source_kind': kind, 'path': path.relative_to(vault).as_posix()}
    body =['# ' + _first_line(title, 160), '', '## Reason', '', why or '_(no reason given)_']
    if url:
        body += ['', '## Source', '', url]
    if text:
        body += ['', '## Selection', '', _quote(text)]
    if copied:
        body += ['', '## Files', ''] + ['- [[' + item + ']]' for item in copied]
    write_card(path, meta, '\n'.join(body))
    return {'status': 'captured', 'id': path.stem, 'source_kind': kind, 'path': path.relative_to(vault).as_posix()}


def is_open(meta):
    return str(meta.get('open')).lower() == 'true' and not meta.get('closed')


def open_loops(vault):
    """Cards the owner flagged `open` and has not closed, newest first, with groups and age in days."""
    rows = []
    for card in cards(vault):
        meta = card['meta']
        if is_open(meta):
            rows.append({'id': card['id'], 'title': meta.get('title'), 'groups': group_list(meta.get('groups')),
                         'captured': meta.get('captured'), 'days': _age(meta, card['path'])[1]})
    rows.sort(key=lambda row: (row['captured'] or '', row['id']), reverse=True)
    return {'status': 'ok', 'cards': rows}


def close_loop(vault, card_id):
    card = find_card(vault, card_id)
    meta = dict(card['meta'])
    if not is_open(meta):
        raise ValueError('This card is not an open loop: ' + card['id'])
    meta['closed'] = now().isoformat(timespec='seconds')
    write_card(card['path'], meta, card['body'])
    return {'status': 'closed', 'id': card['id'], 'closed': meta['closed']}


def url_slug(url):
    """Readable file name before the real title is known: youtube-<id>, x-<user>-<id>, site-last-segment."""
    key = canonical(url)
    if key.startswith('youtube:'):
        return 'youtube-' + slug(key[8:], 20)
    parts = urlparse(url)
    if key.startswith('x:'):
        return slug('x-' + parts.path.strip('/').split('/')[0] + '-' + key[2:])
    host = (parts.hostname or 'link').lower().removeprefix('www.').split('.')[0]
    tail = next((seg for seg in reversed(parts.path.split('/')) if seg), '')
    return slug(host + '-' + re.sub(r'\.[a-z0-9]{2,5}$', '', tail))


def clean_title(title):
    """A page title without the site's own noise: X's "(2) ", "X'te " and " / X", reading `Name — text` for tweets."""
    text = re.sub(r'^\(\d+\)\s+', '', str(title or '').strip())
    text = re.sub(r"^X'te\s+", '', text)
    text = re.sub(r'\s*/\s*X$', '', text)
    text = re.sub(r'\s+on X(?=:)', '', text)
    tweet = re.match(r'^(.+?):\s*[\u201c"](.*?)[\u201d"]$', text, re.S)
    if tweet:
        text = tweet.group(1) + ' \u2014 ' + ' '.join(tweet.group(2).split())
    return text


def _first_line(text, limit=80):
    line = next((part.strip() for part in str(text or '').splitlines() if part.strip()), '')
    return line[:limit] or 'Not'


def _quote(text):
    return '\n'.join('> ' + line if line else '>' for line in str(text).strip().splitlines())


# ---------------------------------------------------------------- extraction

def _which(*names):
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(command, timeout, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', NO_COLOR='1')
    result = subprocess.run(command, capture_output=True, timeout=timeout, cwd=cwd, env=env,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return result.returncode, result.stdout.decode('utf-8', 'replace'), result.stderr.decode('utf-8', 'replace')


def tools():
    """Optional helpers; each one only widens what `process` can do."""
    whisper = _which('mlx_whisper', 'whisper-ctranslate2', 'whisper')
    defuddle = TOOLS / 'node_modules/.bin/defuddle'
    return {'defuddle': str(defuddle) if defuddle.is_file() else None, 'node': _which('node'), 'yt_dlp': _which('yt-dlp', 'yt_dlp'), 'whisper': whisper,
            'ocr': str(OCR_BIN) if OCR_BIN.is_file() else None, 'pdftotext': _which('pdftotext'), 'ffmpeg': _which('ffmpeg')}


OCR_SWIFT = '''import AppKit
import Foundation
import Vision
// Local text recognition for a screenshot: macOS Vision, accurate level, Turkish + English. No network.
guard CommandLine.arguments.count > 1, let image = NSImage(contentsOfFile: CommandLine.arguments[1]),
      let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { exit(2) }
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["tr-TR", "en-US"]
request.usesLanguageCorrection = true
do { try VNImageRequestHandler(cgImage: cg, options: [:]).perform([request]) } catch { exit(3) }
let rows = (request.results ?? []).sorted {
    abs($0.boundingBox.midY - $1.boundingBox.midY) > 0.01 ? $0.boundingBox.midY > $1.boundingBox.midY : $0.boundingBox.minX < $1.boundingBox.minX
}
print(rows.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\\n"))
'''


def install_ocr(binary=None):
    """Compile the Vision OCR helper once (swiftc, local, no network)."""
    binary = Path(binary or OCR_BIN)
    if binary.is_file():
        return {'status': 'installed'}
    swiftc = _which('swiftc')
    if not swiftc:
        return {'status': 'missing', 'note': 'swiftc not found (install Xcode command line tools)'}
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='memrove-ocr-') as work:
        source = Path(work) / 'ocr.swift'
        source.write_text(OCR_SWIFT, encoding='utf-8')
        code, _, err = _run([swiftc, '-O', '-o', str(binary), str(source)], timeout=600)
    return {'status': 'installed'} if code == 0 and binary.is_file() else {'status': 'error', 'note': err.strip()[-300:]}


def ocr_text(image_path, binary=None):
    """Text in an image, '' when none is found, None when the OCR helper is not available."""
    binary = Path(binary or OCR_BIN)
    if not binary.is_file():
        install_ocr(binary)
    if not binary.is_file():
        return None
    try:
        code, out, _ = _run([str(binary), str(image_path)], timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.strip() if code == 0 else None


def via_defuddle(url, found, info=None):
    """Main text as Markdown; title, author and date go into `info` when given."""
    if not (found['defuddle'] and found['node'] and url and not never_fetch(url)):
        return None  # the pinned local install only; never npx -y
    try:
        code, out, _ = _run([found['defuddle'], 'parse', url, '--json', '--markdown'], timeout=180)
        data = json.loads(out) if code == 0 else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if info is not None:
        info.update({key: str(data[key]).strip() for key in ('title', 'author', 'published', 'site')
                     if data.get(key) and str(data[key]).strip()})
    text = str(data.get('content') or '').strip()
    return text if len(text) > 80 else None


def vtt_text(raw):
    """WebVTT to paragraphs that start with a timestamp roughly every 30 seconds; rolling duplicates removed."""
    paragraphs, last, mark, stamp = [], None, -999, None
    for line in raw.splitlines():
        line = line.strip()
        match = re.match(r'^(?:(\d+):)?(\d+):(\d+)[.,]\d+\s+-->', line)
        if match:
            hours, minutes, seconds = (int(part or 0) for part in match.groups())
            stamp = hours * 3600 + minutes * 60 + seconds
            continue
        if not line or line.startswith(('WEBVTT', 'Kind:', 'Language:', 'NOTE')) or line.isdigit() or '-->' in line:
            continue
        line = re.sub(r'<[^>]+>', '', line).strip()
        if not line or line == last:
            continue
        last = line
        if stamp is not None and stamp - mark >= 30 or not paragraphs:
            mark = stamp if stamp is not None else mark
            label = '%d:%02d' % divmod(stamp or 0, 60) if (stamp or 0) < 3600 else '%d:%02d:%02d' % (stamp // 3600, stamp % 3600 // 60, stamp % 60)
            paragraphs.append(['**' + label + '**', line])
        else:
            paragraphs[-1].append(line)
    return '\n\n'.join(' '.join(words) for words in paragraphs).strip()


def via_subtitles(url, found, workdir):
    if not found['yt_dlp']:
        return None
    command = [found['yt_dlp'], '--skip-download', '--write-subs', '--write-auto-subs',
               '--sub-langs', '.*-orig,tr,en', '--sub-format', 'vtt', '--no-playlist',
               '-o', str(Path(workdir) / 'sub.%(ext)s'), url]
    try:
        _run(command, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    def rank(path):  # original-language track first, then a human Turkish/English one
        name = path.name
        return (0 if name.endswith('-orig.vtt') else 1 if name.endswith('.tr.vtt') else 2 if name.endswith('.en.vtt') else 3, name)
    subs = sorted(Path(workdir).glob('sub*.vtt'), key=rank)
    for path in subs:
        text = vtt_text(path.read_text(encoding='utf-8', errors='replace'))
        if len(text) > 80:
            return text
    return None


def via_audio(url, found, workdir):
    """Last resort: download only the audio, transcribe locally, delete the audio."""
    if not (found['yt_dlp'] and found['whisper']):
        return None
    audio = Path(workdir) / 'audio'
    try:
        _run([found['yt_dlp'], '-f', 'bestaudio/best', '--no-playlist', '-o', str(audio) + '.%(ext)s', url], timeout=900)
        files = sorted(Path(workdir).glob('audio.*'))
        if not files:
            return None
        whisper = Path(found['whisper']).name.lower()
        if whisper.startswith('mlx_whisper'):
            command = [found['whisper'], str(files[0]), '--model', 'mlx-community/whisper-large-v3-turbo',
                       '-f', 'txt', '-o', str(workdir)]
        else:
            command = [found['whisper'], str(files[0]), '--output_format', 'txt', '--output_dir', str(workdir)]
        _run(command, timeout=7200)
        texts = sorted(Path(workdir).glob('audio*.txt'))
        return texts[0].read_text(encoding='utf-8', errors='replace').strip() if texts else None
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        for path in Path(workdir).glob('audio.*'):
            if path.suffix != '.txt':
                path.unlink(missing_ok=True)


def via_stdlib(url):
    """No helper installed: plain HTTP fetch and a rough HTML-to-text."""
    from html.parser import HTMLParser
    from urllib.request import Request, urlopen
    if never_fetch(url):
        return None

    class Text(HTMLParser):
        skip = {'script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg', 'form', 'aside'}

        def __init__(self):
            super().__init__()
            self.parts, self.depth, self.title, self.in_title = [], 0, '', False

        def handle_starttag(self, tag, attrs):
            if tag in self.skip:
                self.depth += 1
            if tag == 'title':
                self.in_title = True
            if tag in ('p', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'tr', 'div', 'section', 'article'):
                self.parts.append('\n')

        def handle_endtag(self, tag):
            if tag in self.skip and self.depth:
                self.depth -= 1
            if tag == 'title':
                self.in_title = False

        def handle_data(self, data):
            if self.in_title:
                self.title += data
            elif not self.depth:
                self.parts.append(data)

    try:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (memrove-grab)'})
        with urlopen(request, timeout=30) as response:
            if 'html' not in response.headers.get('Content-Type', 'text/html'):
                return None
            html = response.read(5_000_000).decode(response.headers.get_content_charset() or 'utf-8', 'replace')
    except Exception:
        return None
    parser = Text()
    parser.feed(html)
    text = re.sub(r'\n\s*\n+', '\n\n', re.sub(r'[ \t]+', ' ', ''.join(parser.parts))).strip()
    return (('# ' + parser.title.strip() + '\n\n') if parser.title.strip() else '') + text if len(text) > 200 else None


def _section(body, name):
    match = re.search(r'^## ' + name + r'\s*$(.*?)(?=^## |\Z)', body, re.M | re.S)
    return match.group(1).strip() if match else ''


def _without_section(body, name, replacement):
    return re.sub(r'^## ' + name + r'\s*$.*?(?=^## |\Z)', '## ' + name + '\n\n' + replacement + '\n\n', body, count=1, flags=re.M | re.S)


def extract(vault, card, found, allow_audio=False, info=None, url=None, online=True):
    """Return (text, method, note). `url` is the scrubbed URL. online=False (private card or never-fetch host): local text only."""
    meta, body = card['meta'], card['body']
    url = url if url is not None else (scrub_url(meta['url']) if meta.get('url') else None)
    kind = meta.get('source_kind') or kind_of(url)
    if kind == 'image':  # local Vision OCR only: no fetch, whatever the recognized text says
        for link in re.findall(r'\[\[([^\]]+)\]\]', _section(body, 'Files')):
            if link.lower().endswith('.png') and (Path(vault) / link).is_file():
                text = ocr_text(Path(vault) / link)
                if text is None:
                    return None, None, 'OCR helper unavailable (swiftc missing or the image is unreadable)'
                return text, 'vision-ocr', None
        return None, None, 'image file missing next to the card'
    if kind == 'mail':
        selected = _section(body, 'Selection')
        if selected:
            return selected, 'selection', None
        return None, None, 'mail content needs a login: select the text and capture it'
    if kind == 'text':
        return _section(body, 'Selection') or body, 'text', None
    if kind == 'file':
        parts = []
        for link in re.findall(r'\[\[([^\]]+)\]\]', _section(body, 'Files')):
            path = Path(vault) / link
            if path.suffix.lower() in ('.md', '.txt', '.csv', '.json', '.html', '.srt', '.vtt'):
                parts.append('## ' + path.name + '\n\n' + path.read_text(encoding='utf-8', errors='replace'))
            elif path.suffix.lower() == '.pdf' and found['pdftotext']:
                try:
                    code, out, _ = _run([found['pdftotext'], '-layout', str(path), '-'], timeout=120)
                    if code == 0 and out.strip():
                        parts.append('## ' + path.name + '\n\n' + out)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if parts:
            return '\n\n'.join(parts), 'file', None
        return None, None, 'file type not extractable here (image or binary)'
    if not online or not url:
        selected = _section(body, 'Selection')
        if selected:
            return selected, 'selection', None
        return None, None, 'this card is never fetched (private or never-fetch host); capture a selection instead'
    if kind == 'youtube':
        text = via_defuddle(url, found, info)
        if text and re.search(r'^## Transcript', text, re.M):
            return text, 'defuddle', None
        with tempfile.TemporaryDirectory(prefix='memrove-grab-') as workdir:
            subtitles = via_subtitles(url, found, workdir)
            if subtitles:
                return ((text + '\n\n## Transcript\n\n') if text else '') + subtitles, 'yt-dlp-subs', None
            if allow_audio:
                spoken = via_audio(url, found, workdir)
                if spoken:
                    return ((text + '\n\n## Transcript\n\n') if text else '') + spoken, 'audio-whisper', None
        if text:
            return text, 'defuddle', 'no transcript found; title and description only'
        return None, None, 'YouTube content could not be fetched'
    text = via_defuddle(url, found, info)
    if text:
        return text, 'defuddle', None
    text = via_stdlib(url)
    if text:
        return text, 'http', None
    return None, None, 'page could not be fetched (login needed?); capture a selection instead'


def finish(vault, card_id, sources=(), summary=None, target=None):
    vault = Path(vault)
    card = find_card(vault, card_id)
    meta, body = dict(card['meta']), card['body'].rstrip()
    links = []
    for source in sources:
        source = str(source).replace('\\', '/')
        if not (vault / source).is_file():
            raise ValueError('Info note not found: ' + source)
        links.append('- [[' + source.removesuffix('.md') + ']]')
    meta.update(status='processed', processed=now().isoformat(timespec='seconds'))
    _set_target(meta, target)
    addition = '\n\n## Processed\n\n' + ((summary.strip() + '\n\n') if summary else '') + '\n'.join(links or ['_(no info note)_'])
    write_card(card['path'], meta, body + addition)
    return {'status': 'processed', 'id': card['id'], 'info': sources}


def _set_target(meta, target):
    """A target (open project or question) makes a card used from now on; `used_at` records the first time we saw it used."""
    target = (target or '').strip()
    if target:
        meta['target'] = target
        meta.setdefault('used_at', now().isoformat(timespec='seconds'))


def listing(vault, status=None, group=None):
    rows = []
    for card in cards(vault):
        meta = card['meta']
        if status and meta.get('status') != status:
            continue
        if group and fold(group_name(group)) not in (fold(name) for name in group_list(meta.get('groups'))):
            continue
        rows.append({'id': card['id'], 'status': meta.get('status'), 'source_kind': meta.get('source_kind'), 'groups': group_list(meta.get('groups')),
                     'title': meta.get('title'), 'url': meta.get('url'), 'raw': meta.get('raw')})
    return {'status': 'ok', 'cards': rows, 'pending': sum(1 for r in rows if r['status'] == 'pending')}


# ---------------------------------------------------------------- context (which app, which page)

def _objc():
    import ctypes
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    ctypes.CDLL('/System/Library/Frameworks/AppKit.framework/AppKit')
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(target, selector, restype=ctypes.c_void_p, *args):
        argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [type(a) for a in args]
        function = ctypes.CFUNCTYPE(restype, *argtypes)(('objc_msgSend', objc))
        return function(target, objc.sel_registerName(selector), *args)
    return objc, send


def mac_frontmost():
    import ctypes
    try:
        objc, send = _objc()
        workspace = send(objc.objc_getClass(b'NSWorkspace'), b'sharedWorkspace')
        app = send(workspace, b'frontmostApplication')

        def string(value):
            raw = send(value, b'UTF8String', ctypes.c_char_p) if value else None
            return raw.decode('utf-8') if raw else None
        return {'app': string(send(app, b'localizedName')), 'bundle': string(send(app, b'bundleIdentifier'))}
    except Exception:
        return {}


CHROMIUM = {'com.google.Chrome', 'com.google.Chrome.canary', 'com.brave.Browser', 'com.microsoft.edgemac',
            'company.thebrowser.Browser', 'com.vivaldi.Vivaldi', 'com.operasoftware.Opera'}


def _osascript(script, timeout=2.5):
    try:
        result = subprocess.run(['osascript', '-e', script], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return result.stdout.decode('utf-8', 'replace').strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def mac_context(front=None):
    front = front or mac_frontmost()
    bundle = front.get('bundle') or ''
    context = dict(front)
    if bundle in CHROMIUM:
        out = _osascript('tell application id "' + bundle + '" to return (URL of active tab of front window) & linefeed & (title of active tab of front window)')
    elif bundle in ('com.apple.Safari', 'com.apple.SafariTechnologyPreview'):
        out = _osascript('tell application id "' + bundle + '" to return (URL of front document) & linefeed & (name of front document)')
    else:
        out = None
    if out:
        url, _, title = out.partition('\n')
        if url.startswith(('http://', 'https://')):
            context.update(url=url.strip(), title=title.strip())
    if bundle == 'com.apple.finder':
        out = _osascript('tell application "Finder"\nset out to ""\nrepeat with f in (selection as alias list)\n'
                         'set out to out & POSIX path of f & linefeed\nend repeat\nreturn out\nend tell')
        paths = [line for line in (out or '').splitlines() if line and Path(line).is_file()]
        if paths:
            context['files'] = paths[:20]
    return context


def windows_context():
    import ctypes
    from ctypes import wintypes
    try:
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        window = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(window)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(window, buffer, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        process = kernel32.OpenProcess(0x1000, False, pid.value)
        name = None
        if process:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(1024)
            if kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
                name = Path(path.value).stem
            kernel32.CloseHandle(process)
        return {'app': name, 'window': buffer.value}
    except Exception:
        return {}


def gather_context():
    if sys.platform == 'darwin':
        return mac_context()
    if os.name == 'nt':
        return windows_context()
    return {}


# ---------------------------------------------------------------- popup

# Static copy of Panel's dark chrome (Claude-style look), so grab does not need Panel installed.
# Values are those roles for the
# #222221 / #ECEBE6 theme: bg=surfaceBase, card=surfaceRaised, panel=surfaceOverlay, line/card_line=separator, text=textPrimary,
# muted=textSecondary, soft=between the two, faint=placeholder (glyphTertiary lifted to 4.5:1), accent=accentFill=textPrimary
# (monochrome: no hue in the chrome) with accent_text=surfaceBase, focus=focusRing. Selection is a neutral wash plus a hairline
# (foreground at 11% / 16%, pre-blended because Tk has no alpha): sel_* over card, panel_sel_* over panel. Colour is only for
# signals (ANSI, 20% desaturated, >= 4.5:1): warn=attention, error=signalError, ok=signalSuccess.
PALETTE = {'bg': '#222221', 'card': '#292928', 'card_line': '#302F2E', 'line': '#302F2E', 'panel': '#302F2E',
           'text': '#F0EFE9', 'soft': '#C9C8C3', 'muted': '#A7A7A3', 'faint': '#8E8D8A', 'accent': '#F0EFE9',
           'accent_text': '#222221', 'focus': '#93938F', 'key': '#302F2E', 'key_line': '#494846',
           'sel': '#3F3E3D', 'sel_line': '#494846', 'panel_sel': '#444442', 'panel_line': '#4E4D4B',
           'sunk': '#1C1C1B', 'ok': '#BAC761', 'warn': '#E0C560', 'error': '#D08183'}
# One type scale = Panel's own: interface text size S (UserDefaults key interfaceTextSize, clamped 11-16, default 13).
# Body S, secondary S-1, caption S-2, never below 11 (macOS Body 13, Callout 12, Caption 11); only the reason composer is
# larger, S+4 (Title 2). Every size below comes from this table; layout lengths scale with S through _u().
PANEL_SIZE_KEY, PANEL_SIZE_DEFAULT, PANEL_SIZE_RANGE, MIN_SIZE = 'interfaceTextSize', 13, (11, 16), 11
PANEL_DOMAINS = ('com.panel.app', 'com.panel.Panel', 'Panel')
TYPE_ROLES = {'composer': 4, 'body': 0, 'secondary': -1, 'caption': -2}  # offsets from S; floor MIN_SIZE
LOOP_LABEL = 'Open loop'
KIND_LABEL = {'youtube': 'VIDEO', 'x': 'TWEET', 'mail': 'MAIL', 'github': 'REPO', 'pdf': 'PDF', 'article': 'PAGE'}


def _defaults_read(domain):
    """`defaults read <domain> interfaceTextSize` as text, or None when the domain or key is missing (mocked in tests)."""
    if sys.platform != 'darwin':
        return None
    try:
        done = subprocess.run(['defaults', 'read', domain, PANEL_SIZE_KEY], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def type_s():
    """S = Panel's interface text size (first defaults domain that has it, clamped to 11-16), else 13."""
    for domain in PANEL_DOMAINS:
        try:
            value = int(float(_defaults_read(domain) or ''))
        except ValueError:
            continue
        return max(PANEL_SIZE_RANGE[0], min(PANEL_SIZE_RANGE[1], value))
    return PANEL_SIZE_DEFAULT


def _size(role):
    return max(MIN_SIZE, type_s() + TYPE_ROLES[role])


def _u(n):
    """Layout length n (designed at 13 pt text) scaled with the type scale."""
    return int(round(n * type_s() / PANEL_SIZE_DEFAULT))


def _role(role):
    return _font(_size(role))


def _mono_role(role):
    return _mono(_size(role))


def _font(size, weight='normal'):
    """System UI font. `weight` is accepted and ignored: Panel's hierarchy is size and tone, medium at most, and Tk only has
    normal and bold, so everything is normal."""
    if sys.platform == 'darwin':
        family = '.AppleSystemUIFont'
    elif os.name == 'nt':
        family = 'Segoe UI Variable Text' if sys.getwindowsversion().build >= 22000 else 'Segoe UI'
    else:
        family = 'DejaVu Sans'
    return (family, size, 'normal')


def _mono(size):
    """Panel's mono role (S-1) for URLs, paths and shortcuts."""
    return ('SF Mono' if sys.platform == 'darwin' else 'Consolas' if os.name == 'nt' else 'DejaVu Sans Mono', size, 'normal')


def _nsstring(objc, send, text):
    import ctypes
    return send(objc.objc_getClass(b'NSString'), b'stringWithUTF8String:', ctypes.c_void_p, ctypes.c_char_p(text.encode('utf-8')))


def _style_mac(focus=True):
    """Spotlight-like panel: no title bar or buttons, dark appearance, floats over every Space."""
    try:
        import ctypes
        objc, send = _objc()
        app = send(objc.objc_getClass(b'NSApplication'), b'sharedApplication')
        send(app, b'setActivationPolicy:', ctypes.c_void_p, ctypes.c_long(1))  # accessory: no Dock icon
        dark = send(objc.objc_getClass(b'NSAppearance'), b'appearanceNamed:', ctypes.c_void_p,
                    ctypes.c_void_p(_nsstring(objc, send, 'NSAppearanceNameDarkAqua')))
        send(app, b'setAppearance:', ctypes.c_void_p, ctypes.c_void_p(dark))
        windows = send(app, b'windows')
        for index in range(send(windows, b'count', ctypes.c_ulong)):
            window = send(windows, b'objectAtIndex:', ctypes.c_void_p, ctypes.c_ulong(index))
            mask = send(window, b'styleMask', ctypes.c_ulong)
            send(window, b'setStyleMask:', ctypes.c_void_p, ctypes.c_ulong(mask | (1 << 15)))  # full-size content
            send(window, b'setTitlebarAppearsTransparent:', ctypes.c_void_p, ctypes.c_bool(True))
            send(window, b'setTitleVisibility:', ctypes.c_void_p, ctypes.c_long(1))
            for button in range(3):
                handle = send(window, b'standardWindowButton:', ctypes.c_void_p, ctypes.c_long(button))
                if handle:
                    send(handle, b'setHidden:', ctypes.c_void_p, ctypes.c_bool(True))
            send(window, b'setMovableByWindowBackground:', ctypes.c_void_p, ctypes.c_bool(True))
            send(window, b'setCollectionBehavior:', ctypes.c_void_p, ctypes.c_ulong(1 | 256))  # all Spaces, over full screen
            send(window, b'setLevel:', ctypes.c_void_p, ctypes.c_long(25))
            send(window, b'setHasShadow:', ctypes.c_void_p, ctypes.c_bool(True))
        if focus:
            send(app, b'activateIgnoringOtherApps:', ctypes.c_void_p, ctypes.c_bool(True))
    except Exception:
        pass


def _window_number(under_desktop=False):
    """CGWindowID of this process's Tk window (for screencapture -l), or 0. under_desktop sinks it below the wallpaper."""
    try:
        import ctypes
        objc, send = _objc()
        app = send(objc.objc_getClass(b'NSApplication'), b'sharedApplication')
        windows = send(app, b'windows')
        found = 0
        for index in range(send(windows, b'count', ctypes.c_ulong)):
            window = send(windows, b'objectAtIndex:', ctypes.c_void_p, ctypes.c_ulong(index))
            if under_desktop:
                send(window, b'setLevel:', ctypes.c_void_p, ctypes.c_long(-2147483630))
            number = send(window, b'windowNumber', ctypes.c_long)
            found = found or (number if number > 0 else 0)
        return found
    except Exception:
        return 0


STROKE = 1.5  # one stroke weight for every drawn icon


def _check(canvas, x, y, r, color, tags):
    return canvas.create_line(x - r, y, x - r / 3, y + r * 0.7, x + r, y - r * 0.7, fill=color, width=STROKE, capstyle='round',
                              joinstyle='round', tags=tags)


def _plus(canvas, x, y, r, color, tags):
    canvas.create_line(x - r, y, x + r, y, fill=color, width=STROKE, capstyle='round', tags=tags)
    canvas.create_line(x, y - r, x, y + r, fill=color, width=STROKE, capstyle='round', tags=tags)


def _cross(canvas, x, y, r, color, tags):
    canvas.create_line(x - r, y - r, x + r, y + r, fill=color, width=STROKE, capstyle='round', tags=tags)
    canvas.create_line(x - r, y + r, x + r, y - r, fill=color, width=STROKE, capstyle='round', tags=tags)


def _round_rect(canvas, x1, y1, x2, y2, radius, **options):
    r = min(radius, (x2 - x1) / 2, (y2 - y1) / 2)
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2,
              x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **options)


def _fit(canvas, text, font, width):
    """Trim with an ellipsis so the text fits `width` pixels in `font`."""
    import tkinter.font as tkfont
    measure = tkfont.Font(root=canvas, font=font).measure
    if measure(text) <= width:
        return text
    while text and measure(text + '…') > width:
        text = text[:-1]
    return text.rstrip() + '…'


def repair_mojibake(text):
    """UTF-8 bytes that a tool decoded as Mac Roman or cp1252 ('pahalƒ±' for 'pahalı'); else unchanged."""
    if not text or text.isascii():
        return text
    for codec in ('mac_roman', 'cp1252'):
        try:
            fixed = text.encode(codec).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if fixed != text:
            return fixed
    return text


def _clipboard(root):
    try:
        value = root.clipboard_get()
    except Exception:
        return ''
    return repair_mojibake(value) if isinstance(value, str) else ''


def clipboard_image():
    """PNG bytes when the clipboard holds an image (Shottr and the macOS screenshot tool put one there), else None.
    Tk reads text only, so this goes through osascript."""
    if sys.platform != 'darwin':
        return None
    out = _osascript('the clipboard as «class PNGf»', timeout=5)
    match = re.search(r'«data PNGf([0-9A-Fa-f]+)»', out or '')
    if not match:
        return None
    data = bytes.fromhex(match.group(1))
    return data if data.startswith(b'\x89PNG') else None


def choose_source(image, has_other):
    """'image' only when the clipboard image is the only thing to save; the page or files in front always lead, because a
    screenshot on the clipboard is often meant for something else. The panel offers the image as the other source."""
    return 'image' if image and not has_other else 'other'


def _word_start(text, index):
    i = index
    while i > 0 and not (text[i - 1].isalnum() or text[i - 1] == '_'):
        i -= 1
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == '_'):
        i -= 1
    return i


def _word_end(text, index):
    i = index
    while i < len(text) and not (text[i].isalnum() or text[i] == '_'):
        i += 1
    while i < len(text) and (text[i].isalnum() or text[i] == '_'):
        i += 1
    return i


class _EndVar:
    """A StringVar whose programmatic `set` leaves the caret at the end of the text, like typing would."""
    def __init__(self, var, entry):
        self.var, self.entry = var, entry
        self.get = var.get

    def set(self, text):
        self.var.set(text)
        self.entry.icursor('end')


def bind_native_editing(entry, idle=lambda: False):
    """Native text-field keys for a Tk Entry. macOS: Option+Left/Right by word, Cmd+Left/Right to line start/end,
    Option+Backspace/Delete one word, Cmd+Backspace to line start; Shift extends the selection. Windows/Linux: the Ctrl
    equivalents (Ctrl+Backspace/Delete, Ctrl+Left/Right; Home/End are already native there). `idle()` true = a placeholder is
    showing, so the keys do nothing."""
    mac = sys.platform == 'darwin'
    word, line = ('Option', 'Command') if mac else ('Control', None)

    def move(target, extend):
        if extend:
            if not entry.selection_present():
                entry.selection_from('insert')
                entry.selection_to(target)
            else:
                entry.selection_adjust(target)
        else:
            entry.selection_clear()
        entry.icursor(target)

    def mover(find, extend):
        def handler(_event):
            if not idle():
                text = entry.get()
                move(find(text, entry.index('insert')), extend)
            return 'break'
        return handler

    def deleter(find):
        def handler(_event):
            if not idle():
                if entry.selection_present():
                    entry.delete('sel.first', 'sel.last')
                else:
                    here = entry.index('insert')
                    there = find(entry.get(), here)
                    entry.delete(min(here, there), max(here, there))
            return 'break'
        return handler
    for shift, extend in (('', False), ('Shift-', True)):
        entry.bind('<%s%s-Left>' % (shift, word), mover(_word_start, extend))
        entry.bind('<%s%s-Right>' % (shift, word), mover(_word_end, extend))
        if line:
            entry.bind('<%s%s-Left>' % (shift, line), mover(lambda _t, _i: 0, extend))
            entry.bind('<%s%s-Right>' % (shift, line), mover(lambda t, _i: len(t), extend))
    entry.bind('<%s-BackSpace>' % word, deleter(_word_start))
    entry.bind('<%s-Delete>' % word, deleter(_word_end))
    if line:
        entry.bind('<%s-BackSpace>' % line, deleter(lambda _t, _i: 0))
        entry.bind('<%s-Delete>' % line, deleter(lambda t, _i: len(t)))


# Native panel (macOS): a small SwiftUI program (memrove_v3_grab_panel.swift, next to this file) draws the window and Python
# keeps every decision and every write. It is compiled once into the state folder; until it exists the Tk window is used and
# a background build is started. MEMROVE_GRAB_TK=1 forces the Tk window.
PANEL_SOURCE = 'memrove_v3_grab_panel.swift'
SOURCE_SYMBOL = {'x': ('at', 'Tweet'), 'youtube': ('play.rectangle', 'Video'), 'github': ('chevron.left.forwardslash.chevron.right', 'Repo'),
                 'pdf': ('doc.richtext', 'PDF'), 'mail': ('envelope', 'Mail')}


def panel_paths(vault):
    return Path(__file__).resolve().with_name(PANEL_SOURCE), resolve_state(vault) / 'grab' / 'MemroveGrab'


def build_panel(vault):
    """Compile the native panel now (blocking). Returns the binary path, or raises with the compiler's message."""
    source, binary = panel_paths(vault)
    if not source.is_file():
        raise FileNotFoundError('missing ' + str(source))
    binary.parent.mkdir(parents=True, exist_ok=True)
    temp = binary.with_name(binary.name + '.build')
    done = subprocess.run(['nice', 'swiftc', '-O', '-swift-version', '5', str(source), '-o', str(temp)],
                          capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout or 'swiftc failed').strip()[-800:])
    os.replace(temp, binary)
    return binary


def ensure_panel(vault, wait=False):
    """The binary when it is newer than its source. Stale: still the old binary (better than no panel) while one detached build runs, or None without any binary; wait builds now."""
    if sys.platform != 'darwin' or os.environ.get('MEMROVE_GRAB_TK') or not shutil.which('swiftc'):
        return None
    source, binary = panel_paths(vault)
    if not source.is_file():
        return None
    if binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    if wait:
        return build_panel(vault)
    lock = binary.with_name('MemroveGrab.lock')
    try:
        import time
        if lock.is_file() and time.time() - lock.stat().st_mtime < 900:
            return binary if binary.is_file() else None
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(os.getpid()))
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--vault', str(vault), 'panel'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        pass
    return binary if binary.is_file() else None


def _pbpaste():
    try:
        return subprocess.run(['pbpaste'], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ''


def panel_model(vault, context, clip, image):
    """What the panel shows: the possible sources (first = default), the clipboard text option, groups and the carried reason."""
    clip = repair_mojibake((clip or '').strip())
    clip_is_url = bool(re.match(r'^https?://\S+$', clip))
    url = context.get('url') or (clip if clip_is_url else None)
    files = list(context.get('files') or [])
    clip_text = '' if image or clip_is_url else clip
    sources = []
    if url:
        symbol, label = SOURCE_SYMBOL.get(kind_of(url), ('link', 'Page'))
        sources.append({'id': 'url', 'symbol': symbol, 'label': label, 'title': clean_title(context.get('title')) or url,
                        'detail': (urlparse(url).hostname or '').removeprefix('www.')})
    elif files:
        first = Path(files[0])
        sources.append({'id': 'files', 'symbol': 'doc', 'label': 'File', 'detail': str(first.parent),
                        'title': first.name + ('  +' + str(len(files) - 1) if len(files) > 1 else '')})
    if image:
        entry = {'id': 'image', 'symbol': 'photo', 'label': 'Image', 'title': 'Clipboard image', 'detail': '%d KB' % (len(image) // 1024 + 1)}
        sources.append(entry)  # the page or files in front lead; the image is one click (or Tab) away
    if clip_text and not sources:
        sources.append({'id': 'clip', 'symbol': 'text.quote', 'label': 'Text', 'title': _first_line(clip_text, 200), 'detail': 'From clipboard'})
    selected = active_groups(vault)
    counts = group_counts(vault)
    known = list(dict.fromkeys(recent_groups(vault) + selected))
    return {'s': type_s(), 'palette': PALETTE, 'sources': sources, 'source': sources[0]['id'] if sources else None,
            'clip': _first_line(clip_text, 200) if clip_text and sources and sources[0]['id'] != 'clip' else None,
            'includeClip': False, 'groups': [{'name': name, 'count': counts.get(name, 0)} for name in known],
            'selected': selected, 'hint': set_reason(vault, selected), 'app': context.get('app')}, \
        {'url': url, 'files': files, 'clip_text': clip_text, 'image': image}


def panel_reply(vault, context, raw, request):
    """Answer one panel request. Returns (reply dict, result dict or None when the panel is still open)."""
    op = request.get('op')
    if op == 'hint':
        return {'hint': set_reason(vault, group_list(request.get('groups')))}, None
    if op == 'create':
        try:
            existed = any(fold(name) == fold(group_name(request.get('name'))) for name in _group_store(vault)['groups'])
            return {'name': group_create(vault, request.get('name')), 'existed': existed}, None
        except ValueError as exc:
            return {'error': str(exc)}, None
    if op != 'save':
        return {'ok': True}, {'status': 'cancelled'}
    text, groups, open, source = request.get('text') or '', group_list(request.get('groups')), bool(request.get('open')), request.get('source')
    app = context.get('app')
    try:
        if source is None:
            result = capture(vault, text=text, app=app, groups=groups, open=open, thought=True)
        elif source == 'image':
            result = capture(vault, why=text, app=app, image=raw['image'], groups=groups, open=open)
        elif source == 'clip':
            result = capture(vault, text=raw['clip_text'], why=text, app=app, groups=groups, open=open)
        else:
            url = raw['url'] if source == 'url' else None
            result = capture(vault, url=url, files=raw['files'] if source == 'files' else (), why=text, app=app,
                             text=raw['clip_text'] if request.get('includeClip') else None,
                             title=context.get('title') if url else None, groups=groups, open=open)
    except Exception as exc:
        return {'error': str(exc)[:200]}, None
    if result.get('status') == 'captured' and source is not None:
        spawn_process(vault, result['id'])  # detached: the panel never waits for the network
    return {'ok': True}, result


def native_popup(vault, context, binary, clip=None, image=None):
    """Run the native panel. Returns the capture result, or None when the panel never asked anything (it failed to start), so
    the caller can fall back to the Tk window instead of showing nothing."""
    model, raw = panel_model(vault, context, _pbpaste() if clip is None else clip, clipboard_image() if image is None else image)
    proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
    result, asked = {'status': 'cancelled'}, False
    try:
        proc.stdin.write(json.dumps(model, ensure_ascii=False) + '\n')
        proc.stdin.flush()
        for line in proc.stdout:
            try:
                request = json.loads(line)
            except ValueError:
                continue
            asked = True
            reply, done = panel_reply(vault, context, raw, request)
            proc.stdin.write(json.dumps(reply, ensure_ascii=False) + '\n')
            proc.stdin.flush()
            if done is not None:
                result = done
    except (BrokenPipeError, OSError):
        pass
    finally:
        for pipe in (proc.stdin, proc.stdout):
            try:
                pipe.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result if asked else None


def popup(vault, context=None):
    """Command-palette window: what will be saved, one line for why, Enter saves, Esc closes."""
    context = dict(context or {})
    # A Tk window makes this Python process a Foreground app (Dock icon) and cannot be made accessory, so macOS never uses it.
    # Tk is the Windows/Linux window; MEMROVE_GRAB_TK_MAC=1 lets the Tk tests run on a Mac (they flash a Python icon).
    tk_wanted = (sys.platform != 'darwin' or os.environ.get('MEMROVE_GRAB_TK_MAC')) and (
        os.environ.get('MEMROVE_GRAB_SNAPSHOT') or os.environ.get('MEMROVE_GRAB_HEADLESS') or os.environ.get('MEMROVE_GRAB_TK'))
    if not tk_wanted:
        # The hotkey path: the native panel, else the OS dialog. Tk is never imported here (a Tk window puts a Python icon in the Dock).
        binary = ensure_panel(vault)
        result = native_popup(vault, context, binary) if binary else None
        return result if result is not None else _popup_fallback(vault, context)
    try:
        import tkinter as tk
    except ImportError:
        return _popup_fallback(vault, context)
    snapshot = os.environ.get('MEMROVE_GRAB_SNAPSHOT')
    unseen = bool(snapshot or os.environ.get('MEMROVE_GRAB_HEADLESS'))  # tests and snapshots never show a window on the screen
    root = tk.Tk()
    root.withdraw()
    clip = _clipboard(root).strip()
    clip_is_url = bool(re.match(r'^https?://\S+$', clip))
    image = clipboard_image()
    other_url = context.get('url') or (clip if clip_is_url else None)
    other_files = context.get('files') or []
    # A screenshot is the default source only when fresh (clipboard changed in the last 90 s, judged by the listener) or alone.
    src = {'image': choose_source(image, bool(other_url or other_files)) == 'image'}
    both = bool(image and (other_url or other_files))
    url = files = clip_text = None
    include = {'on': False}

    mode = {'thought': False}  # no source (or chosen with Cmd+D): the typed text is saved as a thought card
    loop = {'on': False}  # "Open loop": explicit, never inferred

    def current():
        if mode['thought']:
            return None, [], None, None
        if src['image']:
            return None, [], None, image
        return other_url, other_files, (clip_text or None) if include['on'] else None, None

    def pick():
        nonlocal url, files, clip_text
        url, files = (None, []) if src['image'] else (other_url, other_files)
        clip_text = '' if image else clip if clip and not clip_is_url else ''
        include['on'] = bool(clip_text) and not url and not files
    pick()
    mode['thought'] = not (src['image'] or url or files or clip_text)

    P, W, pad = PALETTE, _u(600), _u(20)
    root.title('Grab')
    root.configure(bg=P['bg'])
    root.resizable(False, False)
    if os.name == 'nt':
        root.overrideredirect(True)
    canvas = tk.Canvas(root, width=W, highlightthickness=0, bd=0, bg=P['bg'])
    canvas.pack(fill='both', expand=True)

    # Composer: no box, themed caret and selection.
    y = _u(14)
    entry = tk.Entry(root, font=_role('composer'), fg=P['text'], bg=P['bg'], insertbackground=P['accent'], insertwidth=2,
                     selectbackground=P['panel_line'], selectforeground=P['text'], relief='flat', bd=0, highlightthickness=0)
    canvas.create_window(pad, y, window=entry, anchor='nw', width=W - 2 * pad, height=_u(34))
    default_placeholder = "Write what's on your mind…" if mode['thought'] else 'Why are you saving this?'
    hint = {'on': False}

    # Groups: a token field under the reason field; clicking it (or the shortcut) opens a popover list with search, "+" and
    # checkmarked rows. The groups of the last capture stay selected for 60 minutes and the reason used with exactly that
    # set shows as the main field's placeholder.
    tf_y, tf_h, panel_w, row_h = y + _u(42), _u(22), _u(280), _u(28)
    selected = active_groups(vault)
    known = list(dict.fromkeys(recent_groups(vault) + selected))
    counts = group_counts(vault)

    def show_hint():
        entry.delete(0, 'end')
        entry.insert(0, (not mode['thought'] and set_reason(vault, selected)) or default_placeholder)
        entry.configure(fg=P['faint'])
        entry.icursor(0)
        hint['on'] = True

    def on_key(event):
        # Embedded windows sit above canvas items, so the hint lives inside the field itself.
        if event.state & 0x4 or (sys.platform == 'darwin' and event.state & 0x8):
            return None  # Ctrl/Cmd shortcuts are not typing
        if hint['on'] and (event.char and event.char.isprintable() or event.keysym in ('BackSpace', 'Delete')):
            entry.delete(0, 'end')
            entry.configure(fg=P['text'])
            hint['on'] = False
            if event.keysym in ('BackSpace', 'Delete'):
                return 'break'
        if hint['on'] and event.keysym in ('Left', 'Right', 'Home', 'End'):
            return 'break'

    def show_why(text):  # snapshot aid: put typed text in the composer
        hint['on'] = False
        entry.delete(0, 'end')
        entry.insert(0, text)
        entry.configure(fg=P['text'])

    def after_key(_event=None):
        if not hint['on'] and not entry.get():
            show_hint()
    def on_paste(_event=None):
        if hint['on']:
            entry.delete(0, 'end')
            entry.configure(fg=P['text'])
            hint['on'] = False
    entry.bind('<KeyPress>', on_key)
    entry.bind('<<Paste>>', on_paste, add='+')
    entry.bind('<KeyRelease>', after_key)
    entry.bind('<Button-1>', lambda _e: (entry.focus_set(), entry.icursor(0), 'break')[-1] if hint['on'] else None)
    show_hint()
    bind_native_editing(entry, lambda: hint['on'])

    import tkinter.font as tkfont
    ui = {'open': False, 'create': False, 'hl': 0, 'top': 0, 'base_h': 0, 'pill': None, 'keycaps': {}}
    msg_var, search_var, name_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
    panel_top = tf_y + tf_h + _u(4)
    box_h = _u(28)
    popover_fixed = _u(8) + box_h + _u(4) + _u(2) + row_h + _u(8) + 2  # popover height without its list rows
    # The popover: a frame over the card, anchored under the "+" pill. Search, rows with counts, "+ New group" at the bottom.
    panel = tk.Frame(root, bg=P['panel'], highlightthickness=1, highlightbackground=P['panel_line'],
                     highlightcolor=P['panel_line'], name='panel')  # one hairline; the focus ring belongs to the focused field
    top_row = tk.Frame(panel, bg=P['panel'])
    top_row.pack(fill='x', padx=_u(8), pady=(_u(8), 0))
    box = tk.Canvas(top_row, height=box_h, bg=P['panel'], highlightthickness=0, name='box')  # search field
    search = tk.Entry(box, textvariable=search_var, font=_role('body'), fg=P['text'], bg=P['bg'], insertbackground=P['accent'],
                      selectbackground=P['panel_line'], selectforeground=P['text'], relief='flat', bd=0, highlightthickness=0, name='search')
    msg = tk.Label(panel, textvariable=msg_var, font=_role('secondary'), fg=P['warn'], bg=P['panel'], anchor='w')
    lst = tk.Canvas(panel, width=panel_w - 2, height=row_h, bg=P['panel'], highlightthickness=0, name='list')
    lst.pack(fill='x', padx=0, pady=(_u(4), 0))
    bottom = tk.Frame(panel, bg=P['panel'])
    bottom.pack(fill='x', padx=_u(8), pady=(_u(2), _u(8)))
    new_row = tk.Canvas(bottom, height=row_h, bg=P['panel'], highlightthickness=0, name='newrow')  # "+ New group"
    create_row = tk.Frame(bottom, bg=P['panel'])  # replaces it in create mode: inline name field + Create
    create_btn = tk.Canvas(create_row, width=_u(84), height=box_h, bg=P['panel'], highlightthickness=0, name='create')
    box2 = tk.Canvas(create_row, height=box_h, bg=P['panel'], highlightthickness=0, name='box2')
    name_entry = tk.Entry(box2, textvariable=name_var, font=_role('body'), fg=P['text'], bg=P['bg'], insertbackground=P['accent'],
                          selectbackground=P['panel_line'], selectforeground=P['text'], relief='flat', bd=0, highlightthickness=0, name='yeni')
    search_hint = tk.Label(box, text='Search or create a group', font=_role('body'), fg=P['muted'], bg=P['bg'])
    name_hint = tk.Label(box2, text='New group name', font=_role('body'), fg=P['muted'], bg=P['bg'])

    def draw_button(widget, label, fill, color, size):
        widget.delete('all')
        _round_rect(widget, 1, 1, int(widget['width']) - 1, box_h - 1, _u(8), fill=fill, outline=P['sel_line'] if fill == P['sel'] else fill)
        widget.create_text(int(widget['width']) / 2, box_h / 2, text=label, fill=color, font=_role(size))
        widget.fill = fill
    draw_button(create_btn, 'Create', P['sel'], P['muted'], 'secondary')

    def draw_new_row(hover=False):
        new_row.delete('all')
        if hover:
            _round_rect(new_row, 0, 1, new_row.winfo_width() if new_row.winfo_width() > 1 else panel_w - _u(16), row_h - 1, _u(6),
                        fill=P['panel_sel'], outline='')
        _plus(new_row, _u(16), row_h / 2, _u(4), P['accent'], 'ic')
        new_row.create_text(_u(34), row_h / 2, text='New group…', anchor='w', fill=P['soft'], font=_role('body'))
    draw_new_row()
    box_items = {}

    def draw_box(_event=None):
        focused = root.focus_get()
        for widget, field, key in ((box, search, 'sw'), (box2, name_entry, 'nw')):
            widget.delete('frame')
            width = widget.winfo_width() if widget.winfo_width() > 1 else panel_w - _u(16) - (_u(92) if widget is box2 else 0)
            _round_rect(widget, 1, 1, width - 1, box_h - 1, _u(8), fill=P['bg'], tags='frame',
                        outline=P['focus'] if focused is field else P['card_line'])
            widget.tag_lower('frame')
            left = _u(30) if widget is box else _u(10)
            widget.coords(box_items[key], left, box_h / 2)
            widget.itemconfigure(box_items[key], width=max(40, width - left - _u(8)))
    box.create_oval(_u(10), _u(9), _u(19), _u(18), outline=P['muted'], width=2, tags='mag')
    box.create_line(_u(18), _u(17), _u(22), _u(21), fill=P['muted'], width=2, tags='mag')
    box_items['sw'] = box.create_window(_u(30), box_h / 2, window=search, anchor='w', height=_u(22))
    box_items['nw'] = box2.create_window(_u(10), box_h / 2, window=name_entry, anchor='w', height=_u(22))
    for widget in (box, box2):
        widget.bind('<Configure>', draw_box)
    for field in (search, name_entry):
        field.bind('<FocusIn>', lambda _e: draw_box(), add='+')
        field.bind('<FocusOut>', lambda _e: draw_box(), add='+')

    def sync_search_hint():
        if search_var.get():
            search_hint.place_forget()
        else:
            search_hint.place(in_=search, x=2, y=0, relheight=1, anchor='nw')
            search_hint.bind('<Button-1>', lambda _e: search.focus_set())

    def visible_rows():
        """Rows the popover shows: what fits under the pill without growing the window, at least 1, at most 8; the window is built tall enough for three."""
        return max(1, min(8, (ui['base_h'] - panel_top - _u(4) - popover_fixed) // row_h)) if ui['base_h'] else 8

    def panel_x():
        left = ui['pill'][0] if ui['pill'] else pad
        return int(max(pad, min(left, W - pad - panel_w)))

    def rows():
        text = fold(group_name(search_var.get()))
        found = [('g', name) for name in known if text in fold(name)]
        if text and not found:
            found.append(('new', group_name(search_var.get())))
        return found

    def draw_tokens():
        """One compact row: the selected groups as tokens, then a small tag-style "+" pill (never a text field)."""
        canvas.delete('tok')
        font, small = _role('body'), _role('secondary')
        measure = tkfont.Font(root=root, font=font).measure
        small_measure = tkfont.Font(root=root, font=small).measure
        cy = tf_y + tf_h / 2
        label = '+' if selected else '+ Group'
        pill_w = max(tf_h, small_measure(label) + _u(22))
        loop_w = small_measure(LOOP_LABEL) + _u(26)
        x, limit = pad, W - pad - loop_w - _u(8)
        for index, name in enumerate(selected):
            width = measure(name) + _u(32)
            left = len(selected) - index - 1
            if x + width > limit - pill_w - _u(6) - (measure('+%d' % left) + _u(8) if left else 0):
                canvas.create_text(x + _u(2), cy, text='+%d' % (len(selected) - index), anchor='w', fill=P['muted'],
                                   font=font, tags=('tok',))
                x += measure('+%d' % (len(selected) - index)) + _u(10)
                break
            tag = 't%d' % index
            _round_rect(canvas, x, tf_y, x + width, tf_y + tf_h, _u(6), fill=P['sel'], outline=P['sel_line'], tags=('tok', tag))
            canvas.create_text(x + _u(11), cy, text=name, anchor='w', fill=P['text'], font=font, tags=('tok', tag))
            cx = x + width - _u(12)
            _cross(canvas, cx, cy, _u(3), P['soft'], ('tok', tag, tag + 'xl'))
            canvas.itemconfigure(tag + 'xl', state='hidden')  # the x shows on hover only
            canvas.tag_bind(tag, '<Enter>', lambda _e, t=tag: canvas.itemconfigure(t + 'xl', state='normal'))
            canvas.tag_bind(tag, '<Leave>', lambda _e, t=tag: canvas.itemconfigure(t + 'xl', state='hidden'))
            canvas.tag_bind(tag, '<Button-1>', lambda e, n=name, edge=cx - _u(9): remove_token(n) if e.x >= edge else open_panel())
            x += width + _u(6)
        ui['pill'] = (x, tf_y, x + pill_w, tf_y + tf_h)
        opened = ui['open']
        _round_rect(canvas, x, tf_y, x + pill_w, tf_y + tf_h, tf_h / 2, fill=P['sel'] if opened else P['card'],
                    outline=P['focus'] if opened else P['key_line'], tags=('tok', 'pill', 'pillbg'))
        canvas.create_text(x + pill_w / 2, cy, text=label, fill=P['text'] if opened else P['soft'], font=small, tags=('tok', 'pill'))
        canvas.tag_bind('pill', '<Button-1>', lambda _e: toggle_panel())
        canvas.tag_bind('pill', '<Enter>', lambda _e: canvas.itemconfigure('pillbg', fill=P['sel'], outline=P['focus']))
        canvas.tag_bind('pill', '<Leave>', lambda _e: canvas.itemconfigure('pillbg', fill=P['sel'] if ui['open'] else P['card'],
                                                                         outline=P['focus'] if ui['open'] else P['key_line']))
        draw_loop(loop_w)
        if ui['open'] and ui.get('panel_h'):
            panel.place_configure(x=panel_x())

    def draw_loop(width):
        """The "Open loop" toggle at the right end of the group row."""
        canvas.delete('loop')
        on, cy = loop['on'], tf_y + tf_h / 2
        left = W - pad - width
        _round_rect(canvas, left, tf_y, W - pad, tf_y + tf_h, tf_h / 2, fill=P['accent'] if on else P['card'],
                    outline=P['accent'] if on else P['key_line'], tags=('loop', 'loopbg'))
        canvas.create_text(left + width / 2, cy, text=LOOP_LABEL, fill=P['accent_text'] if on else P['soft'],
                           font=_role('secondary'), tags=('loop',))
        canvas.tag_bind('loop', '<Button-1>', lambda _e: toggle_loop())

    def toggle_loop(_event=None):
        loop['on'] = not loop['on']
        draw_tokens()
        return 'break'

    def draw_list():
        lst.delete('all')
        found = rows()
        shown = visible_rows()
        ui['hl'] = max(0, min(ui['hl'], len(found) - 1))
        ui['top'] = max(0, min(ui['top'], max(0, len(found) - shown)))
        if ui['hl'] < ui['top']:
            ui['top'] = ui['hl']
        elif ui['hl'] >= ui['top'] + shown:
            ui['top'] = ui['hl'] - shown + 1
        window = found[ui['top']:ui['top'] + shown]
        lst.configure(height=max(row_h, len(window) * row_h))
        if ui['open']:
            resize_panel()
        font = _role('body')
        if not found:
            lst.create_text(_u(16), row_h / 2, text='No groups yet' if not known else 'No match', anchor='w',
                            fill=P['muted'], font=font)
        for n, (kind, name) in enumerate(window):
            y0 = n * row_h
            if ui['top'] + n == ui['hl'] and not ui['create']:
                _round_rect(lst, _u(6), y0 + 1, panel_w - _u(8), y0 + row_h - 1, _u(6), fill=P['panel_sel'], outline='')
            if kind == 'new':
                lst.create_text(_u(16), y0 + row_h / 2, text='Create "%s"' % name, anchor='w', fill=P['text'], font=font)
                lst.create_text(panel_w - _u(18), y0 + row_h / 2, text='↵', anchor='e', fill=P['muted'], font=_role('secondary'))
                continue
            if name in selected:
                _check(lst, _u(20), y0 + row_h / 2, _u(5), P['accent'], 'ic')
            lst.create_text(_u(34), y0 + row_h / 2, text=name, anchor='w', fill=P['text'], font=font)
            lst.create_text(panel_w - _u(18), y0 + row_h / 2, text=str(counts.get(name, 0)), anchor='e', fill=P['muted'], font=_role('secondary'))

    def resize_panel():
        """Popover = search row + rows + "+ New group" (+ message line), drawn over the card. The window grows only when even
        three rows would not fit under the pill (never)."""
        if msg_var.get():
            if not msg.winfo_manager():
                msg.pack(fill='x', padx=_u(12), pady=(_u(4), 0), before=lst)
        elif msg.winfo_manager():
            msg.pack_forget()
        panel.update_idletasks()
        height = panel.winfo_reqheight()
        ui['panel_h'] = height
        panel.place(x=panel_x(), y=panel_top, width=panel_w, height=height)
        need = max(ui['base_h'], panel_top + height + _u(4))
        root.geometry('%dx%d' % (W, need))
        canvas.configure(height=need)

    def refresh():
        draw_tokens()
        if ui['open']:
            draw_list()
        if hint['on']:
            show_hint()

    def toggle_group(name):
        if name in selected:
            selected.remove(name)
        else:
            selected.append(name)
        refresh()

    def remove_token(name):
        if name in selected:
            selected.remove(name)
        refresh()

    def set_create_mode(on):
        ui['create'] = on
        if on:
            name_var.set(group_name(search_var.get()))
            name_entry.icursor('end')
            new_row.pack_forget()
            create_row.pack(fill='x')
            panel.update_idletasks()
            draw_box()
            name_entry.focus_set()
            update_button()
        else:
            name_hint.place_forget()
            create_row.pack_forget()
            new_row.pack(fill='x')
            draw_box()
            if ui['open']:
                search.focus_set()
        if ui['open']:
            draw_list()

    def update_button():
        ready = bool(group_name(name_var.get()))
        if name_var.get():
            name_hint.place_forget()
        else:
            name_hint.place(in_=name_entry, x=2, rely=0.5, anchor='w')
        draw_button(create_btn, 'Create', P['accent'] if ready else P['sel'], P['accent_text'] if ready else P['muted'], 'secondary')

    def create_group(raw):
        """Create and select a group. A name that already exists (Turkish-aware, after trim) is selected instead."""
        name = group_name(raw)
        if not name:
            return None
        existing = next((known_name for known_name in known if fold(known_name) == fold(name)), None)
        if existing:
            msg_var.set('Already exists, selected')
            name = existing
        else:
            name = group_create(vault, name)
            known.insert(0, name)
            counts[name] = 0
            msg_var.set('')
        if name not in selected:
            selected.append(name)
        if name in known and known.index(name) >= RECENT_TOP:
            known.remove(name)
            known.insert(0, name)
        search_var.set('')
        ui.update(hl=0, top=0)
        set_create_mode(False)
        refresh()
        return name

    def submit_name(_event=None):
        create_group(name_var.get())
        return 'break'

    def activate(_event=None):
        found = rows()
        if not found:
            return 'break'
        kind, name = found[min(ui['hl'], len(found) - 1)]
        if kind == 'new':
            create_group(name)
        else:
            toggle_group(name)
        return 'break'

    def move(delta):
        count = len(rows())
        if count:
            ui['hl'] = (ui['hl'] + delta) % count
        draw_list()
        return 'break'

    def space(_event):
        return activate() if not search_var.get() else None  # inside a name once something is typed

    def backspace(_event):
        if search_var.get():
            return None
        if selected:
            selected.pop()
            refresh()
        return 'break'

    def typed(event):
        if event.keysym not in ('Up', 'Down', 'Return', 'Escape', 'Left', 'Right', 'Shift_L', 'Shift_R'):
            msg_var.set('')
            ui.update(hl=0, top=0)
            draw_list()

    def open_panel(_event=None):
        if ui['open']:
            return 'break'
        ui['open'] = True
        ui.update(hl=0, top=0)
        set_create_mode(False)
        panel.lift()
        draw_tokens()
        draw_list()
        search.focus_set()
        sync_search_hint()
        return 'break'

    def close_panel(focus_main=True):
        if not ui['open']:
            return
        ui['open'] = False
        panel.place_forget()
        msg_var.set('')
        search_var.set('')
        if ui['base_h']:
            root.geometry('%dx%d' % (W, ui['base_h']))
            canvas.configure(height=ui['base_h'])
        draw_tokens()
        if focus_main:
            entry.focus_set()

    def toggle_panel(_event=None):
        close_panel() if ui['open'] else open_panel()
        return 'break'

    def search_escape(_event=None):
        close_panel()
        return 'break'

    def name_escape(_event=None):
        name_var.set('')
        set_create_mode(False)
        return 'break'

    def outside(event):
        if not ui['open']:
            return
        if str(event.widget).startswith(str(panel)):
            return
        if event.widget is canvas and (pad <= event.x <= W - pad and tf_y <= event.y <= tf_y + tf_h
                                       or any('kc' in canvas.gettags(i) for i in canvas.find_overlapping(event.x, event.y, event.x, event.y))):
            return
        close_panel(False)

    def wheel(event):
        ui['top'] = max(0, ui['top'] - (1 if event.delta > 0 else -1))
        found = len(rows())
        ui['top'] = min(ui['top'], max(0, found - visible_rows()))
        ui['hl'] = max(ui['top'], min(ui['hl'], ui['top'] + visible_rows() - 1))
        draw_list()

    def row_hover(event):
        index = ui['top'] + event.y // row_h
        if 0 <= index < len(rows()) and index != ui['hl']:
            ui['hl'] = index
            draw_list()

    def row_click(event):
        found = rows()
        index = ui['top'] + event.y // row_h
        if 0 <= index < len(found):
            ui['hl'] = index
            kind, name = found[index]
            create_group(name) if kind == 'new' else toggle_group(name)

    for key, handler in (('<Up>', lambda _e: move(-1)), ('<Down>', lambda _e: move(1)), ('<Return>', activate),
                         ('<KP_Enter>', activate), ('<space>', space), ('<BackSpace>', backspace),
                         ('<Escape>', search_escape), ('<Tab>', lambda _e: 'break'), ('<KeyRelease>', typed)):
        search.bind(key, handler)
    for key, handler in (('<Return>', submit_name), ('<KP_Enter>', submit_name), ('<Escape>', name_escape),
                         ('<Tab>', lambda _e: 'break'), ('<KeyRelease>', lambda _e: update_button())):
        name_entry.bind(key, handler)
    bind_native_editing(search)
    bind_native_editing(name_entry)
    search_var.trace_add('write', lambda *_: (sync_search_hint(), draw_list() if ui['open'] else None))
    name_var.trace_add('write', lambda *_: update_button() if ui['create'] else None)
    name_hint.bind('<Button-1>', lambda _e: name_entry.focus_set())
    new_row.bind('<Button-1>', lambda _e: set_create_mode(True))
    new_row.bind('<Enter>', lambda _e: draw_new_row(True))
    new_row.bind('<Leave>', lambda _e: draw_new_row(False))
    create_btn.bind('<Button-1>', submit_name)
    msg_var.trace_add('write', lambda *_: resize_panel() if ui['open'] else None)
    lst.bind('<Button-1>', row_click)
    lst.bind('<Motion>', row_hover)
    lst.bind('<MouseWheel>', wheel)
    entry.bind('<FocusIn>', lambda _e: close_panel(False), add='+')
    root.bind('<Button-1>', outside, add='+')
    root.bind('<Command-g>' if sys.platform == 'darwin' else '<Control-g>', toggle_panel)
    root.bind('<Command-t>' if sys.platform == 'darwin' else '<Control-t>', toggle_loop)
    box.pack(fill='x')
    new_row.pack(fill='x')
    create_btn.pack(side='right', padx=(_u(8), 0))
    box2.pack(side='left', fill='x', expand=True)
    draw_tokens()
    root.grab = types.SimpleNamespace(  # handles for the tests
        selected=selected, known=known, ui=ui, open=open_panel, close=close_panel, toggle=toggle_group, remove=remove_token,
        create=create_group, search=_EndVar(search_var, search), name=_EndVar(name_var, name_entry), rows=rows, msg=msg_var, search_entry=search, name_entry=name_entry,
        loop=loop, toggle_loop=toggle_loop, mode=mode, pill=lambda: ui['pill'], why=show_why, canvas=canvas, new_row=new_row, create_row=create_row, visible_rows=visible_rows,
        create_button=create_btn, button_fill=lambda: create_btn.fill, hint_shown=lambda: bool(name_hint.winfo_manager()), entry=entry, panel=panel, enter_create=lambda: set_create_mode(True))
    foot_h = _u(36)
    y = tf_y + tf_h + _u(12)
    canvas.create_line(pad, y, W - pad, y, fill=P['line'])
    y += _u(10)

    # Source row: flat, no card. Kind and domain in secondary, the cleaned title on one ellipsized line.
    preview_y = y
    card_h = _u(40)

    def draw_preview():
        canvas.delete('prev')
        if mode['thought']:
            kind, domain, headline = 'Thought', '', 'Stored as you typed it · nothing is processed'
        elif src['image']:
            kind, domain, headline = 'IMAGE', '', 'Image (clipboard) · %d KB' % (len(image) // 1024 + 1)
        elif files:
            kind, domain = 'FILE', str(Path(files[0]).parent)
            headline = Path(files[0]).name + ('  +' + str(len(files) - 1) if len(files) > 1 else '')
        elif url:
            kind = KIND_LABEL.get(kind_of(url), 'PAGE')
            domain = (urlparse(url).hostname or '').removeprefix('www.')
            headline = clean_title(context.get('title')) or url
        else:
            kind, domain, headline = 'TEXT', '', _first_line(clip_text, 200)
        room = W - 2 * pad
        canvas.create_text(pad, preview_y + _u(9), anchor='w', fill=P['muted'], font=_role('secondary'), tags='prev',
                           text=_fit(canvas, kind + ('  ·  ' + domain if domain else ''), _role('secondary'), room))
        canvas.create_text(pad, preview_y + _u(27), anchor='w', fill=P['text'], font=_role('body'), tags='prev',
                           text=_fit(canvas, headline, _role('body'), room))
    draw_preview()
    y += card_h

    # Optional rows, one compact checkbox line each, only when relevant.
    lines, line_y = [], [y]

    def checkline(name, label, is_on, flip):
        top, tag = line_y[0], 'cl_' + name

        def draw():
            canvas.delete(tag)
            on = is_on()
            _round_rect(canvas, pad, top + _u(5), pad + _u(15), top + _u(20), _u(4), fill=P['accent'] if on else P['bg'],
                        outline=P['accent'] if on else P['faint'], tags=(tag,))
            if on:
                _check(canvas, pad + _u(7.5), top + _u(12.5), _u(3.5), P['accent_text'], (tag,))
            text = canvas.create_text(pad + _u(25), top + _u(12.5), anchor='w', fill=P['soft'], font=_role('secondary'), tags=(tag,),
                                      text=_fit(canvas, label, _role('secondary'), W - 2 * pad - _u(25)))
            canvas.tag_bind(tag, '<Button-1>', lambda _e: flip())
            canvas.tag_bind(tag, '<Enter>', lambda _e: canvas.itemconfigure(text, fill=P['text']))
            canvas.tag_bind(tag, '<Leave>', lambda _e: canvas.itemconfigure(text, fill=P['soft']))
        lines.append(draw)
        draw()
        line_y[0] += _u(26)

    def redraw_lines():
        for draw in lines:
            draw()
    tab_action = None

    def toggle_thought(_event=None):
        """Cmd+D: save the typed text as a thought, ignoring the source; again to bring the source back."""
        if not (src['image'] or url or files or clip_text):
            return 'break'  # nothing else to choose
        mode['thought'] = not mode['thought']
        draw_preview()
        if hint['on']:
            show_hint()
        return 'break'
    root.bind('<Command-d>' if sys.platform == 'darwin' else '<Control-d>', toggle_thought)
    if clip_text and (url or files):
        def toggle(_event=None):
            mode['thought'] = False
            include['on'] = not include['on']
            redraw_lines()
            return 'break'
        checkline('clip', 'Also add the clipboard text · ' + _first_line(clip_text, 120), lambda: include['on'], toggle)
        tab_action = toggle
    if both:
        def switch(_event=None):
            mode['thought'] = False
            src['image'] = not src['image']
            pick()
            draw_preview()
            redraw_lines()
            return 'break'
        checkline('image', 'Save the clipboard image', lambda: src['image'], switch)
        tab_action = switch
    if tab_action:
        root.bind('<Tab>', tab_action)
    y = line_y[0]

    # Footer action bar on a sunken strip: transient status (or "Grab · app") left, clickable actions right.
    height = max(y + _u(10) + foot_h, panel_top + popover_fixed + min(max(len(known), 1), 3) * row_h + _u(4))  # never grows
    ui['base_h'] = height
    canvas.configure(height=height)
    footer_top = height - foot_h
    footer_y = footer_top + foot_h / 2
    canvas.create_rectangle(0, footer_top, W, height, fill=P['sunk'], outline='')
    canvas.create_line(0, footer_top, W, footer_top, fill=P['line'])
    status_room = {'w': W}  # width left of the first action, set once the actions are placed
    app = context.get('app') or ''
    idle_text = 'Grab' + (' · ' + app if app else '')

    def fitted(text):
        return _fit(canvas, text, _role('secondary'), status_room['w'])
    status = canvas.create_text(pad, footer_y, anchor='w', fill=P['muted'], font=_role('secondary'), text='')
    small_font = tkfont.Font(root=root, font=_role('secondary'))
    key_font = tkfont.Font(root=root, font=_mono_role('caption'))

    def action(x_right, name, label, key, command, primary=False):
        """label + key as one clickable group (tags kc, kc_<name>) with a hover wash; returns the x left of it."""
        inner, key_w = _u(8), key_font.measure(key) + _u(12)
        left = x_right - (inner + small_font.measure(label) + _u(8) + key_w + inner)
        tag = 'kc_' + name
        tags = ('kc', tag)
        wash = _round_rect(canvas, left, footer_y - _u(13), x_right, footer_y + _u(13), _u(6), fill=P['sunk'], outline='', tags=tags)
        canvas.create_text(left + inner, footer_y, text=label, anchor='w', fill=P['text'] if primary else P['soft'],
                           font=_role('secondary'), tags=tags)
        _round_rect(canvas, x_right - inner - key_w, footer_y - _u(10), x_right - inner, footer_y + _u(10), _u(5),
                    fill=P['accent'] if primary else P['key'], outline=P['accent'] if primary else P['key_line'], tags=tags)
        canvas.create_text(x_right - inner - key_w / 2, footer_y, text=key, font=_mono_role('caption'), tags=tags,
                           fill=P['accent_text'] if primary else P['soft'])
        canvas.tag_bind(tag, '<Button-1>', lambda _e: command())
        canvas.tag_bind(tag, '<Enter>', lambda _e: canvas.itemconfigure(wash, fill=P['panel_sel']))
        canvas.tag_bind(tag, '<Leave>', lambda _e: canvas.itemconfigure(wash, fill=P['sunk']))
        ui['keycaps'][name] = canvas.bbox(tag)
        return left - _u(4)
    right = W - _u(12)
    right = action(right, 'save', 'Save', '↵', lambda: save(), primary=True)
    right = action(right, 'groups', 'Groups', '⌘G' if sys.platform == 'darwin' else 'Ctrl+G', lambda: toggle_panel())
    status_room['w'] = right - pad

    def say(text, color=None):
        canvas.itemconfigure(status, text=fitted(text), fill=color or P['muted'])
    say(idle_text)
    _round_rect(canvas, 0, 0, W - 1, height - 1, _u(12), fill='', outline=P['line'], tags='border', state='disabled')  # disabled: never takes a click
    result = {}

    def close():
        def fade(alpha):
            if alpha <= 0:
                root.destroy()
                return
            try:
                root.attributes('-alpha', alpha)
            except tk.TclError:
                root.destroy()
                return
            root.after(16, fade, round(alpha - 0.12, 2))
        fade(1.0)

    def save(_event=None):
        url, files, text, shot = current()
        typed = '' if hint['on'] else entry.get()
        why = typed.strip()
        thought = mode['thought']
        if thought and not why:
            say('Write something first', P['warn'])
            return 'break'
        say('Saving…')
        root.update_idletasks()
        try:
            if thought:  # verbatim, no network, nothing started
                result.update(capture(vault, text=typed, app=context.get('app'), groups=list(selected), open=loop['on'], thought=True))
            else:
                result.update(capture(vault, url=url, text=text, files=files, why=why, app=context.get('app'),
                                      title=context.get('title') if url else None, image=shot,
                                      groups=list(selected), open=loop['on']))
        except Exception as exc:
            say('Save_failed: ' + str(exc), P['error'])
            return 'break'
        if result.get('status') == 'captured' and not thought:
            spawn_process(vault, result['id'])  # detached: the popup never waits for the network
        entry.configure(state='disabled', disabledbackground=P['bg'], disabledforeground=P['muted'])
        say('Saved', P['ok'])
        if not snapshot:
            root.after(520, close)
        return 'break'

    root.grab.save = save
    root.grab.say = say
    root.grab.status_item = status
    root.bind('<Return>', save)
    root.bind('<KP_Enter>', save)
    root.bind('<Escape>', lambda _e: close())

    root.update_idletasks()
    x = (root.winfo_screenwidth() - W) // 2
    top = max(80, int(root.winfo_screenheight() * 0.22))
    root.geometry('%dx%d+%d+%d' % (W, height, -30000 if unseen else x, top))
    try:
        root.attributes('-alpha', 0.0)
        root.attributes('-topmost', not unseen)
    except tk.TclError:
        pass
    root.deiconify()
    if sys.platform == 'darwin':
        _style_mac(focus=not snapshot)
    if not snapshot:
        root.lift()
        root.focus_force()
        entry.focus_set()

    def fade_in(alpha=0.0):
        alpha = min(1.0, alpha + 0.2)
        try:
            root.attributes('-alpha', alpha)
        except tk.TclError:
            return
        if alpha < 1.0:
            root.after(14, fade_in, alpha)
    if not unseen:
        fade_in()
    if snapshot and sys.platform == 'darwin':
        # Development aid: capture only this window's rectangle without taking focus, then close.
        def steps():
            # MEMROVE_GRAB_SNAPSHOT_STEPS="open|type:ai|plus|name:x": put the window in a state before the shot.
            api = root.grab
            for step in filter(None, os.environ.get('MEMROVE_GRAB_SNAPSHOT_STEPS', '').split('|')):
                kind, _, arg = step.partition(':')
                if kind == 'open':
                    api.open()
                elif kind == 'type':
                    api.search.set(arg)
                elif kind == 'plus':
                    api.enter_create()
                elif kind == 'name':
                    api.name.set(arg)
                elif kind == 'pick':
                    api.toggle(arg)
                elif kind == 'why':
                    api.why(arg)
                elif kind == 'loop':
                    api.toggle_loop()
                elif kind == 'save':
                    api.save()
                root.update()
        root.after(300, steps)

        def shoot():
            # The window stays transparent until now; then it sinks below the wallpaper, where screencapture -l still reads it.
            _window_number(under_desktop=True)
            root.attributes('-alpha', 1.0)  # fully opaque but below the wallpaper: hidden from the owner, readable by -l
            root.update()
            root.after(300)
            root.update()
            for _ in range(10):  # the window has no number until the window server has it on screen
                number = _window_number()
                if number > 0 and not Path(snapshot).exists():
                    subprocess.run(['screencapture', '-x', '-o', '-l', str(number), snapshot], capture_output=True)
                root.after(150)
                root.update()
            root.destroy()
        root.after(700, shoot)
    root.mainloop()
    return result or {'status': 'cancelled'}


def _popup_fallback(vault, context):
    """No tkinter (some Homebrew/Linux Pythons): the OS dialog asks only for the reason."""
    why = ''
    if sys.platform == 'darwin':
        why = _osascript('text returned of (display dialog "Why are you saving this?" default answer "" with title "memrove Grab")', timeout=300) or ''
    image = clipboard_image()
    if choose_source(image, bool(context.get('url') or context.get('files'))) != 'image':
        image = None
    url = None if image else context.get('url')
    files = [] if image else context.get('files') or []
    if not (url or files or image or why):
        return {'status': 'cancelled'}
    return capture(vault, url=url, text=None if url or files or image else why, files=files,
                   why=why if url or files or image else '', app=context.get('app'), title=context.get('title'), image=image)


# ---------------------------------------------------------------- hotkey spec

DEFAULT_HOTKEY = 'ctrl+alt+b'
MODIFIERS = {'cmd': 'cmd', 'command': 'cmd', '⌘': 'cmd', 'super': 'cmd', 'win': 'cmd',
             'ctrl': 'ctrl', 'control': 'ctrl', 'ctl': 'ctrl', '⌃': 'ctrl',
             'alt': 'alt', 'opt': 'alt', 'option': 'alt', '⌥': 'alt',
             'shift': 'shift', '⇧': 'shift'}
MAC_BITS = {'cmd': 0x100, 'shift': 0x200, 'alt': 0x800, 'ctrl': 0x1000}
MAC_SYMBOL = {'ctrl': '⌃', 'alt': '⌥', 'shift': '⇧', 'cmd': '⌘'}
MAC_NAMED = {'space': 49, 'space': 49, 'return': 36, 'enter': 36, 'tab': 48, 'esc': 53, 'escape': 53,
             'f1': 122, 'f2': 120, 'f3': 99, 'f4': 118, 'f5': 96, 'f6': 97, 'f7': 98, 'f8': 100,
             'f9': 101, 'f10': 109, 'f11': 103, 'f12': 111}
# ANSI positions, used only when the live keyboard layout cannot be read.
MAC_ANSI = dict(zip('asdfhgzxcv bqweryt123465=97-80]ou[ip lj\'k;\\,/nm.', range(50)))
MAC_ANSI.pop(' ', None)


def parse_hotkey(spec):
    """'cmd+"' / 'ctrl+alt+b' / '⌘⇧K' -> (sorted modifiers, key). The key is the last part."""
    text = str(spec or '').strip()
    if not text:
        raise ValueError('The hotkey cannot be empty; example: ctrl+alt+b or cmd+"')
    for symbol in '⌘⌃⌥⇧':
        text = text.replace(symbol, MAC_SYMBOL_REVERSE[symbol] + '+')
    parts = text.split('+')
    if text.endswith('++'):  # the key itself is '+'
        parts = parts[:-2] + ['+']
    key = parts[-1].strip()
    mods = []
    for part in parts[:-1]:
        name = MODIFIERS.get(part.strip().lower())
        if not name:
            raise ValueError('Unknown key: ' + part.strip() + ' (cmd, ctrl, alt/option, shift)')
        if name not in mods:
            mods.append(name)
    if not key:
        raise ValueError('The hotkey has no key; example: cmd+"')
    if len(key) > 1:
        key = key.lower()
        if key not in MAC_NAMED:
            raise ValueError('Unknown key name: ' + key)
    if not mods and not re.fullmatch(r'f\d{1,2}', key):
        raise ValueError('At least one modifier key is needed (cmd, ctrl, alt, shift)')
    order = ['ctrl', 'alt', 'shift', 'cmd']
    return sorted(mods, key=order.index), key


KEYPAD = {65, 67, 69, 71, 75, 76, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92}
MAC_SYMBOL_REVERSE = {'⌘': 'cmd', '⌃': 'ctrl', '⌥': 'alt', '⇧': 'shift'}


def _mac_layout_keycode(char):
    """Physical key that types `char` on the active layout (Turkish Q puts '"' left of 1)."""
    import ctypes
    from ctypes import POINTER, byref, c_uint8, c_uint16, c_uint32, c_ulong, c_void_p
    carbon = ctypes.CDLL('/System/Library/Frameworks/Carbon.framework/Carbon')
    cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    carbon.TISCopyCurrentKeyboardLayoutInputSource.restype = c_void_p
    carbon.TISGetInputSourceProperty.restype = c_void_p
    carbon.TISGetInputSourceProperty.argtypes = [c_void_p, c_void_p]
    cf.CFDataGetBytePtr.restype = c_void_p
    cf.CFDataGetBytePtr.argtypes = [c_void_p]
    carbon.LMGetKbdType.restype = c_uint8
    carbon.UCKeyTranslate.argtypes = [c_void_p, c_uint16, c_uint16, c_uint32, c_uint32, c_uint32,
                                      POINTER(c_uint32), c_ulong, POINTER(c_ulong), POINTER(c_uint16)]
    source = carbon.TISCopyCurrentKeyboardLayoutInputSource()
    data = carbon.TISGetInputSourceProperty(source, c_void_p.in_dll(carbon, 'kTISPropertyUnicodeKeyLayoutData'))
    if not data:
        return None
    layout, kind = cf.CFDataGetBytePtr(data), carbon.LMGetKbdType()
    for shift in (0, 2):  # unshifted first; 2 = shiftKey >> 8
        for code in (c for c in range(128) if c not in KEYPAD):
            dead, length, buffer = c_uint32(0), c_ulong(0), (c_uint16 * 4)()
            carbon.UCKeyTranslate(layout, code, 3, shift, kind, 1, byref(dead), 4, byref(length), buffer)
            typed = ''.join(chr(buffer[i]) for i in range(length.value))
            if typed == char or (len(char) == 1 and typed.lower() == char.lower() and char.isalpha() and not shift):
                return code, bool(shift)
    return None


def mac_hotkey(spec):
    """-> (keycode, carbon modifier bits, label such as ⌘\")."""
    mods, key = parse_hotkey(spec)
    if (set(mods) == {'cmd'} and key == '"') or (set(mods) == {'cmd', 'shift'} and key == "'"):
        raise ValueError('⌘" belongs to Shottr (area screenshot); pick another key. The default is ctrl+alt+b.')
    shifted = False
    if key in MAC_NAMED:
        code = MAC_NAMED[key]
    else:
        found = None
        try:
            found = _mac_layout_keycode(key)
        except (OSError, AttributeError, ValueError):
            found = None
        if found is None and key.lower() in MAC_ANSI:
            found = (MAC_ANSI[key.lower()], False)
        if found is None:
            raise ValueError('This character is not a key in the active keyboard layout: ' + key)
        code, shifted = found
    if shifted and 'shift' not in mods:
        mods.append('shift')  # the character itself needs Shift on this layout
    bits = sum(MAC_BITS[m] for m in mods)
    label = ''.join(MAC_SYMBOL[m] for m in ('ctrl', 'alt', 'shift', 'cmd') if m in mods) + \
        (key.upper() if len(key) == 1 else key.capitalize())
    return code, bits, label


def windows_hotkey(spec):
    """Start-menu shortcut keys accept Ctrl/Alt/Shift with a letter, digit or F-key; no Windows key."""
    mods, key = parse_hotkey(spec)
    if 'cmd' in mods:
        raise ValueError('Win/Cmd cannot be used in a Windows hotkey; use ctrl, alt and shift')
    if not re.fullmatch(r'[a-z0-9]|f\d{1,2}', key.lower()):
        raise ValueError('A Windows hotkey can only use a letter, a digit or an F key')
    if len([m for m in mods if m in ('ctrl', 'alt')]) == 0:
        raise ValueError('A Windows hotkey must include Ctrl or Alt')
    names = {'ctrl': 'CTRL', 'alt': 'ALT', 'shift': 'SHIFT'}
    value = '+'.join([names[m] for m in mods] + [key.upper()])
    return value, '+'.join([m.capitalize() for m in mods] + [key.upper()])


def saved_hotkey(state):
    try:
        return json.loads(_state_path(state).read_text(encoding='utf-8')).get('tus') or DEFAULT_HOTKEY
    except (OSError, ValueError, AttributeError):
        return DEFAULT_HOTKEY


# The hotkey itself lives in the resident Swift menu bar agent (MemroveGrab agent): see install().

def _state_path(state):
    return Path(state) / STATE_FILE


def _launch_agent():
    return Path.home() / 'Library/LaunchAgents' / (LAUNCH_LABEL + '.plist')


def _listener_ok(wait=1.2):
    """The listener exits at once when another app already owns the combination."""
    import time
    time.sleep(wait)
    probe = subprocess.run(['launchctl', 'print', 'gui/' + str(os.getuid()) + '/' + LAUNCH_LABEL], capture_output=True)
    return probe.returncode == 0 and b'state = running' in probe.stdout


def _agent_vault():
    """Vault the installed LaunchAgent serves, or None when this home has none."""
    import plistlib
    try:
        arguments = plistlib.loads(_launch_agent().read_bytes()).get('ProgramArguments', [])
        return arguments[arguments.index('--vault') + 1]
    except (OSError, ValueError, IndexError, plistlib.InvalidFileException):
        return None


def _windows_dirs():
    appdata = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming')))
    return appdata / 'Microsoft/Windows/Start Menu/Programs', appdata / 'Microsoft/Windows/SendTo'


def _ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _windows_shortcut(path, target, arguments, hotkey=None):
    import base64
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ['$s = (New-Object -ComObject WScript.Shell).CreateShortcut(' + _ps_quote(path) + ')',
             '$s.TargetPath = ' + _ps_quote(target), '$s.Arguments = ' + _ps_quote(arguments),
             '$s.Description = ' + _ps_quote('Grab: capture a source into the second brain'), '$s.WindowStyle = 7']
    if hotkey:
        lines.append('$s.Hotkey = ' + _ps_quote(hotkey))
    lines.append('$s.Save()')
    encoded = base64.b64encode('\n'.join(lines).encode('utf-16le')).decode('ascii')
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                   check=True, capture_output=True, timeout=60, creationflags=NO_WINDOW)


def _argline(values):
    return subprocess.list2cmdline([str(v) for v in values])


# ---------------------------------------------------------------- process: card -> snapshot + catalog + note

def snap_slug(text):
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower().rsplit('/', 1)[-1]).strip('-')  # same rule as `wiki source fetch`


def classify(url):
    """('repo', 'owner/name') or ('paper', 'arxiv id') when memrove_v3_sources.fetch can take the URL, else (None, None)."""
    parts = urlparse(url or '')
    host = (parts.hostname or '').lower().removeprefix('www.')
    segs = [s for s in parts.path.split('/') if s]
    if host == 'github.com' and len(segs) >= 2:
        return 'repo', segs[0] + '/' + re.sub(r'\.git$', '', segs[1])
    if host == 'arxiv.org' and len(segs) >= 2 and segs[0] in ('abs', 'pdf', 'html'):
        match = re.match(r'(\d{4}\.\d{4,5})', segs[1])
        if match:
            return 'paper', match.group(1)
    return None, None


CATALOG_KIND = {'image': 'image', 'github': 'repo', 'youtube': 'video', 'x': 'tweet', 'pdf': 'pdf', 'article': 'article',
                'mail': 'mail', 'text': 'text', 'file': 'file'}


def _reason(body):
    text = _section(body, 'Reason')
    return '' if text.startswith('_(no reason given)_') else text


def _snapshot_slug(vault, wanted, card_id):
    """A slug whose snapshot dir is free or already this card's (a second `process` rewrites it)."""
    import memrove_v3_sources as sources
    for number in range(1, 100):
        name = wanted if number == 1 else wanted + '-' + str(number)
        meta = sources.snap_dir(vault, name) / sources.SNAP_META
        if not meta.is_file():
            return name
        try:
            if json.loads(meta.read_text(encoding='utf-8')).get('card') == card_id:
                return name
        except ValueError:
            pass
    raise ValueError('too many snapshots named ' + wanted)


def process_card(vault, card, found, allow_audio=False, fetch=None, note=None, target=None):
    """One card -> snapshot under snap/<slug>/, catalog entry, memrove note, status processed. Returns the report entry."""
    import memrove_v3_research as research
    import memrove_v3_sources as sources
    fetch = fetch or sources.fetch
    meta, body = dict(card['meta']), card['body']
    card_id, kind = card['id'], meta.get('source_kind') or kind_of(meta.get('url'))
    url = scrub_url(meta['url']) if meta.get('url') else None
    private = meta.get('visibility') == 'private' or kind in PRIVATE_KINDS or never_fetch(url)
    reason = _reason(body)
    entry = {'id': card_id, 'source_kind': kind, 'title': meta.get('title'), 'url': url, 'reason': reason}
    route, ident = (None, None) if private else classify(url)
    if route:
        slug_name = snap_slug(ident)
        dest = sources.snap_dir(vault, slug_name)
        if not (dest / sources.SNAP_META).is_file():
            fetch(vault, route, ident, slug_name)
            snap_meta = json.loads((dest / sources.SNAP_META).read_text(encoding='utf-8'))
            snap_meta['card'] = card_id
            (dest / sources.SNAP_META).write_text(json.dumps(snap_meta, indent=1) + '\n', encoding='utf-8')
        method = 'sources-' + route
        title = meta.get('title') if meta.get('title') and not str(meta['title']).startswith('http') else ident
        catalog_id = research.item_id(url, ident)
    else:
        info = {}
        text, method, problem = extract(vault, card, found, allow_audio, info, url, online=not private)
        if text is None or (not text and kind != 'image'):
            raise ValueError(problem or 'no text could be extracted')
        title = info.get('title') or meta.get('title') or card_id
        raw = inbox(vault) / RAW / (card_id + '.md')
        raw.parent.mkdir(parents=True, exist_ok=True)
        head = ['---', 'type: grab-raw', 'card: ' + json.dumps(card_id), 'title: ' + json.dumps(title, ensure_ascii=False),
                'url: ' + json.dumps(url or ''), 'method: ' + json.dumps(method),
                'extracted: ' + json.dumps(now().isoformat(timespec='seconds')), 'visibility: private', '---', '']
        raw.write_text('\n'.join(head) + text.strip() + '\n', encoding='utf-8', newline='\n')
        slug_name = _snapshot_slug(vault, slug(title) if title != card_id else card_id, card_id)
        dest = sources.snap_dir(vault, slug_name)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        page = '# ' + title + '\n\n'
        if kind == 'image':
            image = next(Path(vault) / link for link in re.findall(r'\[\[([^\]]+)\]\]', _section(body, 'Files')) if link.lower().endswith('.png'))
            shutil.copy2(image, dest / 'image.png')
            page += '![](image.png)\n\n## OCR\n\n' + (text.strip() or '_(no text recognized)_') + '\n'
        else:
            page += text.strip() + '\n'
        (dest / 'page.md').write_text(page, encoding='utf-8', newline='\n')
        snap = {'kind': CATALOG_KIND.get(kind, 'article'), 'url': url, 'card': card_id, 'method': method}
        if private:
            snap['visibility'] = 'private'
        sources._finish(dest, snap)
        meta['raw'] = raw.relative_to(vault).as_posix()
        meta.update({target: info[source] for source, target in (('author', 'author'), ('published', 'published'), ('site', 'site'))
                     if info.get(source)})
        catalog_id = 'grab:' + card_id
    research.add_capture(vault, {'id': catalog_id, 'kind': 'paper' if route == 'paper' else CATALOG_KIND.get(kind, 'article'),
                                 'title': title, 'url': None if private else url, 'card': card_id, 'reason': reason,
                                 'snapshot': slug_name, 'visibility': 'private' if private else 'internal',
                                 'card_path': INBOX + '/' + card_id + '.md'})
    if note:  # a private card's title may be a mail subject: the note carries only the kind
        label = 'a private ' + kind if private else title
        text = 'owner captured ' + label + (' because ' + reason if reason else ' (no reason given)')
        note(text, 'grab card ' + card_id + ((' ' + url) if url and not private else ''))
    meta.update(status='processed', method=method, snapshot=slug_name, processed=now().isoformat(timespec='seconds'))
    meta.pop('error', None)
    _set_target(meta, target)
    if title and (not meta.get('title') or str(meta['title']).startswith('http')):
        meta['title'] = _first_line(title, 160)
        body = re.sub(r'^# .*$', lambda _m: '# ' + meta['title'], body, count=1, flags=re.M)
    if '## Processed' not in body:
        body = body.rstrip() + '\n\n## Processed\n\nsnapshot: `wiki/_sources/snap/' + slug_name + '/` catalog: `' + catalog_id + '`'
    write_card(card['path'], meta, body)
    entry.update(status='processed', method=method, snapshot='wiki/_sources/snap/' + slug_name, catalog=catalog_id)
    return entry


def process(vault, card_ids=None, allow_audio=False, retry=False, fetch=None, note=None, target=None, background=False):
    vault = Path(vault)
    found = tools()
    targets = [find_card(vault, cid) for cid in card_ids] if card_ids else \
        [c for c in cards(vault) if c['meta'].get('status') in ('pending', 'extracted') or (retry and c['meta'].get('status') == 'error')]
    def lead(card):
        return fold((group_list(card['meta'].get('groups')) or [card['id']])[0])
    first = {}
    for index, card in enumerate(targets):  # a group's cards run together (by first group), at the place its first card had
        first.setdefault(lead(card), index)
    targets = sorted(targets, key=lambda card: first[lead(card)])
    results = []
    for card in targets:
        try:
            results.append(process_card(vault, card, found, allow_audio, fetch, note, target))
        except Exception as exc:  # one broken source must not stop the queue
            meta = dict(card['meta'])
            meta.update(status='pending' if background else 'error', error=(type(exc).__name__ + ': ' + str(exc))[:200])
            write_card(card['path'], meta, card['body'])  # background: the card stays pending, the reason sits in `error`
            results.append({'id': card['id'], 'title': meta.get('title'), 'status': meta['status'], 'note': meta['error']})
    return {'status': 'ok', 'processed': len(results), 'cards': results,
            'tools': {name: bool(path) for name, path in found.items()}}


def spawn_process(vault, card_id):
    """Start `process` for one card, detached (own session, no stdio), through the same guarded path as the CLI. Never waits and
    never raises: a spawn failure leaves the card pending with a short reason."""
    if os.environ.get('MEMROVE_GRAB_NO_PROCESS'):  # tests and snapshots must not start real processing
        return False
    command = [sys.executable, str(Path(__file__).resolve()), '--vault', str(vault), 'process', '--background', card_id]
    options = {'creationflags': 0x00000008 | NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    try:
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        return True
    except Exception as exc:
        try:
            card = find_card(vault, card_id)
            meta = dict(card['meta'])
            meta.update(status='pending', error=('process did not start: ' + str(exc))[:200])
            write_card(card['path'], meta, card['body'])
        except Exception:
            pass
        return False


# ---------------------------------------------------------------- intention: used or hoarded

ARCHIVE = 'archive'
STALE_DAYS, RAW_DAYS = 14, 3  # processed-but-unused after 14 days, unprocessed after 3


def _age(meta, path):
    try:
        then = datetime.fromisoformat(meta['captured'])
    except (KeyError, ValueError, TypeError):
        then = datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone()
    return then, (now() - then).days


def _when(text, path, keys):
    """Date a file says it was written (front matter `updated` / `valid_from`), else its mtime."""
    for key in keys:
        match = re.search(r'^\s*"?' + key + r'"?\s*[:=]\s*"?(\d{4}-\d{2}-\d{2}(?:[T ][\d:+.Z-]+)?)', text[:2000], re.M)
        if match:
            try:
                value = datetime.fromisoformat(match.group(1).replace('Z', '+00:00'))
                return value if value.tzinfo else value.astimezone()
            except ValueError:
                pass
    return datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone()


def _uses(folder, keys, cite, ids, found):
    for path in Path(folder).rglob('*.md'):
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        hits = [cid for cid in ids if cite(cid, text)]
        if hits:
            when = _when(text, path, keys)
            for cid in hits:
                if cid not in found or when < found[cid]:
                    found[cid] = when


def _used_ids(vault, ids):
    """{card id: earliest time the evidence itself dates a use}. Wiki pages cite [grab:<id>] (page `updated`, else mtime); memrove
    records name the id (record `valid_from`, else mtime). process's own note ('grab card <id>') does not count."""
    found = {}
    tail = r'(?![a-z0-9]|-[a-z0-9])'
    _uses(Path(vault) / 'wiki' / '_concepts', ('updated',), lambda cid, t: '[grab:' + cid + ']' in t, ids, found)
    _uses(Path(vault) / 'notes' / 'observed', ('valid_from', 'recorded_at'),
          lambda cid, t: re.search(r'(?<!grab card )(?<![\w-])' + re.escape(cid) + tail, t), ids, found)
    return found


def _all_cards(vault):
    """Inbox cards plus archived ones: the metric must still see what was archived as unused."""
    found = [(c, False) for c in cards(vault)]
    folder = inbox(vault) / ARCHIVE
    if folder.is_dir():
        for path in sorted(folder.glob('*.md')):
            try:
                meta, body = read_card(path)
            except (OSError, UnicodeDecodeError):
                continue
            if meta.get('type') == 'grab':
                found.append(({'path': path, 'id': path.stem, 'meta': meta, 'body': body}, True))
    return found


def intention(vault):
    """Which captures serve an intention. Use dates come from the evidence, not from when this runs; only `shown` is stamped."""
    vault = Path(vault)
    entries = _all_cards(vault)
    used = _used_ids(vault, [c['id'] for c, _ in entries])
    stamp = now().isoformat(timespec='seconds')
    stale, raw, matured, within = [], [], 0, 0
    for card, archived in entries:
        meta = card['meta']
        then, days = _age(meta, card['path'])
        is_used = bool(meta.get('target')) or card['id'] in used
        dirty = False
        dates = [used[card['id']]] if card['id'] in used else []
        if meta.get('target') and meta.get('used_at'):  # time the target was set; the only stamp we keep as a date
            try:
                dates.append(datetime.fromisoformat(meta['used_at']))
            except ValueError:
                pass
        processed = meta.get('status') == 'processed'
        if processed and days >= STALE_DAYS:
            matured += 1
            within += bool(dates) and (min(dates) - then).days <= STALE_DAYS
        row = {'id': card['id'], 'title': meta.get('title'), 'days': days}
        if not is_used and not archived:
            if processed and days > STALE_DAYS:
                stale.append(row)
                if not meta.get('shown'):
                    meta['shown'], dirty = stamp, True
            elif meta.get('status') in ('pending', 'extracted') and days > RAW_DAYS:
                raw.append(row)
        if dirty:
            write_card(card['path'], meta, card['body'])
    ratio = round(within / matured, 3) if matured else None
    return {'status': 'ok', 'pending_processed': stale, 'pending_unprocessed': raw,
            'metric': {'processed': matured, 'used': within, 'rate': ratio, 'window_days': STALE_DAYS, 'kill_floor': 0.2}}


def archive(vault):
    """Unused processed cards older than 14 days that a status already showed once go to archive/. Snapshots and catalog entries stay."""
    import memrove_v3_research as research
    vault = Path(vault)
    entries = [c for c, archived in _all_cards(vault) if not archived]
    used = set(_used_ids(vault, [c['id'] for c in entries]))  # re-checked now: a citation since the showing wins
    moved = []
    for card in entries:
        meta = card['meta']
        if card['id'] in used or meta.get('target') or meta.get('status') != 'processed' or not meta.get('shown') \
                or _age(meta, card['path'])[1] <= STALE_DAYS:
            continue
        target = inbox(vault) / ARCHIVE / card['path'].name
        target.parent.mkdir(exist_ok=True)
        os.replace(card['path'], target)
        moved.append(card['id'])
    if moved:
        try:
            cat = json.loads(research._catalog_path(vault).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            cat = None
        if cat:
            for item in cat['items']:
                if item.get('card') in moved:
                    item['card_path'] = INBOX + '/' + ARCHIVE + '/' + item['card'] + '.md'
            research._write_catalog(vault, cat)
    return {'status': 'ok', 'archived': moved, 'folder': INBOX + '/' + ARCHIVE}


# ---------------------------------------------------------------- install / remove


def _tk_ok(python):
    try:
        return subprocess.run([python, '-c', 'import tkinter'], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install_defuddle(tools_dir=None):
    """One pinned install: package.json with the exact version, a lock file, `npm ci --ignore-scripts`. No npx -y at run time."""
    tools_dir = Path(tools_dir or TOOLS)
    npm = _which('npm')
    if not npm:
        return {'status': 'missing', 'note': 'npm not found (install Node)'}
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / 'package.json').write_text(json.dumps({'name': 'memrove-defuddle', 'private': True,
                                                        'dependencies': {'defuddle': DEFUDDLE_VERSION}}, indent=1) + '\n', encoding='utf-8')
    if (tools_dir / 'node_modules/.bin/defuddle').is_file():
        return {'status': 'installed', 'version': DEFUDDLE_VERSION}
    for command in (['install', '--package-lock-only', '--ignore-scripts'] if not (tools_dir / 'package-lock.json').is_file() else [],
                    ['ci', '--ignore-scripts']):
        if command:
            code, _, err = _run([npm] + command, timeout=600, cwd=str(tools_dir))
            if code:
                return {'status': 'error', 'note': err.strip()[-300:]}
    return {'status': 'installed', 'version': DEFUDDLE_VERSION}


def install(vault, state, hotkey=True, spec=None, defuddle=True):
    vault, state = Path(vault).resolve(), Path(state).resolve()
    spec = spec or saved_hotkey(state)
    if hotkey and sys.platform == 'darwin':  # validate before anything is written, so a bad key changes nothing
        keycode, modifiers, label = mac_hotkey(spec)
    inbox(vault).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    done = {'status': 'installed', 'folder': INBOX, 'hotkey': None}
    if defuddle:
        done['defuddle'] = install_defuddle()
        done['ocr'] = install_ocr()
    if hotkey and sys.platform == 'darwin':
        # The resident menu bar agent is the MemroveGrab binary (accessory app: no Dock icon). It owns the status item, the
        # hotkey and the panel; saving stays in this script, run headless. Script and panel source are copied from the one
        # directory this script lives in, so the state copy cannot lag behind the vault scripts; a failed build changes nothing.
        folder = state / 'grab'
        folder.mkdir(parents=True, exist_ok=True)
        runner, source, binary = folder / script.name, folder / PANEL_SOURCE, folder / 'MemroveGrab'
        shutil.copy2(script, runner)
        shutil.copy2(script.with_name(PANEL_SOURCE), source)
        swiftc = _which('swiftc')
        if not swiftc:
            raise RuntimeError('swiftc not found (install the Xcode command line tools): the menu bar agent needs it')
        temp = binary.with_name(binary.name + '.build')
        built = subprocess.run(['nice', swiftc, '-O', '-swift-version', '5', str(source), '-o', str(temp)], capture_output=True, text=True, timeout=600)
        if built.returncode != 0:
            raise RuntimeError((built.stderr or built.stdout or 'swiftc failed').strip()[-800:])
        os.replace(temp, binary)
        done['panel'] = str(binary)
        import plistlib
        plist = {'Label': LAUNCH_LABEL, 'RunAtLoad': True, 'KeepAlive': True,
                 'ProgramArguments': [str(binary), 'agent', '--vault', str(vault), '--python', sys.executable, '--script', str(runner),
                                      '--keycode', str(keycode), '--mods', str(modifiers), '--label', label],
                 'ProcessType': 'Interactive', 'StandardErrorPath': str(folder / 'listener.log'),
                 'EnvironmentVariables': {'PATH': os.environ.get('PATH', '/usr/bin:/bin:/usr/sbin:/sbin'),
                                          'LANG': 'en_US.UTF-8'}}
        agent = _launch_agent()
        previous = _agent_vault()
        if previous and previous != str(vault):
            done['onceki_vault'] = previous  # the hotkey moves to this vault
        agent.parent.mkdir(parents=True, exist_ok=True)
        domain = 'gui/' + str(os.getuid())
        subprocess.run(['launchctl', 'bootout', domain + '/' + LAUNCH_LABEL], capture_output=True)
        agent.write_bytes(plistlib.dumps(plist))
        subprocess.run(['launchctl', 'bootstrap', domain, str(agent)], check=True, capture_output=True)
        done['hotkey'] = label
        done['hotkey_running'] = _listener_ok()
    state.mkdir(parents=True, exist_ok=True)
    _state_path(state).write_text(json.dumps({'schema': 1, 'hotkey': done['hotkey'],
                                              'tus': spec if done['hotkey'] else saved_hotkey(state)}, ensure_ascii=False) + '\n',
                                  encoding='utf-8')
    done['tools'] = {name: bool(path) for name, path in tools().items()}
    return done


def uninstall(vault, state):
    state = Path(state).resolve()
    removed = []
    if sys.platform == 'darwin' and _agent_vault() == str(Path(vault).resolve()):
        # Only the listener this vault installed: one hotkey serves one vault at a time.
        subprocess.run(['launchctl', 'bootout', 'gui/' + str(os.getuid()) + '/' + LAUNCH_LABEL], capture_output=True)
        _launch_agent().unlink()
        removed.append('LaunchAgent')
    runner = state / 'grab' / Path(__file__).name
    if runner.exists():
        runner.unlink()
    if _state_path(state).exists():
        _state_path(state).unlink()
        removed.append(STATE_FILE)
    return {'status': 'removed', 'removed': removed, 'notes_kept': True}


def status(vault, state):
    installed = _state_path(state).is_file()
    running = None
    if sys.platform == 'darwin' and _agent_vault() == str(Path(vault).resolve()):
        probe = subprocess.run(['launchctl', 'print', 'gui/' + str(os.getuid()) + '/' + LAUNCH_LABEL], capture_output=True)
        running = probe.returncode == 0 and b'state = running' in probe.stdout
    return {'status': 'ok', 'installed': installed, 'listener_running': running, **intention(vault),
            'pending': pending(vault), 'folder': INBOX, 'tk': _tk_ok(sys.executable),
            'tools': {name: bool(path) for name, path in tools().items()}}


# ---------------------------------------------------------------- CLI

def default_state(vault):
    import hashlib
    base = Path.home() / 'Library/Application Support' if sys.platform == 'darwin' else \
        Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state')))
    return base / 'memrove-v3' / hashlib.sha256(str(vault).encode()).hexdigest()[:16]


def resolve_state(vault):
    config = Path(vault) / '.memrove-runtime.json'
    try:
        pinned = Path(json.loads(config.read_text(encoding='utf-8'))['state']).expanduser()
    except (OSError, ValueError, KeyError, TypeError):
        pinned = None
    return pinned if pinned and pinned.is_absolute() else default_state(Path(vault).resolve())


def _waiting(result):
    lines = []
    for title, key in (('Processed but unused (>14 days)', 'pending_processed'), ('Not processed (>3 days)', 'pending_unprocessed')):
        if result.get(key):
            lines.append('\n' + title + ':')
            lines += ['  %3d d  %s  %s' % (r['days'], r['id'], str(r['title'] or '')[:50]) for r in result[key]]
    m = result['metric']
    lines.append('\nUse rate (within 14 days, matured ' + str(m['processed']) + ' cards): ' + ('-' if m['rate'] is None else '%d%%' % round(m['rate'] * 100)))
    return '\n'.join(lines)


def human(result, command):
    if result.get('status') == 'captured':
        return 'Grabbed: ' + result['path']
    if result.get('status') == 'existing_card_added':
        return 'This source already existed; the new note was added to the card: ' + result['path']
    if command == 'process':
        lines = []
        for entry in result['cards']:
            lines.append('[' + {'processed': 'ok', 'error': 'ERROR'}.get(entry['status'], entry['status']) + '] ' +
                         str(entry.get('title') or entry['id'])[:70] + ((' -> ' + entry['snapshot']) if entry.get('snapshot') else '') +
                         (('\n    ' + entry['note']) if entry.get('note') else ''))
        return '\n'.join(lines) or 'No pending sources.'
    if command == 'open':
        return '\n'.join('%3d d  %s  %s%s' % (r['days'], r['id'], str(r['title'] or '')[:60], ('  [' + ', '.join(r['groups']) + ']') if r['groups'] else '')
                         for r in result['cards']) or 'No open loops.'
    if result.get('status') == 'closed':
        return 'Closed: ' + result['id']
    if command == 'list':
        return '\n'.join('%-10s %-8s %s%s' % (r['status'], r['source_kind'], str(r['title'])[:70], ('  [' + ', '.join(r['groups']) + ']') if r.get('groups') else '')
                         for r in result['cards']) or 'No captured sources.'
    if command == 'hotkey' and result.get('status') == 'hotkey':
        if not result.get('installed'):
            return 'Grab installed degil. Install for: memrove grab kur'
        return 'Hotkey: ' + str(result.get('hotkey') or 'none') + '\nTo change: memrove grab hotkey \'cmd+"\''
    if command in ('install', 'hotkey'):
        lines = ['Hotkey changed.' if command == 'hotkey' else 'Grab installed.']
        if result.get('hotkey'):
            lines.append('Hotkey: ' + result['hotkey'] + ' (works in every app).')
        if result.get('hotkey_running') is False:
            lines.append('WARNING: the hotkey listener did not start; this key may be registered by another app. Try another key.')
        if result.get('tk') is False:
            lines.append('WARNING: this python has no Tk; only a "why" box opens instead of the window (brew install python-tk@3.14).')
        return '\n'.join(lines)
    if command == 'remove':
        return 'Grab removed. Captured cards stay in place.'
    if command == 'archive':
        return str(len(result['archived'])) + ' cards archived: ' + ', '.join(result['archived']) if result['archived'] else 'No cards to archive.'
    if command == 'status':
        return ('Installed: ' + ('yes' if result['installed'] else 'no') +
                ('' if result['listener_running'] is None else '\nHotkey listener: ' + ('running' if result['listener_running'] else 'stopped')) +
                '\nTk: ' + ('yes' if result['tk'] else 'no') + '\nPending sources: ' + str(result['pending']) +
                '\nTools: ' + ', '.join(n + ('=yes' if ok else '=no') for n, ok in result['tools'].items()) + _waiting(result))
    if result.get('status') == 'processed':
        return 'Card marked as processed: ' + result['id']
    return json.dumps(result, ensure_ascii=False, indent=2)


def main(argv=None, vault=None, state=None, note=None, fetch=None):
    def shared(default):
        # Subcommands must not reset a --vault/--json given before them, hence SUPPRESS there.
        common = argparse.ArgumentParser(add_help=False)
        common.add_argument('--vault', type=Path, dest='vault_sub', default=default, help=argparse.SUPPRESS)
        common.add_argument('--json', action='store_true', dest='force_json',
                            default=False if default is None else default, help=argparse.SUPPRESS)
        return common
    parser = argparse.ArgumentParser(prog='memrove grab', description='Capture a source and turn it into wiki evidence', parents=[shared(None)])
    sub = parser.add_subparsers(dest='command')
    common = shared(argparse.SUPPRESS)

    def command_parser(name, **options):
        return sub.add_parser(name, parents=[common], **options)
    add = command_parser('add', help='capture a URL, text or file (no network)')
    add.add_argument('items', nargs='*', help='URL, file path or text')
    add.add_argument('--reason', default='')
    add.add_argument('--text')
    add.add_argument('--title')
    add.add_argument('--tool', default='cli')
    add.add_argument('--group', help='groups for the card, comma separated (an empty --reason then reuses the group set\'s fresh one)')
    add.add_argument('--open', action='store_true', help='flag the card as an open loop')
    add.add_argument('--thought', action='store_true', help='store the text verbatim as a thought card (source_kind thought)')
    add.add_argument('--clipboard-image', action='store_true', help='capture the image on the clipboard (a screenshot)')
    window = command_parser('window', help='open the capture window')
    window.add_argument('--context', help=argparse.SUPPRESS)
    run = command_parser('process', help='turn pending cards into snapshot + catalog entry + note (uses the network, guarded)')
    run.add_argument('ids', nargs='*')
    run.add_argument('--audio', action='store_true', help='YouTube without subtitles: download audio and transcribe locally')
    run.add_argument('--target', help='open project or question these cards serve')
    run.add_argument('--retry', action='store_true', help='retry cards in error too')
    run.add_argument('--background', action='store_true', help=argparse.SUPPRESS)  # popup-spawned: a failure leaves the card pending
    command_parser('open', help='open loops (cards flagged open, not closed), newest first')
    command_parser('panel', help='build the native capture panel now (macOS, needs swiftc)')
    close = command_parser('close', help='close an open loop: sets closed')
    close.add_argument('id')
    done = command_parser('done', help='mark a card processed by hand')
    done.add_argument('id')
    done.add_argument('--note', action='append', default=[], help='vault-relative note that was written')
    done.add_argument('--summary')
    done.add_argument('--target', help='open project or question this card serves')
    show = command_parser('list', help='captured sources')
    show.add_argument('--status', choices=STATUSES)
    show.add_argument('--group', help='only the cards that have this group')
    setup = command_parser('install', help='install the hotkey LaunchAgent and the pinned Defuddle')
    setup.add_argument('--no-hotkey', action='store_true', help='skip the hotkey')
    setup.add_argument('--no-defuddle', action='store_true', help='skip the Defuddle install')
    setup.add_argument('--keys', help='hotkey, e.g. ctrl+alt+b (default) or cmd+"')
    keys = command_parser('hotkey', help='show or change the hotkey')
    keys.add_argument('keys', nargs='?')
    command_parser('remove', help='remove the hotkey; cards stay')
    command_parser('status', help='install and tool status, unused captures, use metric')
    command_parser('archive', help='move unused cards older than 14 days (shown once in status) to archive/')
    args = parser.parse_args(argv)
    vault = vault or args.vault_sub
    if vault is None:
        parser.error('--vault is required')
    vault = Path(vault).expanduser().resolve()
    state = Path(state) if state else resolve_state(vault)
    command = args.command or 'window'
    as_json = args.force_json or not sys.stdout or not sys.stdout.isatty()
    if command == 'window':
        context = (mac_context(json.loads(args.context)) if sys.platform == 'darwin' else json.loads(args.context)) if getattr(args, 'context', None) else gather_context()
        result = popup(vault, context)
    elif command == 'add':
        url, files, texts = None, [], []
        for item in args.items:
            if re.match(r'^https?://', item) and not url:
                url = item
            elif Path(item).expanduser().is_file():
                files.append(item)
            else:
                texts.append(item)
        text = args.text or ('\n'.join(texts) if texts else None)
        image = clipboard_image() if args.clipboard_image else None
        if args.clipboard_image and not image:
            raise ValueError('The clipboard holds no image.')
        result = capture(vault, url=url, text=text, files=files, why=args.reason, title=args.title, tool=args.tool, image=image,
                         groups=args.group, open=args.open, thought=args.thought)
    elif command == 'process':
        result = process(vault, args.ids or None, allow_audio=args.audio, retry=args.retry, fetch=fetch, note=note, target=args.target,
                         background=args.background)
        states = {k.get('status') for k in result.get('cards', [])}
        if states & {'processed', 'error'}:  # nothing pending = no stamp
            try:  # the state copy of this script has no sibling modules
                __import__('memrove_v3_heartbeat').stamp(resolve_state(vault), 'grab_process', None if 'processed' in states else 'error')
            except ImportError:
                pass
    elif command == 'open':
        result = open_loops(vault)
    elif command == 'panel':
        try:
            result = {'status': 'ok', 'panel': str(build_panel(vault))}
        finally:
            panel_paths(vault)[1].with_name('MemroveGrab.lock').unlink(missing_ok=True)
    elif command == 'close':
        result = close_loop(vault, args.id)
    elif command == 'done':
        result = finish(vault, args.id, args.note, args.summary, args.target)
    elif command == 'list':
        result = listing(vault, args.status, args.group)
    elif command == 'install':
        result = install(vault, state, hotkey=not args.no_hotkey, spec=args.keys, defuddle=not args.no_defuddle)
    elif command == 'hotkey':
        if args.keys:
            result = install(vault, state, spec=args.keys, defuddle=False)
            result['status'] = 'hotkey_changed'
        else:
            installed = json.loads(_state_path(state).read_text(encoding='utf-8')) if _state_path(state).is_file() else {}
            result = {'status': 'hotkey', 'hotkey': installed.get('hotkey'), 'tus': installed.get('tus') or DEFAULT_HOTKEY,
                      'installed': bool(installed)}
    elif command == 'remove':
        result = uninstall(vault, state)
    elif command == 'archive':
        result = archive(vault)
    else:
        result = status(vault, state)
    if sys.stdout is None:
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2) if as_json else human(result, command))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ValueError as exc:
        if sys.stderr:
            print(str(exc), file=sys.stderr)
        raise SystemExit(1)
