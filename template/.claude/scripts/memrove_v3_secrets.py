"""Secret redaction for text the user writes through memrove commands. On by default; the preferences switch it off."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

MASK = "[REDACTED]"
_NAME = r"api[_ -]?key|access[_ -]?token|token|secret|password|passwd"
_WORD, _END = r"(?<![A-Za-z0-9_])", r"(?![A-Za-z0-9_])"  # a token that stands alone, not inside a longer identifier


def _compiled(*sources):
    return tuple(re.compile(s) for s in sources)


PATTERNS = _compiled(
    # name=value in a query string, and name: value / name = value in text, JSON or escaped JSON
    rf"(?i)(?<=[?&])(?:{_NAME})=[^&\s]{{8,}}",
    rf"""(?i)(?<![a-zA-Z0-9_])(?:\\{{0,8}}["']?(?:{_NAME})\\{{0,8}}["']?)\s*[:=]\s*(?:\\{{0,8}}["']\S{{8,}}\\{{0,8}}["']|[^"'\s,;][^\s,;]{{7,}})""",
    r"(?<=://)[^/\s?#:]*:[^/\s?#]+(?=@[^/\s?#@]+)",  # user:password inside a URL
    # tokens with a fixed prefix
    r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b",
    r"\bsk-[A-Za-z0-9_-]{20,255}\b",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",  # a block cut off before its end line
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}",
    r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,255}\b",  # Stripe uses underscores, so the dashed sk- rule misses it
    r"\bxox[abeoprs]-[0-9]+-[A-Za-z0-9-]{8,255}\b",
    r"\bxapp-[0-9]+-[A-Za-z0-9-]{8,255}\b",
    r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])",
    r"\bnpm_[A-Za-z0-9]{36}\b",
    r"(?<![A-Za-z0-9_.-])SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-]|\.[A-Za-z0-9_-])",
    # JWT: three dotted base64url parts starting with the encoded `{"` header (five for an encrypted one)
    r"(?<![A-Za-z0-9_.-])eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]{10,}){2}(?:\.[A-Za-z0-9_-]+){0,2}(?![A-Za-z0-9_-])",
    # Discord bot token: encoded user id (starts with M, N or O), 6 character time stamp, 27+ character HMAC
    r"(?<![A-Za-z0-9_.-])[MNO][A-Za-z0-9_-]{22,27}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,40}(?![A-Za-z0-9_-]|\.[A-Za-z0-9_-])",
    # more prefixed providers; account ids that are no secret (card numbers, Twilio sids) stay out on purpose
    *(_WORD + body + _END for body in (
        r"ya29\.[0-9A-Za-z_-]{20,}", r"xai-[A-Za-z0-9]{40,}", r"gsk_[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{30,}",
        r"r8_[A-Za-z0-9]{30,}", r"pplx-[A-Za-z0-9]{40,}", r"dapi[A-Za-z0-9]{32}", r"dop_v1_[a-f0-9]{64}",
        r"glpat-[A-Za-z0-9_-]{20,}", r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{20,}", r"sq0[a-z]{3}-[A-Za-z0-9_-]{22,}",
        r"access_token\$production\$[a-z0-9]{16}\$[a-f0-9]{32}", r"shpat_[a-fA-F0-9]{32}", r"key-[A-Za-z0-9]{32}",
        r"[0-9]{8,10}:[A-Za-z0-9_-]{35}")),
    r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]{8,}/B[A-Za-z0-9_]{8,}/[A-Za-z0-9_]{20,}",
    # AWS secret key: 40 base64 characters, recognisable only by the words before it
    r"""(?i)aws[^\n]{0,20}?access[\s_-]?key[\s_-]?[:=][\s"']*[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])""",
)


def _custom_literals(state):
    """Extra exact strings to mask, one per line in <state>/secret-patterns.txt (literals, never regular expressions)."""
    path = Path(state) / "secret-patterns.txt"
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("secret-patterns.txt must be a regular file no larger than 64 KiB")
    lines = [(n, raw.strip()) for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)]
    literals = [(n, v) for n, v in lines if v and not v.startswith("#")]
    for number, value in literals:
        if not 4 <= len(value) <= 1000 or value == MASK:
            raise ValueError(f"invalid secret literal at line {number}")
    if len(literals) > 100:
        raise ValueError("secret-patterns.txt supports at most 100 literals")
    return [v for _, v in literals]


def redact(text, state):
    """-> (masked text, number of masked spans)."""
    if not isinstance(text, str):
        raise ValueError("secret filter input must be text")
    count = 0
    for pattern in PATTERNS:
        text, hits = pattern.subn(MASK, text)
        count += hits
    for literal in _custom_literals(state):
        count += text.count(literal)
        text = text.replace(literal, MASK)
    return text, count


def record(state, count):
    """Add to the redaction totals kept outside the vault. Only counts are stored, never what matched."""
    if type(count) is not int or count < 0:
        raise ValueError("redaction count must be a nonnegative integer")
    Path(state).mkdir(parents=True, exist_ok=True, mode=0o700)
    path = Path(state) / "secret-filter.sqlite3"
    if path.is_symlink():
        raise ValueError("secret filter health database must not be a symlink")
    with closing(sqlite3.connect(path, timeout=10)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS health (id INTEGER PRIMARY KEY CHECK(id=1), total INTEGER NOT NULL, last_count INTEGER NOT NULL, last_at TEXT)")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT total FROM health WHERE id=1").fetchone()
        total = (row[0] if row else 0) + count
        db.execute("INSERT OR REPLACE INTO health VALUES (1,?,?,?)", (total, count, datetime.now(timezone.utc).isoformat()))
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return {"total": total, "last_count": count}


def health(state):
    path = Path(state) / "secret-filter.sqlite3"
    empty = {"total": 0, "last_count": 0, "last_at": None}
    if not path.exists():
        return empty
    if path.is_symlink() or not path.is_file():
        raise ValueError("secret filter health database must be a regular file")
    with closing(sqlite3.connect(path, timeout=2)) as db:
        row = db.execute("SELECT total,last_count,last_at FROM health WHERE id=1").fetchone()
    return dict(zip(("total", "last_count", "last_at"), row)) if row else empty
