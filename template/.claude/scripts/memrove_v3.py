"""SQLite store of typed records. Each record points at a Markdown file in the vault; no model or network calls."""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata

JSON_ARGS = dict(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
VISIBILITY = ("public", "internal", "private")
KEEP_REVISIONS = 5  # full snapshots kept per record in `events`; older ones shrink to a stub
HIDDEN_STATUS = ("untrusted", "dropped_llm", "deleted_by_user", "source_missing")


class RevisionConflict(ValueError):  # a record id already exists with other content
    pass


def _kinds():
    try:
        import memrove_v3_kinds as kinds
    except ImportError:  # loaded by path: the scripts directory is not importable yet
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import memrove_v3_kinds as kinds
    return kinds


def _json(value) -> str:
    """Canonical JSON: the same payload always serialises to the same bytes."""
    return json.dumps(value, **JSON_ARGS)


def _need(ok, message):
    if not ok:
        raise ValueError(message)


# Token stems. Turkish glues suffixes onto roots, so "notlar" must meet "not". Each rule is
# "suffix:minimum stem:class of the letter before it"; rules are tried in order and a word
# is peeled until no rule fits. Text is already ASCII-folded when it gets here.
_RULES = ("imiz umuz iniz unuz:4:c nden ndan:5:v ten tan:4:s den dan:4:z nin nun nde nda:5:v "
          "miz muz niz nuz:4:v yla yle:4:v ler lar:3:a te ta:4:s de da:4:z si su ya ye yi yu:4:v "
          "in un im um le la:4:c i u e a:4:c")
_VOWELS = set("aeiou")


def _rule_table():
    table = []
    for chunk in re.findall(r"((?:[a-z]+ )*[a-z]+):(\d):([a-z])", _RULES):
        for suffix in chunk[0].split():
            table.append((suffix, int(chunk[1]), chunk[2]))
    return table


def _fits(prev, cls):
    return {"v": prev in _VOWELS, "c": prev not in _VOWELS, "s": prev in "cfhkpst",
            "z": prev not in "cfhkpst", "a": True}[cls]


def _harmony(stem, suffix):
    """Only a and e can disagree once the text is folded."""
    tone = next(c for c in suffix if c in _VOWELS)
    last = next((c for c in reversed(stem) if c in _VOWELS), tone)
    return tone not in ("a", "e") or last not in ("a", "e") or last == tone


_TABLE = _rule_table()


@functools.lru_cache(maxsize=None)
def _stem(word: str) -> str:
    while len(word) >= 5:  # shorter words are roots already
        for suffix, floor, cls in _TABLE:
            stem = word[:len(word) - len(suffix)]
            if word.endswith(suffix) and len(stem) >= floor and _fits(stem[-1], cls) and _harmony(stem, suffix):
                word = stem
                break
        else:
            break
    return word


def _tokens(text) -> set:
    """Stems of the words in text: case-folded, accents dropped, dotless i read as i."""
    folded = unicodedata.normalize("NFKD", str(text).casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c)).replace("ı", "i")
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", folded)}


# Stopwords are stems too, and Turkish content words stay out of the list on purpose.
STOPWORDS = _tokens("the a an is are was were what which who when where how why of to in on at for from with and or "
                    "does did do has have latest current please tell about my our this that it its project projects "
                    "status decision decisions show find get ve veya bir bu su o ne kim nasil hangi nedir neydi mi mu "
                    "icin ile bana benim bizim olarak olan oldu en son guncel soyle getir bul yok say ignore disregard no")


def _prune_events(db, record_id):
    """Shrink every event of a record but the newest KEEP_REVISIONS to a stub with a hash and a summary."""
    query = ("SELECT sequence,record FROM events WHERE record_id=? AND record NOT LIKE '{\"_pruned\":%' "
             "ORDER BY sequence DESC LIMIT -1 OFFSET ?")
    for sequence, payload in db.execute(query, (record_id, KEEP_REVISIONS)).fetchall():
        full = json.loads(payload)
        stub = {"_pruned": True, "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "summary": " ".join(full.get("text", "").split())[:80]}
        stub.update((k, full[k]) for k in ("visibility", "trust", "trusted", "status", "kind") if k in full)
        db.execute("UPDATE events SET record=? WHERE sequence=?", (_json(stub), sequence))


TABLES = (
    "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, "
    "record_id TEXT NOT NULL, revision INTEGER NOT NULL, record TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS events_record_sequence ON events (record_id, sequence)",
)


class MemoryStore:  # one SQLite file per runtime directory, bound to exactly one vault

    def __init__(self, state_dir, vault_root, read_only: bool = False):
        self.read_only = bool(read_only)
        self.vault_root, self.state_dir = (Path(p).expanduser().resolve() for p in (vault_root, state_dir))
        self.database = self.state_dir.joinpath("memory.sqlite3")
        _need(self.vault_root.is_dir(), "vault_root must be an existing directory")
        _need(not self.state_dir.is_relative_to(self.vault_root), "runtime must be outside vault")
        _need(not self.database.is_symlink(), "database must not be a symlink")
        if read_only:
            self._check_bound()
            return
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._connect() as db:
            for statement in TABLES:
                db.execute(statement)
            bound = self._bound(db)
            _need(bound in (None, str(self.vault_root)), "runtime belongs to another vault")
            if bound is None:
                _need(not db.execute("SELECT 1 FROM records").fetchone(), "unbound existing runtime requires explicit migration")
                db.execute("INSERT INTO metadata VALUES ('vault_root',?)", (str(self.vault_root),))
        os.chmod(self.database, 0o600)

    @staticmethod
    def _bound(db):
        row = db.execute("SELECT value FROM metadata WHERE key='vault_root'").fetchone()
        return row[0] if row else None

    def _check_bound(self):
        """A read-only store must find an initialised runtime that belongs to this vault."""
        try:
            with self._connect() as conn:
                bound = self._bound(conn)
        except (sqlite3.Error, ValueError) as exc:
            raise ValueError("read-only runtime is not initialized") from exc  # missing or foreign database
        _need(bound is not None, "read-only runtime is not bound to a vault")
        _need(bound == str(self.vault_root), "runtime belongs to another vault")

    @contextmanager
    def _connect(self):
        """One short-lived connection per operation; read-only stores open the file with mode=ro."""
        if self.read_only:
            _need(self.database.is_file(), "read-only runtime is not initialized")
            db = sqlite3.connect(f"{self.database.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
            db.execute("PRAGMA query_only = ON")
        else:
            db = sqlite3.connect(str(self.database), timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _source(self, value) -> str:
        """Vault-relative POSIX path of an existing regular file inside the vault."""
        relative = Path(value) if isinstance(value, str) and value else None
        _need(relative is not None and not relative.is_absolute() and ".." not in relative.parts,
              "source must be an existing vault-relative file")
        resolved = (self.vault_root / relative).resolve()
        _need(resolved.is_relative_to(self.vault_root) and resolved.is_file(), "source missing or outside vault")
        return relative.as_posix()

    def _validate(self, record) -> dict:
        """Normalised copy of a record, with the source hash taken from the file as it is now."""
        _need(isinstance(record, dict), "record must be an object")
        rec = json.loads(_json(record))  # deep copy; also rejects values JSON cannot hold
        _need(isinstance(rec.get("id"), str) and rec["id"].strip(), "record id required")
        _need(isinstance(rec.get("text"), str), "record text required")
        rec["source"] = self._source(rec.get("source"))
        rec["source_sha256"] = hashlib.sha256((self.vault_root / rec["source"]).read_bytes()).hexdigest()
        if not rec.get("updated_at"):
            # Other tools date notes as updated, modified, ...; take the first one that starts with an ISO date.
            dated = (rec.get(a) for a in ("updated", "modified", "last_modified", "date_modified"))
            rec["updated_at"] = next((v.strip() for v in dated if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v.strip())),
                                     rec.get("updated_at"))
            if rec["updated_at"] is None:
                del rec["updated_at"]
        for field in ("project", "kind", "status", "updated_at"):  # text fields
            _need(field not in rec or isinstance(rec[field], str), f"{field} must be a string")
        _kinds().validate_record(rec)
        _need(isinstance(rec.setdefault("facts", {}), dict), "facts must be an object")
        _need(type(rec.setdefault("revision", 1)) is int and rec["revision"] >= 1, "positive revision required")
        _need(rec.setdefault("visibility", "internal") in VISIBILITY, "invalid visibility")
        older = rec.get("supersedes", [])
        older = [older] if isinstance(older, str) else older
        _need(isinstance(older, list) and all(isinstance(v, str) for v in older), "supersedes must contain record ids")
        _need(rec["id"] not in older, "record cannot supersede itself")
        rec["supersedes"] = older
        return rec

    def ingest(self, record) -> dict:
        """Store a new record. The same record again is a no-op; the same id with other content conflicts."""
        _need(not self.read_only, "read-only runtime does not permit writes")
        rec = self._validate(record)
        payload = _json(rec)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")  # one writer at a time
            stored = db.execute("SELECT payload FROM records WHERE id=?", (rec["id"],)).fetchone()
            if stored and stored[0] != payload:
                raise RevisionConflict("record id already exists with other content")
            if not stored:
                db.execute("INSERT INTO records VALUES (?,?)", (rec["id"], payload))
                db.execute("INSERT INTO events (event_type, record_id, revision, record) VALUES ('ingest',?,?,?)",
                           (rec["id"], rec["revision"], payload))
                _prune_events(db, rec["id"])
        return rec
