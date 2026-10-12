"""Markdown vault <-> store reconciliation. Runs in process; no services, no model calls."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from memrove_v3 import MemoryStore, RevisionConflict, VISIBILITY, _json
from memrove_v3_links import Links
from memrove_v3_projections import OWNED, project_record, split_user_note
from memrove_v3_preferences import read as read_preferences
import memrove_v3_trust as trust
from memrove_v3_secrets import redact as redact_secrets, record as record_redactions

SKIP_NAMES = {"agents.md", "claude.md", "gemini.md", "skill.md", "hooks.md", "config.md", "settings.md",
              "instructions.md", "codex.md", "setup.md", "install.md"}
SKIP_DIRS = {"node_modules", "receipts", "__pycache__"}
GATE_KEYS = ("visibility", "trust", "trusted", "remote_allowed")  # narrowing keys must sit at the top level
UNSUPPORTED = "unsupported YAML metadata; use JSON frontmatter"
STATUSES = ("inbox", "active", "waiting", "blocked", "done", "cancelled")


def _hash(data) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def atomic(path, text) -> None:
    """Write through a temp file in the same directory, then rename over the target."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".memrove-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


# --- front matter -------------------------------------------------------------------------------

_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\']|\'\')*\'')
_PLACEHOLDER = re.compile(r"\{\{[^{}\n]*\}\}")
_KEY = r"([^\W\d][\w-]*):[ \t]*(.*)"


def _quoted(text):
    return json.loads(text) if text[0] == '"' else text[1:-1].replace("''", "'")


def _flow(text):
    """`[a, "b"]`: items are never coerced to numbers or booleans."""
    if not text.endswith("]"):
        raise ValueError(UNSUPPORTED)
    rest, items = text[1:-1].strip(), []
    while rest:
        found = _QUOTED.match(rest) or re.match(r"[^,]+", rest)
        item = found[0].strip()
        if item[0] not in "\"'" and set(item) & set("[]{}"):
            raise ValueError(UNSUPPORTED)
        items.append(_quoted(item) if item[0] in "\"'" else item)
        rest = rest[found.end():].strip()
        if rest:
            if not rest.startswith(",") or not rest[1:].strip():
                raise ValueError(UNSUPPORTED)
            rest = rest[1:].strip()
    return items


def _value(text):
    text = text.strip()
    if not text:
        return None
    if _PLACEHOLDER.fullmatch(text):  # an Obsidian Templater tag stays literal until it is expanded
        return text
    if text[0] in "|>&*!{":
        raise ValueError(UNSUPPORTED)
    if text[0] == "[":
        return _flow(text)
    if text[0] in "\"'":
        found = _QUOTED.match(text)
        if not found or not re.fullmatch(r"(\s+#.*)?", text[found.end():]):
            raise ValueError(UNSUPPORTED)
        return _quoted(found[0])
    plain = re.split(r"(?:^|\s)#", text, maxsplit=1)[0].rstrip()
    if not plain:
        raise ValueError(UNSUPPORTED)
    try:
        return json.loads(plain)  # numbers, true/false/null
    except ValueError:
        return plain


def _block(lines):
    """Indented lines under an empty `key:`: one list (`- x`) or one flat mapping (`k: v`)."""
    indent = re.match(r" +", lines[0])
    if not indent or any(not line.startswith(indent[0]) or line[len(indent[0]):][:1] in (" ", "\t") for line in lines):
        raise ValueError(UNSUPPORTED)
    bodies = [line[len(indent[0]):] for line in lines]
    if all(b.startswith("- ") for b in bodies):
        return [_quoted(b[2:].strip()) if b[2:].lstrip()[:1] in ("\"", "'") else b[2:].strip() for b in bodies]
    mapping = {}
    for body in bodies:
        found = re.fullmatch(_KEY, body)
        if not found or found[1] in mapping:
            raise ValueError(UNSUPPORTED)
        value = _value(found[2])
        if found[1] in GATE_KEYS or (found[1] in ("status", "kind") and value == "untrusted"):
            raise ValueError(f"nested {found[1]} is not applied; move it to top-level frontmatter")
        mapping[found[1]] = value
    return mapping


def parse(text) -> tuple:
    """-> (metadata, body). JSON front matter, or flat YAML: scalars, inline lists, block lists, one-level mappings."""
    parts = text.splitlines(keepends=True)
    if not parts or parts[0].strip() != "---":
        return {}, text  # no front matter
    end = next((i for i in range(1, len(parts)) if parts[i].strip() == "---"), None)
    if end is None:
        raise ValueError("unterminated frontmatter")
    header, body = "".join(parts[1:end]).strip(), "".join(parts[end + 1:])
    if header.startswith("{"):
        meta = json.loads(header)
        if not isinstance(meta, dict):
            raise ValueError("frontmatter must be an object")
        return meta, body
    meta, rows, i = {}, header.splitlines(), 0
    while i < len(rows):
        row, i = rows[i], i + 1
        if not row.strip() or row.lstrip().startswith("#"):
            continue
        found = re.fullmatch(_KEY, row)
        if not found or found[1] in meta:
            raise ValueError(UNSUPPORTED)
        key, raw = found.groups()
        if raw.strip():
            meta[key] = _value(raw)
            continue
        nested = []
        while i < len(rows) and rows[i].startswith((" ", "\t")):
            if rows[i].startswith("\t"):
                raise ValueError(UNSUPPORTED)
            nested.append(rows[i])
            i += 1
        meta[key] = _block(nested) if nested else None
    return meta, body


def render(metadata, body) -> str:
    return "---\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n---\n" + body


# --- engine -------------------------------------------------------------------------------------

class SyncEngine:  # reconciles Markdown files with the store; owns the write-ahead journal
    def __init__(self, vault_root, state_dir):
        store = self.store = MemoryStore(state_dir, vault_root)
        self.root, self.state = store.vault_root, store.state_dir
        self.journal = self.state.joinpath("markdown-journal")
        os.makedirs(self.journal, mode=0o700, exist_ok=True)
        with store._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS markdown_sources(id TEXT PRIMARY KEY, source TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS scan_state(source TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL)")
        self.links = Links(self.store)

    def projection_helpers(self) -> tuple:
        return (_hash, atomic, render)

    def _path(self, relative, existing: bool = False):
        """Absolute path of a vault-relative source; refuses escapes and symlinks."""
        parts = Path(relative).parts if isinstance(relative, str) else None
        if not parts or Path(relative).is_absolute() or ".." in parts:
            raise ValueError("relative source required")
        path = self.root.joinpath(relative)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("source outside vault")
        if any(self.root.joinpath(*parts[:n]).is_symlink() for n in range(1, len(parts) + 1)):
            raise ValueError("symlink source rejected")
        if existing and not path.is_file():  # the caller needs to read it
            raise ValueError("source missing")
        return path

    # secret filter -------------------------------------------------------------------------------

    def _clean(self, value, hits):
        """Redact secrets in every string of a nested value; `hits` collects the counts."""
        if isinstance(value, str):
            text, count = redact_secrets(value, self.state)
            hits.append(count)
            return text
        if isinstance(value, dict):
            return {k: self._clean(v, hits) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clean(v, hits) for v in value]
        return value

    def _protect(self, text, metadata):
        """-> (text, metadata, redaction count); metadata is cleaned only where free text can sit."""
        if not read_preferences(self.root)["secret_filter"]:
            return text, metadata, 0
        hits = []
        free = {k: self._clean(metadata[k], hits) for k in ("title", "next_action", "facts") if k in metadata}
        return self._clean(text, hits), dict(metadata, **free), sum(hits)

    # scanning ------------------------------------------------------------------------------------

    @staticmethod
    def _overrides(prev, meta, body):
        """From a hand-edited memrove-owned note only verdict, pin and appended text are taken."""
        changes = {}
        verdict = meta.get("verdict")
        if verdict in ("keep", "reject", "pin"):
            changes.update(verdict=verdict, status="rejected" if verdict == "reject" else "active", pinned=verdict == "pin")
        if meta.get("pin") is True:
            changes["pinned"] = True
        note = split_user_note(body, prev["text"])
        if note:
            changes["user_note"] = note
        new = {k: v for k, v in prev.items() if note or k != "user_note"} | changes
        return dict(new, revision=prev["revision"] + 1) if new != prev else prev

    def _read(self, relative, db):
        """Parse one source into a record; an owned note resolves to its stored record. None = not indexed."""
        raw = self._path(relative, existing=True).read_bytes()
        meta, body = parse(raw.decode("utf-8"))
        if meta.get("kind") == "task" and body.lstrip().startswith("---"):
            raise ValueError("task has embedded frontmatter; reconcile metadata and body explicitly")
        if meta.get("kind") == "receipt" or meta.get("generated") is True:
            return None
        if relative.startswith(OWNED):
            row = db.execute("SELECT payload FROM records WHERE id=? OR json_extract(payload,'$.source')=?",
                             (str(meta.get("id")), relative)).fetchone()
            if row:
                prev = json.loads(row[0])
                if prev.get("status") == "deleted_by_user":
                    return None
                if prev["source_sha256"] != _hash(raw):
                    prev = self._overrides(prev, meta, body)
                    self._overridden.add(prev["id"])
                return prev
        record = self.store._validate(dict(meta, source=relative, text=body,
                                           id=meta.get("id") or "md-" + _hash(relative)[:24],
                                           kind=meta.get("kind") or "note"))
        if record["source_sha256"] != _hash(raw):
            raise ValueError("source changed while scanning")
        return record

    def _scan(self, known, db):
        """known=None reads everything; otherwise only files whose (mtime_ns, size) differ from it."""
        records, warnings, dupes = {}, [], set()
        self._overridden, self._present, self._reparsed, stamps = set(), set(), set(), {}
        for here, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d.casefold() not in SKIP_DIRS
                             and not (Path(here) / d).is_symlink())
            for name in sorted(files):  # stable order, so duplicate reports are stable
                low = name.casefold()
                if name.startswith(".") or not low.endswith(".md") or low in SKIP_NAMES or low.startswith("setup-"):
                    continue
                relative = (Path(here) / name).relative_to(self.root).as_posix()
                self._present.add(relative)
                try:
                    info = (Path(here) / name).stat()
                    stamp = (info.st_mtime_ns, info.st_size)  # taken before the read: a racing edit is re-read next time
                    if known is not None and known.get(relative) == stamp:
                        continue
                    self._reparsed.add(relative)
                    record = self._read(relative, db)
                except (ValueError, OSError, UnicodeError) as exc:  # one bad file must not stop the scan
                    warnings.append({"source": relative, "reason": str(exc)})
                    continue
                if record is None:
                    stamps[relative] = stamp + (None,)
                    continue
                if record["id"] in records:
                    dupes.add(record["id"])
                records[record["id"]] = record
                stamps[relative] = stamp + (record["id"],)
        for rid in dupes:
            records.pop(rid)
        # duplicates stay unstamped, so every scan reports them again
        self._stamps = {k: v for k, v in stamps.items() if v[2] not in dupes}
        return records, warnings, [{"id": i, "reason": "duplicate source id; all copies quarantined"} for i in sorted(dupes)]

    def _recover(self):
        """Finish source writes a crash interrupted. -> (journal files done, conflicts)."""
        done, conflicts = [], []
        for entry in sorted(self.journal.glob("*.json")):
            try:
                intent = json.loads(entry.read_text(encoding="utf-8"))
                path = self._path(intent["source"])
                actual = _hash(path.read_bytes()) if path.exists() else None  # None: file not there yet
                if actual == intent["old_hash"]:
                    atomic(path, intent["content"])
                elif actual != _hash(intent["content"]):
                    conflicts.append({"source": intent["source"], "reason": "recovery conflict; manual edit preserved"})
                    continue
                done.append(entry)
            except (ValueError, OSError, KeyError) as exc:  # unreadable journal entry
                conflicts.append({"journal": entry.name, "reason": str(exc)})
        return done, conflicts

    @staticmethod
    def _tombstone(db, rid, payload):
        """A user-deleted owned note stays as a deleted_by_user record, so it never comes back."""
        record = json.loads(payload)
        record.update(status="deleted_by_user", valid_to=datetime.now(timezone.utc).date().isoformat(),
                      revision=record["revision"] + 1)
        db.execute("UPDATE records SET payload=? WHERE id=?", (_json(record), rid))
        db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES ('deleted_by_user',?,?,?)",
                   (rid, record["revision"], _json(record)))

    def _mark_source_missing(self, db, gone):
        """A record whose every derived_from source is gone leaves recall but stays for the audit trail."""
        from memrove_v3_projections import update_record
        dead = ("deleted_by_user", "source_missing")

        def alive(source):
            if source.startswith("transcript:"):
                return os.path.isfile(source[11:])
            row = db.execute("SELECT payload FROM records WHERE id=?", (source,)).fetchone()
            return bool(row) and json.loads(row[0]).get("status") not in dead

        ids = [r[0] for r in db.execute("SELECT DISTINCT records.id FROM records, json_each(records.payload,'$.derived_from') j "
                                        f"WHERE j.value IN ({','.join('?' * len(gone))})", tuple(gone))]
        for rid in ids:
            rec = json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])
            if rec.get("status") not in dead and not any(alive(s) for s in rec["derived_from"]):
                update_record(self, rid, {"status": "source_missing", "status_before": rec.get("status")}, db)

    def sync(self, full=True):
        """Bring the store in line with the Markdown files. full=False (hook path) reads only changed files."""
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            done, conflicts = self._recover()
            known = None if full else {r[0]: (r[1], r[2]) for r in db.execute("SELECT source,mtime_ns,size FROM scan_state")}
            records, warnings, dupes = self._scan(known, db)
            conflicts += dupes
            owned = dict(db.execute("SELECT id,source FROM markdown_sources"))
            if full:
                gone = set(owned) - records.keys()
            else:  # gone: file vanished, or was re-read and no longer yields the record
                gone = {i for i, src in owned.items()
                        if src not in self._present or (src in self._reparsed and i not in records)}
            deleted = 0
            for rid in gone:
                row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
                if row and owned[rid].startswith(OWNED):
                    if json.loads(row[0]).get("status") != "deleted_by_user":
                        self._tombstone(db, rid, row[0])
                    continue
                if row:
                    db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES ('delete',?,?,?)",
                               (rid, json.loads(row[0])["revision"], row[0]))
                    db.execute("DELETE FROM records WHERE id=?", (rid,))
                    deleted += 1
                db.execute("DELETE FROM markdown_sources WHERE id=?", (rid,))
            for rid, record in records.items():
                row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
                before = json.loads(row[0]) if row else None
                if before and rid in owned and rid not in self._overridden:
                    changed = before["source_sha256"] != record["source_sha256"]
                    record["revision"] = max(record["revision"], before["revision"] + changed)
                if rid in self._overridden:
                    record = project_record(self, record)
                payload = _json(record)  # canonical form, comparable with the stored one
                if row and rid not in owned and row[0] != payload:
                    conflicts.append({"id": rid, "reason": "id already owned by another record"})
                    continue
                if before and rid in self._overridden and before["revision"] == record["revision"]:
                    db.execute("UPDATE records SET payload=? WHERE id=?", (payload, rid))  # new file hash only
                elif not row or row[0] != payload:
                    db.execute("INSERT OR REPLACE INTO records VALUES (?,?)", (rid, payload))
                    db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES (?,?,?,?)",
                               ("update" if row else "ingest", rid, record["revision"], payload))
                db.execute("INSERT OR REPLACE INTO markdown_sources VALUES (?,?)", (rid, record["source"]))
            if gone:
                self._mark_source_missing(db, gone)
            self.links.forget(gone, db)
            names = self.links.name_map(db)
            for record in records.values():
                self.links.derive(record, db, names)
            self.links.derive_missing(db)
            if known is None:
                db.execute("DELETE FROM scan_state")
            else:
                db.executemany("DELETE FROM scan_state WHERE source=?", [(s,) for s in set(known) - self._present])
            db.executemany("INSERT OR REPLACE INTO scan_state VALUES (?,?,?)", [(k, v[0], v[1]) for k, v in self._stamps.items()])
        for entry in done:
            entry.unlink(missing_ok=True)  # the intended write is on disk and indexed
        return {"status": "conflict" if conflicts else "degraded" if warnings else "succeeded", "indexed": len(records),
                "mode": "full" if full else "incremental", "deleted": deleted, "warnings": warnings, "conflicts": conflicts}

    # creating sources ----------------------------------------------------------------------------

    def _create(self, source, metadata, text, kind):
        """Write a new source through the journal, sync, and read the record back. -> (record, sync result, redactions)."""
        text, metadata, redacted = self._protect(text, metadata)
        path, content = self._path(source), render(metadata, text + "\n")
        entry = self.journal / (_hash(source) + ".json")
        with self.store._connect() as db:
            db.execute("BEGIN IMMEDIATE")  # no second creator between the checks and the write
            if path.exists():
                raise ValueError("source exists; preserve and edit deliberately")
            if kind == "task" and db.execute("SELECT 1 FROM records WHERE id=?", (metadata["id"],)).fetchone():
                raise ValueError("task id exists")
            if entry.exists():  # an earlier write never finished
                raise RevisionConflict("pending source operation; sync and resolve first")
            atomic(entry, _json({"source": source, "old_hash": None, "content": content, "kind": kind}))
            atomic(path, content)
        result = self.sync()  # the index must see the new file before we read it back
        rid = metadata.get("id", "md-" + _hash(source)[:24])
        if any(isinstance(x, dict) and (x.get("source") == source or x.get("id") == rid)
               for key in ("warnings", "conflicts") for x in result[key]):
            raise RevisionConflict(f"{kind} requires projection review")
        with self.store._connect() as db:
            row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
        record = json.loads(row[0]) if row else {}
        if record.get("source") != source or record.get("source_sha256") != _hash(content) or path.read_bytes() != content.encode():
            raise RevisionConflict(f"created {kind} source readback mismatch")
        if redacted:
            record_redactions(self.state, redacted)
        return record, result, redacted

    def note_create(self, source, text, metadata=None):
        """Create notes/ or knowledge/ Markdown plus its record. Returns the redaction and sync summary."""
        if not (isinstance(text, str) and text.strip()) or not (metadata is None or isinstance(metadata, dict)):
            raise ValueError("note text and metadata required")
        if text.lstrip().startswith("---"):
            raise ValueError("text must be body only; put frontmatter fields in metadata")
        if not source.startswith(("notes/", "knowledge/")) or not source.endswith(".md"):
            raise ValueError("new semantic notes belong under notes/ or knowledge/")
        meta = {k: v for k, v in (metadata or {}).items() if not (k in ("project", "kind", "status", "updated_at") and v is None)}
        if meta.get("kind") in ("task", "receipt") or meta.get("generated"):
            raise ValueError("managed note kinds require their dedicated command")
        if "id" in meta and (not isinstance(meta["id"], str) or not meta["id"].strip()):
            raise ValueError("record id must be nonempty text")
        if not isinstance(meta.get("facts", {}), dict) or meta.get("visibility", "internal") not in VISIBILITY:
            raise ValueError("invalid facts or visibility")
        supersedes = [meta["supersedes"]] if isinstance(meta.get("supersedes"), str) else meta.get("supersedes", [])
        if not isinstance(supersedes, list) or not all(isinstance(v, str) for v in supersedes) or meta.get("id") in supersedes:
            raise ValueError("supersedes must list other record ids")
        with self.store._connect() as db:
            old = db.execute("SELECT payload FROM records WHERE id=?", (meta.get("id"),)).fetchone()
        if old and json.loads(old[0]).get("status") == "deleted_by_user":
            raise ValueError("record was deleted by the user")
        _, result, redacted = self._create(source, trust.stamp(meta), text, "note")
        return {"status": "succeeded", "source": source, "redacted": redacted, "secrets_redacted": redacted,
                "warnings": result["warnings"], "conflicts": result["conflicts"],
                "source_sync": {"status": result["status"], "warnings": result["warnings"], "conflicts": result["conflicts"]}}

    def task_create(self, source, text, metadata):
        """Create tasks/ Markdown plus its record. Needs an id, an owner and a status."""
        if not isinstance(source, str) or not source.startswith("tasks/") or not source.endswith(".md"):
            raise ValueError("task-create source must be a tasks/ Markdown path")
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict) or text.lstrip().startswith("---"):
            raise ValueError("task body and metadata required")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}", str(metadata.get("id"))):
            raise ValueError("stable task id required")
        if not isinstance(metadata.get("owner"), str) or not metadata["owner"].strip():
            raise ValueError("explicit task owner required")
        if metadata.get("status") not in STATUSES:
            raise ValueError("valid explicit task status required")
        record, result, redacted = self._create(source, dict(metadata, kind="task", revision=1), text, "task")
        return dict(record, redacted=redacted, secrets_redacted=redacted, warnings=result["warnings"],
                    conflicts=result["conflicts"], source_sync={"status": result["status"]})
