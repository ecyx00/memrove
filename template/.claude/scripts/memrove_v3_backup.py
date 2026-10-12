"""Backup and restore of a vault + state dir pair (requirement items 300-302). Stdlib only, no model call.

Layout of a backup dir: Vault/, state/, manifest.json (sha256 of every file + record count).
SQLite files are copied through the sqlite3 backup API (a consistent snapshot, WAL included).
Restore refuses non-empty targets (it never overwrites live data), checks the copy against the manifest and reports
whether the store and the vault agree (every live record's source file exists with the stored hash)."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

SKIP = ("-wal", "-shm", "-journal")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and not p.name.endswith(SKIP) and "lock" not in p.name)


def _copy_tree(src, dst):
    for p in _files(src):
        out = Path(dst) / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".sqlite3":
            with sqlite3.connect(p) as a, sqlite3.connect(out) as b:
                a.backup(b)
            b.close()
            a.close()
        else:
            shutil.copy2(p, out)


def consistency(vault, state):
    """Store vs vault: -> {records, missing: [id], changed: [id]}. Tombstones, source_missing and store-only records need no file."""
    out = {"records": 0, "missing": [], "changed": []}
    db_path = Path(state) / "memory.sqlite3"
    if not db_path.exists():
        return out
    db = sqlite3.connect(db_path)
    try:
        rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM records")]
        owned = {r[0] for r in db.execute("SELECT id FROM markdown_sources")}
    finally:
        db.close()
    out["records"] = len(rows)
    for r in rows:
        if r.get("status") in ("deleted_by_user", "source_missing") or r["id"] not in owned or not r.get("source"):
            continue
        f = Path(vault) / r["source"]
        if not f.is_file():
            out["missing"].append(r["id"])
        elif r.get("source_sha256") and _sha(f) != r["source_sha256"]:
            out["changed"].append(r["id"])  # edited since the last sync: the next sync picks it up, not an error
    return out


def backup(vault, state, dest):
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("backup destination must be empty or missing")
    _copy_tree(vault, dest / "Vault")
    _copy_tree(state, dest / "state")
    files = {str(p.relative_to(dest)): _sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    check = consistency(dest / "Vault", dest / "state")
    (dest / "manifest.json").write_text(json.dumps({"at": time.time(), "vault": str(Path(vault).resolve()), "state": str(Path(state).resolve()), "files": files, "consistency": check}, indent=1), encoding="utf-8")
    return {"status": "backed_up", "dest": str(dest), "files": len(files), "consistency": check}


def _swap(text, pairs):
    """Replace old absolute paths; PowerShell -EncodedCommand blobs (base64 UTF-16LE) are decoded, swapped and re-encoded."""
    def enc(m):
        inner = base64.b64decode(m.group(2)).decode("utf-16-le")
        for a, b in pairs:
            inner = inner.replace(a, b)
        return m.group(1) + base64.b64encode(inner.encode("utf-16-le")).decode()
    text = re.sub(r"(-EncodedCommand )([A-Za-z0-9+/=]+)", enc, text)
    for a, b in pairs:
        text = text.replace(a, b)
    return text


def rebind(manifest, vault, state):
    """Point the restored state at the target vault/state: store binding, install record (commands + the managed files it
    pins, with their hashes), last-update.json, and the managed files themselves. -> list of what changed.
    Old paths come from the manifest (new backups) or the store binding + .memrove-runtime.json (older backups)."""
    vault, state = Path(vault).resolve(), Path(state).resolve()
    old_vault, old_state = manifest.get("vault"), manifest.get("state")
    db_path = state / "memory.sqlite3"
    if db_path.exists():
        db = sqlite3.connect(db_path)
        try:
            row = db.execute("SELECT value FROM metadata WHERE key='vault_root'").fetchone()
        finally:
            db.close()
        old_vault = old_vault or (row and row[0])
    runtime = vault / ".memrove-runtime.json"
    if not old_state and runtime.is_file():
        old_state = json.loads(runtime.read_text(encoding="utf-8")).get("state")
    pairs = [(a, str(b)) for a, b in ((old_vault, vault), (old_state, state)) if a and a != str(b)]
    if not pairs:
        return []
    done = []
    if db_path.exists():
        db = sqlite3.connect(db_path)
        try:
            with db:
                db.execute("UPDATE metadata SET value=? WHERE key='vault_root'", (str(vault),))
        finally:
            db.close()
        done.append("store vault_root")
    record = state / "v3-install.json"
    data = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else None
    for name, entry in (data or {}).get("files", {}).items():
        f = vault / name
        if not f.is_file():
            continue
        raw = f.read_bytes()
        new = _swap(raw.decode("utf-8"), pairs).encode("utf-8")
        if new != raw:
            f.write_bytes(new)
            entry["installed_hash"] = hashlib.sha256(new).hexdigest()
            entry["installed_content"] = base64.b64encode(new).decode()
            done.append(name)
    if data:
        data["commands"] = [_swap(c, pairs) for c in data.get("commands", [])]
        record.write_text(json.dumps(data, indent=2), encoding="utf-8")
        done.append("v3-install.json")
    last = state / "last-update.json"
    if last.is_file() and any(a in last.read_text(encoding="utf-8") for a, _ in pairs):
        last.write_text(_swap(last.read_text(encoding="utf-8"), pairs), encoding="utf-8")
        done.append("last-update.json")
    return done


def restore(src, vault, state):
    src = Path(src)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, sha in manifest["files"].items() if not (src / name).is_file() or _sha(src / name) != sha]
    if bad:
        raise ValueError("backup is damaged: " + ", ".join(bad[:5]))
    for target in (vault, state):
        if Path(target).exists() and any(Path(target).iterdir()):
            raise ValueError(f"{target} is not empty; move it aside first, restore never overwrites")
    _copy_tree(src / "Vault", vault)
    _copy_tree(src / "state", state)
    rebound = rebind(manifest, vault, state)
    check = consistency(vault, state)
    return {"status": "restored" if not check["missing"] else "restored_inconsistent", "consistency": check, "rebound": rebound,
            "next": "run `sync` then `doctor`; a restore only counts once a capture/sync/inject/recall cycle works"}
