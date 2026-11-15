"""Receipt and close: neighbours a save showed and nobody answered are booked as (new, neighbour) pairs; `settle` answers one;
`card_line` counts what is still open. Deterministic, no model call. Booking never raises into `remember`; the record is saved first."""
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from datetime import date

from memrove_v3_nudge import GONE

SCHEMA = ("CREATE TABLE IF NOT EXISTS suspense(new_id TEXT NOT NULL, nb_id TEXT NOT NULL, day TEXT NOT NULL, session TEXT, project TEXT, "
          "state TEXT NOT NULL DEFAULT 'open', PRIMARY KEY(new_id, nb_id))")   # state: open | replaces | rests_on | unrelated
MAX_IDS = 5
OFF = GONE[:-1]   # a replacer in any of these statuses is dead; a `superseded` one (chain A<-B<-C) still counts as A's replacer
MISSING = "\0missing"


def _begin(db):
    """Take the write lock before any read, so a read-then-write answer cannot interleave with another writer."""
    db.execute("BEGIN IMMEDIATE")
    db.execute(SCHEMA)


def resolve(db, ref):
    """Record id from a unique id or prefix."""
    ref = ref.removeprefix("rec:")
    ids = [r[0] for r in db.execute("SELECT id FROM records WHERE id=? OR id LIKE ? LIMIT 2", (ref, ref + "%"))]
    if ref in ids:
        return ref
    if len(ids) != 1:
        raise ValueError(f"no unique record for {ref!r}")
    return ids[0]


def _payload(db, rid):
    return json.loads(db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0])


def _mark(db, new, nb, state, keep=False):
    """Set a pair's state; keep=True books the pair only when it is new (a pair already answered is not reopened)."""
    p = _payload(db, new)
    db.execute("INSERT INTO suspense(new_id,nb_id,day,session,project,state) VALUES (?,?,?,?,?,?) ON CONFLICT(new_id,nb_id) DO "
               + ("NOTHING" if keep else "UPDATE SET state=excluded.state"), (new, nb, date.today().isoformat(), p.get("session"), p.get("project"), state))


class _Journal:
    """The note files a transaction rewrote, with their text from before. It is also on disk, fsynced before each write: a process killed before the
    commit leaves it, and the next SyncEngine puts the notes back (SyncEngine.resolve_rollbacks)."""
    def __init__(self, engine):
        self.entries, self.file = [], engine.rollback / f"{os.getpid()}-{uuid.uuid4().hex}.json"

    def add(self, path, source, old, sha):
        from memrove_v3_sync import atomic
        self.entries.append((path, {"source": source, "old": old, "sha": sha}))
        atomic(self.file, json.dumps([e for _, e in self.entries]))

    def restore(self):
        """Every file, newest first; a file that fails is named on stderr and the journal stays for the next command. Never raises."""
        from memrove_v3_sync import atomic
        failed = []
        for path, e in reversed(self.entries):
            try:
                atomic(path, e["old"])
            except Exception:
                failed.append(e["source"])
        if failed:
            print(f"memrove: could not put back {', '.join(failed)}; the next memrove command finishes it", file=sys.stderr)
        else:
            self.file.unlink(missing_ok=True)


@contextmanager
def _tx(engine):
    """-> (db, journal): one write transaction. On any error (Ctrl-C included) the notes are put back while the write lock is still held, then the
    transaction rolls back; after the commit the journal goes."""
    journal = _Journal(engine)
    with engine.store._connect() as db:
        _begin(db)
        engine.rollback_journal = journal   # project_record (an owned note) journals its write too
        try:
            yield db, journal
        except BaseException:
            journal.restore()
            raise
        finally:
            engine.rollback_journal = None
    journal.file.unlink(missing_ok=True)


def _file_fields(engine, db, rid, fields, journal):
    """Write fields (a falsy value drops the key) into the front matter of a note of memrove's own (notes/observed/...), so a store rebuilt
    from the vault or a hand edit keeps them (as Core._fields_to_file does for a close). -> {'source_sha256': new} for the payload, or {}.
    Any other note (a hand-written Vault file) is never rewritten: its link and status live in the store and in the replacer's own note.
    A note whose front matter `edit_front` cannot change safely is left alone too: the field stays in the store only, and stderr says so."""
    from memrove_v3_sync import _hash, atomic, edit_front
    rec = _payload(db, rid)
    if not fields or not str(rec.get("source") or "").startswith("notes/observed/"):
        return {}
    try:
        path = engine._path(rec["source"], existing=True)
        old = path.read_bytes().decode("utf-8")   # bytes, not read_text: no newline translation
    except (KeyError, ValueError, OSError):
        return {}   # no file behind this record
    if (text := edit_front(old, fields)) is None:
        print(f"memrove: {rec['source']}: front matter cannot be edited safely; {', '.join(fields)} kept in the store only", file=sys.stderr)
        return {}
    if text == old:
        return {}
    journal.add(path, rec["source"], old, _hash(text))
    atomic(path, text)
    return {"source_sha256": _hash(text)}


def _replace(engine, db, new, nb, journal):
    """The link is the new record's `supersedes`; the stored status becomes `superseded` only when it was active (a retired or untrusted record keeps its status)."""
    from memrove_v3_projections import update_record
    had = _payload(db, new).get("supersedes") or []
    if nb not in had:
        update_record(engine, new, {"supersedes": (sup := [*had, nb]), **_file_fields(engine, db, new, {"supersedes": sup}, journal)}, db)
    old = _payload(db, nb)   # the same stored state as `remember --replaces`: closed window and back reference, status only when active
    by = old.get("superseded_by") or []
    changes = {**({"status": "superseded"} if old.get("status") in (None, "active") else {}),
               **({"valid_to": date.today().isoformat()} if not old.get("valid_to") else {}),
               **({"superseded_by": [*by, new]} if new not in by else {})}
    if changes:
        update_record(engine, nb, {**changes, **_file_fields(engine, db, nb, {k: changes[k] for k in ("valid_to", "superseded_by") if k in changes}, journal)}, db)


def _reaches(db, start, goal):
    """True when a chain of `supersedes` links leads from start to goal."""
    seen, todo = set(), [start]
    while todo:
        if (x := todo.pop()) not in seen:
            seen.add(x)
            row = db.execute("SELECT payload FROM records WHERE id=?", (x,)).fetchone()
            nxt = json.loads(row[0]).get("supersedes") or [] if row else []
            if goal in nxt:
                return True
            todo += nxt
    return False


def _not_live(db, new, nb, ok=None):
    """Why `new` cannot replace `nb` ('' when it can): nb is closed, or a live record already replaces it. A link `new` already has is no error."""
    if nb in (_payload(db, new).get("supersedes") or []):
        return ""
    if (st := _payload(db, new).get("status")) in GONE:   # a reversed settle: the replaced record would replace the live one and both would vanish
        return f"{new} is not live (status {st}) and cannot replace {nb}"
    if _reaches(db, nb, new) or nb in (_payload(db, new).get("superseded_by") or []):   # also a pair replaced before the stored status was set
        return f"{nb} already replaces {new}; the reverse link would hide both"
    by = [i for (i, p) in db.execute("SELECT id, payload FROM records WHERE id<>? AND EXISTS (SELECT 1 FROM json_each(payload,'$.supersedes') j WHERE j.value=?)", (nb, nb))
          if i != new and json.loads(p).get("status") not in GONE]
    status = _payload(db, nb).get("status")
    if by:   # a restricted reader is not told the id of a replacer it may not see
        seen = [i for i in by if not ok or ok(i)]
        return f"{nb} is already replaced by {', '.join(seen)}" if seen else f"{nb} is already replaced"
    return f"{nb} is not live (status {status})" if status in GONE else ""


def answer(engine, new, replaces, rests, session=None, ok=None):
    """The save's own answers, given as ids or prefixes -> (ids answered, error lines). A target that is missing, ambiguous or (for --replaces) not live
    is an error line and no link; the others go on. Each pair is booked open and committed first, then the links, the replaced record's
    status / valid_to / superseded_by and the pair states are written in ONE transaction: a failure leaves the open pair and nothing else."""
    import memrove_v3_graph as G
    prov = G.provenance("agent:memrove-cli", "agent", session=session)
    errors, reps, rsts = [], [], []
    with engine.store._connect() as db:
        _begin(db)
        for flag, ref, why in [("--replaces", r, None) for r in replaces] + [("--rests-on", r, w) for r, w in rests]:
            try:
                old = resolve(db, ref)
                problem = "" if old == new or why is not None else _not_live(db, new, old, ok)
            except ValueError as exc:
                old, problem = None, str(exc)
            if problem:
                errors.append(f"error: {flag} {ref}: {problem}; no link written (the record {new} is saved)")
            elif old != new:   # identical text comes back with its own id
                (rsts if why is not None else reps).append((old, why) if why is not None else old)
                _mark(db, new, old, "open", keep=True)
    if not (reps or rsts):
        return [], errors
    with _tx(engine) as (db, journal):
        for old, why in rsts:
            G.add_edge(db, new, old, "rests_on", why, prov)
            _mark(db, new, old, "rests_on")
        for old in reps:
            if bad := _not_live(db, new, old, ok):   # again under the write lock: another save may have linked it since commit 1
                raise ValueError(f"{bad}; no link written")
            _replace(engine, db, new, old, journal)
            _mark(db, new, old, "replaces")
    try:
        from memrove_v3_checks import forget
        forget(engine.state, reps)
    except Exception:
        pass
    return [*reps, *(o for o, _ in rsts)], errors


def offer(engine, state, project, text, new, session=None, answered=()):
    """-> the neighbour lines to print; each shown neighbour is booked as an open pair (a pair already settled is neither shown nor re-booked). Never raises."""
    try:
        from memrove_v3_nudge import line, neighbours
        nbs = [n for n in neighbours(state, project, text, new) if n[0] not in answered]
        try:
            with engine.store._connect() as db:
                db.execute(SCHEMA)
                settled = {r[0] for r in db.execute("SELECT nb_id FROM suspense WHERE new_id=? AND state!='open'", (new,))}
                nbs = [n for n in nbs if n[0] not in settled]
                db.executemany("INSERT OR IGNORE INTO suspense(new_id,nb_id,day,session,project) VALUES (?,?,?,?,?)",
                               [(new, n[0], date.today().isoformat(), session, project) for n in nbs])
        except Exception:   # booking failed: the lines are still shown and the record is saved
            pass
        return [line(n, new) for n in nbs]
    except Exception:
        return []


def settle(engine, new, nb, mode, why=None):
    """-> one line. mode: replaces | rests_on | unrelated | undo. Link, stored status and pair state change in one transaction."""
    import memrove_v3_graph as G
    from memrove_v3_projections import update_record
    with _tx(engine) as (db, journal):
        G.ensure(db)
        new, nb = resolve(db, new), resolve(db, nb)
        if new == nb:
            raise ValueError("a record cannot be settled against itself")
        if mode == "replaces":
            if bad := _not_live(db, new, nb):
                raise ValueError(f"{bad}; no link written")
            _replace(engine, db, new, nb, journal)
        elif mode == "rests_on":
            G.add_edge(db, new, nb, "rests_on", why, G.provenance("agent:memrove-cli", "agent"))
        elif mode == "undo":
            had = _payload(db, new).get("supersedes") or []
            if nb in had:
                keep = [x for x in had if x != nb]
                update_record(engine, new, {"supersedes": keep, **_file_fields(engine, db, new, {"supersedes": keep}, journal)}, db)
                others = [p for (p,) in db.execute("SELECT payload FROM records WHERE EXISTS (SELECT 1 FROM json_each(records.payload,'$.supersedes') WHERE value=?)", (nb,))]
                # restore only what a replace set: a record with another status, or another live replacer, is left alone
                old = _payload(db, nb)
                changes = {"superseded_by": [x for x in old.get("superseded_by") or [] if x != new]}
                if not any(json.loads(p).get("status") not in GONE for p in others):
                    changes.update({"valid_to": None} if new in (old.get("superseded_by") or []) else {},
                                   **({"status": "active"} if old.get("status") == "superseded" else {}))
                update_record(engine, nb, {**changes, **_file_fields(engine, db, nb, {k: changes[k] for k in ("valid_to", "superseded_by") if k in changes}, journal)}, db)
            elif db.execute("DELETE FROM graph_edges WHERE src=? AND dst=? AND type='rests_on'", (new, nb)).rowcount == 0:
                raise ValueError(f"no replaces or rests-on link from {new} to {nb}")
        _mark(db, new, nb, "open" if mode == "undo" else mode)
    if mode in ("replaces", "undo"):
        try:
            from memrove_v3_checks import forget
            forget(engine.state, [nb])
        except Exception:
            pass
    return f"{new} {nb} {mode}"


def card_line(engine, project, session, ok=None):
    """One line: counts and ids (about MAX_IDS, one at least per group) of open pairs (new>neighbour, both sides live), one-sided replaces, records with two
    live replacers, replaced records with no live replacer, replace cycles and live records whose rests-on target was replaced; '' when all are zero. Never raises."""
    try:
        with engine.store._connect() as db:
            db.execute(SCHEMA)
            pairs = db.execute("SELECT new_id, nb_id, session FROM suspense WHERE state='open' AND project=? ORDER BY day DESC, new_id, nb_id", (project,)).fetchall()
            recs = [json.loads(p) for (p,) in db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.project')=?", (project,))]
            status = {r["id"]: r.get("status") for r in recs}

            def state(rid):
                if rid not in status:
                    row = db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()
                    status[rid] = json.loads(row[0]).get("status") if row else MISSING
                return status[rid]
            pairs = [p for p in pairs if state(p[0]) not in GONE + (MISSING,) and state(p[1]) not in GONE + (MISSING,)]
            replacers = {}
            for r in recs:
                if r.get("status") not in GONE:
                    for t in r.get("supersedes") or []:
                        replacers.setdefault(t, []).append(r["id"])
            booked = set(db.execute("SELECT new_id, nb_id FROM suspense"))   # a replace made before this feature has no pair: not counted
            one_sided = sorted(t for t, v in replacers.items() if state(t) == "active" and any((r, t) in booked for r in v))
            def moved(t):   # replaced: closed by a link, or a live replacer holds it
                return state(t) == "superseded" or t in replacers
            named, held = set(), set()   # any project; `held`: a replacer that is itself replaced still holds its target; `named`: every record's supersedes, a retired replacer included (recall hides those targets too)
            for st, sup in db.execute("SELECT coalesce(json_extract(payload,'$.status'),''), json_extract(payload,'$.supersedes') FROM records WHERE json_extract(payload,'$.supersedes') IS NOT NULL"):
                named |= set(json.loads(sup))
                if st not in OFF:
                    held |= set(json.loads(sup))
            lost = sorted(r["id"] for r in recs if r.get("status") not in OFF and (r.get("status") == "superseded" or r["id"] in named) and r["id"] not in held)
            succ = {r["id"]: r.get("supersedes") or [] for r in recs if r.get("supersedes")}

            def reach(n):   # records a replace chain from n leads to
                seen, todo = set(), list(succ.get(n, ()))
                while todo:
                    x = todo.pop()
                    if x not in seen:
                        seen.add(x)
                        todo += succ.get(x, ())
                return seen
            cycle = sorted(n for n in succ if n in reach(n))
            try:   # a live record that rests on a replaced one: its basis changed
                changed = sorted({s for s, d in db.execute("SELECT src, dst FROM graph_edges WHERE type='rests_on'") if s in status and state(s) not in GONE and moved(d)})
            except Exception:   # no graph_edges table yet
                changed = []
        groups = [(f"open pairs: {len(pairs)} in project, {sum(p[2] == session for p in pairs)} this session", [f"{p[0]}>{p[1]}" for p in pairs if not ok or (ok(p[0]) and ok(p[1]))], len(pairs)),
                  ("one-sided replaces", one_sided, None),
                  ("records with two live replacers", sorted(t for t, v in replacers.items() if len(v) > 1), None),
                  ("replaced, no live replacer", lost, None), ("replace cycle", cycle, None), ("basis changed", changed, None)]
        out, room = [], MAX_IDS
        for name, ids, n in groups:
            if not (n or ids):
                continue
            label = f"{name}" if n else f"{name}: {len(ids)}"
            if not n and ok:
                ids = [i for i in ids if ok(i)]   # a restricted reader gets the count, and only the ids it may see
            out.append(label + (f" [{' '.join(ids[:max(room, 1)])}]" if ids else ""))
            room = max(0, room - len(ids))
        return "memrove open: " + "; ".join(out) if out else ""
    except Exception:
        return ""
