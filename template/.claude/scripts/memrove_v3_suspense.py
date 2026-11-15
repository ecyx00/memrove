"""Receipt and close: neighbours a save showed and nobody answered are booked as (new, neighbour) pairs; `settle` answers one;
`card_line` counts what is still open. Deterministic, no model call. Booking never raises into `remember`; the record is saved first."""
import json
from datetime import date

SCHEMA = ("CREATE TABLE IF NOT EXISTS suspense(new_id TEXT NOT NULL, nb_id TEXT NOT NULL, day TEXT NOT NULL, session TEXT, project TEXT, "
          "state TEXT NOT NULL DEFAULT 'open', PRIMARY KEY(new_id, nb_id))")   # state: open | replaces | rests_on | unrelated
MAX_IDS = 5
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


def _mark(db, new, nb, state):
    p = _payload(db, new)
    db.execute("INSERT INTO suspense(new_id,nb_id,day,session,project,state) VALUES (?,?,?,?,?,?) ON CONFLICT(new_id,nb_id) DO UPDATE SET state=excluded.state",
               (new, nb, date.today().isoformat(), p.get("session"), p.get("project"), state))


def _replace(engine, db, new, nb):
    """The link is the new record's `supersedes`; the stored status becomes `superseded` only when it was active (a retired or untrusted record keeps its status)."""
    from memrove_v3_projections import update_record
    had = _payload(db, new).get("supersedes") or []
    if nb not in had:
        update_record(engine, new, {"supersedes": [*had, nb]}, db)
    old = _payload(db, nb)   # the same stored state as `remember --replaces`: closed window and back reference, status only when active
    by = old.get("superseded_by") or []
    changes = {**({"status": "superseded"} if old.get("status") in (None, "active") else {}),
               **({"valid_to": date.today().isoformat()} if not old.get("valid_to") else {}),
               **({"superseded_by": [*by, new]} if new not in by else {})}
    if changes:
        update_record(engine, nb, changes, db)


def answer(engine, new, replaces, rests, session=None):
    """The save's own answers. First each pair is booked open and committed, then the links, status and pair state are written in one
    transaction: a failed or interrupted link leaves the open pair for the card. Also used for a record that already exists."""
    import memrove_v3_graph as G
    prov = G.provenance("agent:memrove-cli", "agent", session=session)
    replaces, rests = [o for o in replaces if o != new], [(o, w) for o, w in rests if o != new]   # identical text comes back with its own id
    if not (replaces or rests):
        return
    with engine.store._connect() as db:
        _begin(db)
        for old in [*replaces, *(o for o, _ in rests)]:
            _mark(db, new, old, "open")
    with engine.store._connect() as db:
        _begin(db)
        for old, why in rests:
            G.add_edge(db, new, old, "rests_on", why, prov)
            _mark(db, new, old, "rests_on")
        for old in replaces:
            _replace(engine, db, new, old)
            _mark(db, new, old, "replaces")


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
    from memrove_v3_nudge import GONE
    from memrove_v3_projections import update_record
    with engine.store._connect() as db:
        _begin(db)
        G.ensure(db)
        new, nb = resolve(db, new), resolve(db, nb)
        if new == nb:
            raise ValueError("a record cannot be settled against itself")
        if mode == "replaces":
            _replace(engine, db, new, nb)
        elif mode == "rests_on":
            G.add_edge(db, new, nb, "rests_on", why, G.provenance("agent:memrove-cli", "agent"))
        elif mode == "undo":
            had = _payload(db, new).get("supersedes") or []
            if nb in had:
                update_record(engine, new, {"supersedes": [x for x in had if x != nb]}, db)
                others = [p for (p,) in db.execute("SELECT payload FROM records WHERE EXISTS (SELECT 1 FROM json_each(records.payload,'$.supersedes') WHERE value=?)", (nb,))]
                # restore only what a replace set: a record with another status, or another live replacer, is left alone
                old = _payload(db, nb)
                changes = {"superseded_by": [x for x in old.get("superseded_by") or [] if x != new]}
                if not any(json.loads(p).get("status") not in GONE for p in others):
                    changes.update({"valid_to": None} if new in (old.get("superseded_by") or []) else {},
                                   **({"status": "active"} if old.get("status") == "superseded" else {}))
                update_record(engine, nb, changes, db)
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


def card_line(engine, project, session):
    """One line: counts and at most MAX_IDS ids of open pairs (new>neighbour, both sides live), one-sided replaces and records with two live
    replacers in the project; '' when all are zero. Never raises."""
    try:
        from memrove_v3_nudge import GONE
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
        groups = [(f"open pairs: {len(pairs)} in project, {sum(p[2] == session for p in pairs)} this session", [f"{p[0]}>{p[1]}" for p in pairs], len(pairs)),
                  ("one-sided replaces", one_sided, None),
                  ("records with two live replacers", sorted(t for t, v in replacers.items() if len(v) > 1), None)]
        out, room = [], MAX_IDS
        for name, ids, n in groups:
            if not (n or ids):
                continue
            out.append((f"{name}" if n else f"{name}: {len(ids)}") + (f" [{' '.join(ids[:room])}]" if room and ids else ""))
            room = max(0, room - len(ids))
        return "memrove open: " + "; ".join(out) if out else ""
    except Exception:
        return ""
