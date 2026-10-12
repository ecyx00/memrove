"""Projections of store records into Markdown notes, graph views and verdict ingestion."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

import memrove_v3_graph as _graph
import memrove_v3_papercuts as _papercuts

try:  # E3 builds the link table; the projection derives core edges itself until it lands.
    import memrove_v3_links as _links
except ImportError:
    _links = None

MEMROVE = 'memrove'
NODE_DIRS = {'decision': 'decisions', 'preference': 'preferences', 'rejected': 'rejected', 'lesson': 'lessons'}
VERDICTS = {'keep', 'reject', 'pin'}
RECENT = 10
LATEST = 12  # newest records per kind that a project hub lists; older ones are reached through sessions and concepts


def _slug(text, default='genel'):
    text = unicodedata.normalize('NFKD', str(text or '')).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', text).strip('-') or default


def _file_id(record_id):
    safe = re.sub(r'[^A-Za-z0-9._-]+', '-', record_id).strip('-.') or 'x'
    return safe if safe == record_id else safe[:60] + '-' + hashlib.sha256(record_id.encode()).hexdigest()[:8]


def _project_slug(record):
    return _graph.view_project(record.get('project'))  # the home folder, unknown and claude are 'genel' in the view


def _fence(text):
    run = max([len(m) for m in re.findall(r'`+', text)] + [2]) + 1
    return '`' * run + '\n' + text + '\n' + '`' * run


def _link(path):
    return '[[' + path[:-3] + ']]'


def _edge_ids(node_id):
    """Extra edges from E3's link table: [(dst, rel)]; tolerant of tuple/dict shapes."""
    if _links is None or not hasattr(_links, 'for_node'):
        return []
    try:
        out = []
        for edge in _links.for_node(node_id) or []:
            if isinstance(edge, dict):
                out.append((edge.get('dst'), edge.get('rel')))
            elif len(edge) >= 4:
                out.append((edge[1] if edge[0] == node_id else edge[0], edge[2]))
            else:
                out.append((edge[0], edge[1]))
        return out
    except Exception:  # a broken optional layer must not stop projection
        return []


DOC_KINDS = ('reference', 'rule', 'problem', 'solution', 'daylog', 'note')


def _doc_paths(docs):
    """doc id -> stub path memrove/docs/<project>/<slug>.md; colliding slugs all get a stable hash suffix."""
    slugs = {d[0]: _slug(d[2].rsplit('.', 1)[0])[:60] or 'doc' for d in docs}
    count = {}
    for d in docs:
        count[(_graph.view_project(d[1]), slugs[d[0]])] = count.get((_graph.view_project(d[1]), slugs[d[0]]), 0) + 1
    return {d[0]: f"{MEMROVE}/docs/{_graph.view_project(d[1])}/{slugs[d[0]]}" + (('-' + hashlib.sha256(d[0].encode()).hexdigest()[:8]) if count[(_graph.view_project(d[1]), slugs[d[0]])] > 1 else '') + '.md' for d in docs}


def _doc_notes(docs, edges, doc_paths, paths, render):
    by_src = {}
    for src, dst, rel, _w in edges:
        by_src.setdefault(src, []).append((rel, dst))
    out = {}
    for d in docs:
        id_, project, rel, kind, title, summary, day = d[0], d[1], d[2], d[5], d[6], d[7], d[8]
        slug = _graph.view_project(project)
        lines = [f'# {title}', '', f'- Kind: {kind}', f'- Source: `{rel}`', f'- Project: {_link(paths["project:" + slug])}']
        if day:
            lines.append(f'- Day: {day}')
        if summary:
            lines += ['', summary[:240] + ('…' if len(summary) > 240 else '')]
        lines += ['']
        for erel, dst in sorted(set(by_src.get(id_, []))):
            if erel == 'about' or dst == id_:
                continue
            if dst in doc_paths:
                lines.append(f'- {erel}: {_link(doc_paths[dst])}')
            elif dst.startswith(('area:', 'topic:')):
                lines.append(f'- {erel}: `{dst.split(":", 1)[1]}`')
        for src, others in sorted(by_src.items()):  # inbound solves
            if src != id_ and src in doc_paths and ('solves', id_) in others:
                lines.append(f'- solved by: {_link(doc_paths[src])}')
        lines += ['']
        out[doc_paths[id_]] = render({'generated': True, 'kind': 'doc', 'doc_kind': kind, 'id': id_, 'project': slug}, '\n'.join(lines))
    return out


def _entity_notes(entities, docs, doc_paths, paths, render):
    proj = {d[0]: _graph.view_project(d[1]) for d in docs}
    out = {}
    for e in entities:
        lines = [f'# {e["name"]}', '', f'- Kind: {e.get("entity_kind", "concept")}']
        if e.get('aliases'):
            lines.append('- Aliases: ' + ', '.join(e['aliases']))
        lines += ['', '## Profile', ''] + ([f'- {p["s"]}: {p["t"]}' for p in e.get('profile', [])] or ['- (not compiled yet)'])
        lines += ['', '## Sources', ''] + [f'- {_link(doc_paths[s])}' for s in e['sources'] if s in doc_paths]
        hubs = sorted({proj[s] for s in e['sources'] if s in proj and 'project:' + proj[s] in paths})
        if hubs:
            lines += ['', '## Projects', ''] + [f'- {_link(paths["project:" + h])}' for h in hubs]
        lines += ['']
        out[f"{MEMROVE}/entities/{e['id'].split(':', 1)[1]}.md"] = render({'generated': True, 'kind': 'entity', 'id': e['id'], 'stale': bool(e.get('stale'))}, '\n'.join(lines))
    return out


def _note_link(source):
    return '[[' + source[:-3] + ']]'


def _build(records, render, session_paths=None, docs=None, entities=None, live_sources=None, graph=None):
    """Pure: records -> {relative_path: content}. Deterministic.

    live_sources: set of note paths known to exist on disk (None = trust the records). Owned and user
    notes are only linked while their file exists, so generated links never dangle.
    graph: {'concepts': {id: {name, why, prov}}, 'edges': [{src, dst, type, why, prov}], 'cards': [card dicts],
    'session_days': {session: day}, 'hub_targets': {slug}, 'wiki_concepts': {slug}} from the store; edges are rendered, never invented here."""
    graph = graph or {}
    G = _graph

    def alive(r):
        src = str(r.get('source') or '')
        return bool(src) and r.get('status') != 'deleted_by_user' and (live_sources is None or src in live_sources)
    records = [r for r in records if r.get('visibility', 'internal') != 'private'
               and r.get('status') != 'untrusted' and r.get('trusted') is not False]
    records.sort(key=lambda r: r['id'])
    paths, notes = {}, [r for r in records if r.get('kind') in NODE_DIRS]
    titles, taken = {}, {}
    for r in notes:
        titles[r['id']] = G.record_title(r)
        paths[r['id']] = f"{MEMROVE}/{NODE_DIRS[r['kind']]}/{G.node_name(titles[r['id']], r['id'], taken)}.md"
    cards = graph.get('cards') or []
    for c in cards:
        paths[c['id']] = c['source']
        titles[c['id']] = c['title']
    docs, doc_edges = docs or ([], [])
    doc_paths = _doc_paths(docs)
    paths.update(doc_paths)
    owned = [r for r in records if str(r.get('source') or '').startswith(OWNED) and alive(r)]
    card_ids = {c['id'] for c in cards}
    users = [r for r in records if r.get('kind') == 'note' and alive(r) and r['id'] not in card_ids and not str(r['source']).startswith(OWNED + (MEMROVE + '/',))]
    card_projects = {c['project'] for c in cards}
    hub_targets = set(graph.get('hub_targets') or ())
    projects = sorted({_project_slug(r) for r in records if r.get('kind') in NODE_DIRS or r.get('kind') == 'task' or r.get('session')}
                      | {G.view_project(d[1]) for d in docs} | {_project_slug(r) for r in owned}
                      | {_project_slug(r) for r in users if r.get('project')} | card_projects)
    alias_slugs = sorted(hub_targets - set(projects))  # a note written before the junk mapping still points at its old hub
    if alias_slugs and G.GENERIC not in projects:
        projects = sorted(projects + [G.GENERIC])
    user_hub = {}  # user note id -> hub slug (record project, else a file stem that names a project)
    for r in users:
        stem = _slug(str(r['source']).rsplit('/', 1)[-1].rsplit('.', 1)[0])
        user_hub[r['id']] = _project_slug(r) if r.get('project') else (stem if stem in projects else None)
    src_of = {r['id']: r['source'] for r in owned}
    for r in notes:  # an observed note is a source too, whatever directory holds it
        if r['id'] not in src_of and alive(r) and str(r.get('source') or '').startswith('notes/') and (live_sources is not None):
            src_of[r['id']] = r['source']
    for slug in projects:
        paths['project:' + slug] = f'{MEMROVE}/projects/{slug}.md'
    day_of = lambda r: str(r.get('updated_at') or r.get('created_at') or r.get('valid_from') or r.get('last_seen') or r.get('recorded_at') or '')[:10]
    # sessions: id -> (project, date, member records)
    sessions = {}
    for r in records:
        sid = r.get('session') or (r['id'] if r.get('kind') == 'session' else None)
        if sid:
            s = sessions.setdefault(sid, {'project': _project_slug(r), 'dates': [], 'members': [], 'files': []})
            if day_of(r):
                s['dates'].append(day_of(r))
            s['files'] += [str(f) for f in r.get('files') or []]
            if r.get('kind') != 'session' or r['id'] in src_of:
                s['members'].append(r)
    session_days = graph.get('session_days') or {}
    for sid, s in sessions.items():
        s['date'] = min(s['dates']) if s['dates'] else session_days.get(sid) or 'tarihsiz'
        paths['session:' + sid] = f"{MEMROVE}/sessions/{G.node_name(s['date'] + '-' + s['project'], sid, taken)}.md"
        paths.setdefault(sid, paths['session:' + sid])
    concepts = {cid: c for cid, c in (graph.get('concepts') or {}).items()}
    edges = [e for e in graph.get('edges') or []]
    for c in cards:  # a group is the owner's word for a concept: derived, but explicit
        for g in c.get('groups') or []:
            cid = G.concept_id(g)
            concepts.setdefault(cid, {'name': g, 'why': 'Grab group', 'prov': {'by': 'owner', 'basis': 'user', 'at': ''}})
            edges.append({'src': c['id'], 'dst': cid, 'type': 'part_of', 'why': 'group chosen at capture', 'prov': {'by': 'owner', 'basis': 'user', 'at': ''}})
    back = {}  # concept id -> [edges into it]
    for e in edges:
        if e['dst'] in concepts and e['src'] in paths:
            back.setdefault(e['dst'], []).append(e)
    wiki_concepts = set(graph.get('wiki_concepts') or ())
    concept_slug_of = lambda cid: cid.split(':', 1)[1]
    for cid in concepts:
        if back.get(cid) or any(e['src'] == cid and e['dst'] in paths for e in edges):
            paths[cid] = G.concept_path(cid)
            titles[cid] = concepts[cid]['name']
    node_project = {r['id']: _project_slug(r) for r in notes}
    node_project.update({c['id']: c['project'] for c in cards})
    proj_concepts = {}  # project -> {concept id: count}
    concept_projects = {}  # concept id -> {project: count}
    for cid, es in back.items():
        for e in es:
            p = node_project.get(e['src'])
            if p:
                proj_concepts.setdefault(p, {})[cid] = proj_concepts.setdefault(p, {}).get(cid, 0) + 1
                concept_projects.setdefault(cid, {})[p] = concept_projects.setdefault(cid, {}).get(p, 0) + 1
    by_key = {}
    for r in notes:
        by_key.setdefault((r.get('project'), r.get('key')), []).append(r)
    successor = {}
    for r in notes:
        for old in r.get('supersedes') or []:
            successor.setdefault(old, []).append(r['id'])
    proj_sessions = {}
    for sid, s in sessions.items():
        proj_sessions.setdefault(s['project'], []).append((s['date'], sid))
    for v in proj_sessions.values():
        v.sort(reverse=True)

    def known(node):
        return paths.get(node)

    def ref(node):  # `[[path|title]]` of a known node
        return G.link(paths[node], titles.get(node))

    out_edges, in_edges = {}, {}
    for e in edges:
        if e['src'] in paths and e['dst'] in paths and e['src'] != e['dst']:
            out_edges.setdefault(e['src'], []).append(e)
            if e['src'] in card_ids:  # a card's file is the owner's: its edges are shown where we may write
                in_edges.setdefault(e['dst'], []).append(e)

    def edge_line(e, other):
        return f"- {e['type']}: {ref(other)} — {e['why']}"

    out = {}
    for r in notes:
        slug, sid = _project_slug(r), r.get('session')
        text = str(r.get('quote') or r['text'])
        lines = [f"# {titles[r['id']]}", '', _fence(text), '', f'- Project: {_link(paths["project:" + slug])}']
        if r['id'] in src_of:
            lines.append(f'- Source note: {_note_link(src_of[r["id"]])}')
        if sid and known('session:' + sid):
            lines.append(f'- Source session: {_link(paths["session:" + sid])}')
        if r['kind'] == 'rejected' and r.get('key'):
            for d in by_key[(r.get('project'), r['key'])]:
                if d['kind'] == 'decision':
                    lines.append(f'- Rejects: {ref(d["id"])}')
        if r['kind'] == 'decision' and r.get('key'):
            for d in by_key[(r.get('project'), r['key'])]:
                if d['kind'] == 'rejected':
                    lines.append(f'- Rejected option: {ref(d["id"])}')
        for old in r.get('supersedes') or []:
            if known(old):
                lines.append(f'- Supersedes: {ref(old)}')
        for new in successor.get(r['id'], []):
            lines.append(f'- Superseded by: {ref(new)}')
        seen = set()
        for dst, rel in _edge_ids(r['id']):
            if dst and known(dst) and rel not in ('in_project', 'from_session', 'supersedes', 'rejects') and (dst, rel) not in seen and dst != r['id']:
                seen.add((dst, rel))
                lines.append(f'- {rel}: {ref(dst)}')
        mine_out = [e for e in out_edges.get(r['id'], [])]
        mine_in = in_edges.get(r['id'], [])
        if mine_out or mine_in:
            lines += ['', '## Links', ''] + [edge_line(e, e['dst']) for e in mine_out] + [f"- {e['type']} (incoming): {ref(e['src'])} — {e['why']}" for e in mine_in]
        meta = {'generated': True, 'kind': r['kind'], 'id': r['id'], 'title': titles[r['id']], 'status': r.get('status', ''), 'project': slug}
        out[paths[r['id']]] = render(meta, '\n'.join(lines + ['']))
    for sid, s in sorted(sessions.items()):
        slug = s['project']
        members = sorted(s['members'], key=lambda m: m['id'])
        lead = next((titles[m['id']] for m in members if m['id'] in titles), '')
        lines = [f'# Oturum {s["date"]} · {slug}' + (f': {lead}' if lead else ''), '', f'- Project: {_link(paths["project:" + slug])}']
        older = [x for x in proj_sessions[slug] if x < (s['date'], sid)]
        if older:
            lines.append(f'- Previous: {_link(paths["session:" + older[0][1]])}')
        lines += ['', '## Captured', '']
        lines += [f'- {ref(m["id"])}' + (f' ({_note_link(src_of[m["id"]])})' if m['id'] in src_of else '') if m['id'] in paths else f'- {m["kind"]}: {_note_link(src_of[m["id"]])}' if m['id'] in src_of else f'- Session note: {_link(m["source"])}' if str(m.get('source', '')).startswith('notes/sessions/') else f'- {m["kind"]}: {str(m.get("quote") or m["text"]).splitlines()[0][:120]}' for m in members] or ['- (none)']
        areas = sorted({f.rsplit('/', 1)[0] if '/' in f else '.' for f in s['files']})[:8]
        if areas:
            lines += ['', '## Touched areas', ''] + [f'- `{a}`' for a in areas]
        out[paths['session:' + sid]] = render({'generated': True, 'kind': 'session', 'id': sid, 'date': s['date'], 'project': slug}, '\n'.join(lines + ['']))
    ent_by_project = {}
    proj_of_doc = {d[0]: G.view_project(d[1]) for d in docs}
    for e in entities or []:
        for h in sorted({proj_of_doc[s] for s in e['sources'] if s in proj_of_doc}):
            ent_by_project.setdefault(h, []).append(e)
    for slug in projects:
        mine = [r for r in records if _project_slug(r) == slug]
        title = 'Project ' + slug + (' (home folder, unknown and claude sessions)' if slug == G.GENERIC else '')
        lines = [f'# {title}']
        for heading, kind in (('Decisions', 'decision'), ('Lessons', 'lesson'), ('Preferences', 'preference'), ('Rejected', 'rejected')):
            group = sorted([r for r in mine if r['kind'] == kind and r['id'] in paths], key=lambda r: (day_of(r), r['id']), reverse=True)
            lines += ['', f'## Latest {heading.lower()}', ''] + ([f'- {ref(r["id"])}' for r in group[:LATEST]] or ['- (none)'])
            if len(group) > LATEST:
                lines.append(f'- (+{len(group) - LATEST} older, reachable through sessions and concepts)')
        pc = sorted((proj_concepts.get(slug) or {}).items(), key=lambda kv: (-kv[1], kv[0]))
        if pc:
            lines += ['', '## Concepts used', ''] + [f'- {_link(paths[c])} ({n})' for c, n in pc if c in paths][:25]
            mates = {}
            for c, _ in pc:
                for p, n in (concept_projects.get(c) or {}).items():
                    if p != slug:
                        mates.setdefault(p, []).append(c)
            if mates:
                lines += ['', '## Related projects', ''] + [f'- {_link(paths["project:" + p])} — via ' + ', '.join(concepts[c]['name'] for c in cs[:4])
                                                            for p, cs in sorted(mates.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:8] if "project:" + p in paths]
        mine_cards = [c for c in cards if c['project'] == slug]
        loops = [c for c in mine_cards if c.get('open')]
        if loops:
            lines += ['', '## Open loops', ''] + [f'- {_link(c["source"])}' for c in loops]
        if mine_cards:
            lines += ['', '## Captured sources', ''] + [f'- {_link(c["source"])}' for c in sorted(mine_cards, key=lambda c: c['source'])[:40]]
            if len(mine_cards) > 40:
                lines.append(f'- (+{len(mine_cards) - 40} more, reachable through their concepts)')
        tasks = [r for r in mine if r.get('kind') == 'task' and r.get('status') in ('active', 'waiting')]
        lines += ['', '## Open tasks', ''] + ([f'- {str(r["text"]).strip().splitlines()[0][:120] if str(r["text"]).strip() else r["id"]}' for r in tasks] or ['- (none)'])
        lines += ['', '## Recent sessions', ''] + ([f'- {_link(paths["session:" + sid])}' for _, sid in proj_sessions.get(slug, [])[:RECENT]] or ['- (none)']) + ['']
        for gtitle, test in (('Lessons', 'notes/inferred/'), ('Captured notes', 'notes/captured/'), ('Session notes', 'notes/sessions/')):
            group = [r for r in owned if _project_slug(r) == slug and r['source'].startswith(test)]
            if group:
                lines += [f'## {gtitle}', ''] + [f'- {_note_link(r["source"])}' for r in sorted(group, key=lambda r: r['source'])] + ['']
        group = [r for r in users if user_hub[r['id']] == slug]
        if group:
            lines += ['## User notes', ''] + [f'- {_note_link(r["source"])}' for r in sorted(group, key=lambda r: r['source'])] + ['']
        if slug in ent_by_project:
            lines += ['## Entities', ''] + [f'- {_link(MEMROVE + "/entities/" + e["id"].split(":", 1)[1] + ".md")}' for e in sorted(ent_by_project[slug], key=lambda e: e['id'])] + ['']
        for kind in DOC_KINDS:
            group = [d for d in docs if G.view_project(d[1]) == slug and d[5] == kind]
            if group:
                lines += [f'## Docs: {kind}', ''] + [f'- {_link(doc_paths[d[0]])}' for d in group] + ['']
        out[paths['project:' + slug]] = render({'generated': True, 'kind': 'project', 'id': slug}, '\n'.join(lines))
    for slug in alias_slugs:
        out[f'{MEMROVE}/projects/{slug}.md'] = render({'generated': True, 'kind': 'alias', 'id': slug},
                                                    f'# {slug}\n\nOlder notes still point here; this name is not a project.\n\n- Project: {_link(paths["project:" + G.GENERIC])}\n')
    for cid, c in sorted(concepts.items()):
        if cid not in paths:
            continue
        lines = [f'# {c["name"]}', '', c.get('why') or '']
        if concept_slug_of(cid) in wiki_concepts:
            lines += ['', f'- Wiki: [[wiki/_kavramlar/{concept_slug_of(cid)}]]']
        cp = sorted((concept_projects.get(cid) or {}).items(), key=lambda kv: (-kv[1], kv[0]))
        if cp:
            lines += ['', '## Projeler', ''] + [f'- {_link(paths["project:" + p])} ({n})' for p, n in cp if 'project:' + p in paths]
        grouped = {}
        for e in back.get(cid, []):
            grouped.setdefault(node_project.get(e['src'], G.GENERIC), []).append(e)
        for p, es in sorted(grouped.items()):
            lines += ['', f'## {p}', ''] + [edge_line(e, e['src']) for e in sorted(es, key=lambda e: (titles.get(e['src'], ''), e['src']))]
        own = out_edges.get(cid, [])
        if own:
            lines += ['', '## Links', ''] + [edge_line(e, e['dst']) for e in own]
        out[paths[cid]] = render({'generated': True, 'kind': 'concept', 'id': cid, 'created_by': c.get('prov', {}).get('by', '')}, '\n'.join(lines) + '\n')
    cuts = [r for r in records if r.get('kind') == 'papercut' and r.get('status') == 'active']
    for slug in sorted({'harness' if r.get('harness') else _project_slug(r) for r in cuts}):
        mine = [r for r in cuts if ('harness' if r.get('harness') else _project_slug(r)) == slug]
        hub_slug = slug if 'project:' + slug in paths else G.GENERIC
        out[f'{MEMROVE}/papercuts/{slug}.md'] = render({'generated': True, 'kind': 'papercuts', 'id': 'papercuts:' + slug},
                                                     _papercuts.render(mine, slug) + (f'\n- Project: {_link(paths["project:" + hub_slug])}\n' if 'project:' + hub_slug in paths else ''))
    out.update(_doc_notes(docs, doc_edges, doc_paths, paths, render))
    out.update(_entity_notes(entities or [], docs, doc_paths, paths, render))
    lines = ['# Memrove index', '', '## Projects', ''] + ([f'- {_link(paths["project:" + p])}' for p in projects] or ['- (none)'])
    cl = [cid for cid in sorted(concepts) if cid in paths]
    if cl:
        lines += ['', '## Concepts', ''] + [f'- {G.link(paths[c], concepts[c]["name"])}' for c in cl]
    recent = sorted(((s['date'], sid) for sid, s in sessions.items()), reverse=True)[:RECENT]
    lines += ['', '## Recent sessions', ''] + ([f'- {_link(paths["session:" + sid])}' for _, sid in recent] or ['- (none)']) + ['']
    loose = sorted(r['source'] for r in users if not user_hub[r['id']])
    if loose:
        lines += ['## Other notes', ''] + [f'- {_note_link(x)}' for x in loose] + ['']
    out[f'{MEMROVE}/index.md'] = render({'generated': True, 'kind': 'index'}, '\n'.join(lines))
    nodes = {v: k for k, v in paths.items() if k.startswith('session:') or k.startswith('doc:') or k in {r['id'] for r in notes}}
    return out, nodes


def _verdict(text):
    head = text.split('\n---', 1)[0] if text.startswith('---') else ''
    m = re.search(r'''^\s*"?verdict"?\s*:\s*["']?(\w+)''', head, re.M)
    return m.group(1).lower() if m and m.group(1).lower() in VERDICTS else None


def ingest_verdicts(engine, db=None):
    """Only `verdict: keep|reject|pin` from generated notes reaches record status. Returns applied ids."""
    _hash, _atomic, _render = engine.projection_helpers()
    applied = []
    with engine.store._connect() as db:
        _views_schema(db)
        rows = db.execute("SELECT node,path,hash,seen FROM graph_views WHERE path LIKE ? AND path NOT LIKE ? AND path NOT LIKE ?",
                          (f'{MEMROVE}/%/%', f'{MEMROVE}/projects/%', f'{MEMROVE}/sessions/%')).fetchall()
        for node, rel, tracked, seen in rows:
            try:
                raw = engine._path(rel, existing=True).read_bytes()
            except (ValueError, OSError):
                continue
            current = _hash(raw)
            if current in (tracked, seen):
                continue
            verdict = _verdict(raw.decode('utf-8', 'replace'))
            db.execute('UPDATE graph_views SET seen=? WHERE path=?', (current, rel))
            if not verdict:
                continue
            row = db.execute('SELECT payload FROM records WHERE id=?', (node,)).fetchone()
            if not row:
                continue
            changes = {'status': 'rejected' if verdict == 'reject' else 'active', 'pinned': verdict == 'pin', 'verdict': verdict}
            if update_record(engine, node, changes, db):
                applied.append(node)
    return applied


OWNED = ('notes/captured/', 'notes/inferred/', 'notes/sessions/')  # memrove-owned notes: projections of store records
USER_NOTE_MARK = '## My notes (user_note)'
_NOT_FRONTMATTER = {'text', 'source', 'source_sha256', 'revision', 'user_note'}


def split_user_note(body, text):
    """The text the user added to a projected body: everything beyond the record text, minus the marker line."""
    body, text = body.strip(), text.strip()
    if body.startswith(text):
        extra = body[len(text):].strip()
    else:
        known = set(text.splitlines())
        extra = '\n'.join(l for l in body.splitlines() if l not in known).strip()
    if extra.startswith(USER_NOTE_MARK):
        extra = extra[len(USER_NOTE_MARK):].strip()
    return extra


def project_record(engine, record):
    """The one projection function: write a memrove-owned record's note (frontmatter + text + user_note) from the store payload.
    Returns the record with source_sha256 of the projected file. No-op when the source is not owned or the file is gone
    (a vanished file is a user delete; the next sync tombstones it, projection must not resurrect it)."""
    src = record.get('source') or ''
    if not src.startswith(OWNED):
        return record
    path = engine._path(src)
    if not path.is_file():
        return record
    _hash, _atomic, _render = engine.projection_helpers()
    body = record['text']
    if record.get('user_note'):
        body = body.rstrip('\n') + '\n\n' + USER_NOTE_MARK + '\n' + record['user_note'] + '\n'
    text = _render({k: v for k, v in record.items() if k not in _NOT_FRONTMATTER and v is not None and not (k == 'facts' and not v)}, body)
    _atomic(path, text)
    return dict(record, source_sha256=_hash(text))


def _kind_changed(record, new, by):
    try:
        import memrove_v3_decide_log as dl
        dl.kind_corrected(str(record.get("text") or record.get("title") or ""), record.get("id"), new, by)
    except Exception:
        pass


def update_record(engine, node, changes, db=None):
    """In-place payload update (revision+1, event logged). -> True when something changed.
    Memrove-owned notes are projections: the store change is re-written through project_record. Tombstones are never updated."""
    def apply(db):
        row = db.execute('SELECT payload FROM records WHERE id=?', (node,)).fetchone()
        if not row:
            return False
        record = json.loads(row[0])
        if record.get('status') == 'deleted_by_user' or all(record.get(k) == v for k, v in changes.items()):
            return False
        was = record.get('kind')
        record.update(changes)
        if 'kind' in changes and was != changes['kind']:
            _kind_changed(record, changes['kind'], 'edit')
        record['revision'] += 1
        record = project_record(engine, record)
        record = engine.store._validate(record)
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        db.execute('UPDATE records SET payload=? WHERE id=?', (payload, node))
        db.execute("INSERT INTO events(event_type,record_id,revision,record) VALUES ('update',?,?,?)", (node, record['revision'], payload))
        return True
    if db is not None:
        return apply(db)
    with engine.store._connect() as conn:
        return apply(conn)


def _views_schema(db):
    db.execute('CREATE TABLE IF NOT EXISTS graph_views(path TEXT PRIMARY KEY, node TEXT NOT NULL, hash TEXT NOT NULL, seen TEXT)')


def _is_hub(path):
    return path.endswith('index.md') or '/projects/' in path


def project_graph(engine, budget_s=0.15):
    """Incremental regeneration of <vault>/memrove/. Never blocks past budget_s: leftovers are deferred.

    Returns {'written','unchanged','deferred','conflicts','verdicts'}; None-engine writes nothing.
    """
    result = {'written': 0, 'unchanged': 0, 'deferred': 0, 'conflicts': [], 'verdicts': []}
    if engine is None or not getattr(engine, 'root', None):
        return result
    deadline = time.monotonic() + budget_s
    _hash, atomic, _render = engine.projection_helpers()  # render comes from sync (it imports this module)
    try:
        result['verdicts'] = ingest_verdicts(engine)
        with engine.store._connect() as db:
            db.execute('PRAGMA busy_timeout=100')
            _views_schema(db)
            records = [json.loads(r[0]) for r in db.execute('SELECT payload FROM records ORDER BY id')]
            tracked = {row[0]: row[1] for row in db.execute('SELECT path,hash FROM graph_views')}
            try:
                import memrove_v3_docs as _docs
                docs = (_docs.all_docs(db), _docs.all_edges(db))
            except Exception:
                docs = None
            try:
                import memrove_v3_entities as _ent
                ents = _ent.all_entities(db)
            except Exception:
                ents = []
            live = {r for r in (rec.get('source') for rec in records) if r and engine._path(r).is_file()}
            desired, ids = _build(records, _render, None, docs, ents, live, _graph_inputs(engine, db, records, live))
            order = sorted(desired, key=lambda p: (not _is_hub(p), p.endswith('index.md'), p))  # hubs first: owned notes link to them
            for relative in order:
                content, want = desired[relative], _hash(desired[relative])
                old_hash = tracked.get(relative)
                if old_hash == want:
                    result['unchanged'] += 1
                    continue
                if time.monotonic() > deadline and not _is_hub(relative):  # hubs are few and are link targets: never deferred
                    result['deferred'] += 1
                    continue
                path = engine._path(relative)
                disk = _hash(path.read_bytes()) if path.exists() else None
                if disk not in (None, want, old_hash):
                    result['conflicts'].append({'source': relative, 'reason': 'manual generated-note edit preserved'})
                    continue
                if disk != want:
                    atomic(path, content)
                    result['written'] += 1
                db.execute('INSERT INTO graph_views(path,node,hash,seen) VALUES (?,?,?,NULL) ON CONFLICT(path) DO UPDATE SET hash=excluded.hash',
                           (relative, ids.get(relative, relative), want))
            if not result['deferred'] and not result['conflicts']:
                result['retired'] = _retire_graph_views(engine, db, desired, _hash, tracked)
    except sqlite3.OperationalError:
        result['deferred'] += 1  # locked or busy: next hook run catches up
    return result


def _graph_inputs(engine, db, records, live):
    """What the store holds beyond records that the view renders: model/agent edges, grab cards, session days, hub names notes point at."""
    concepts, edges = _graph.load(db)
    cards = []
    for r in records:
        if _graph.is_card(r) and _graph.graph_visible(r) and r.get('source') in live:
            meta = _graph.card_meta(engine._path(r['source']))
            groups = meta.get('groups')
            cards.append({'id': r['id'], 'source': r['source'], 'project': _graph.view_project(meta.get('project') or meta.get('project')),
                          'title': _graph.first_sentence(meta.get('title')) or _graph.first_sentence(r.get('text')) or r['id'],
                          'groups': [str(g) for g in groups if str(g).strip()] if isinstance(groups, list) else [],
                          'open': meta.get('open') in (True, 'true') and not meta.get('closed')})
    days = {}
    try:  # a session without a dated record: its note's file name, else its transcript's last write
        smap = json.loads((Path(engine.state) / 'session-map.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        smap = {}
    for r in records:
        sid = r.get('session') or (r['id'] if r.get('kind') == 'session' else None)
        m = re.match(r'notes/sessions/(\d{4}-\d\d-\d\d)-', str(r.get('source') or ''))
        if sid and m:
            days.setdefault(sid, m.group(1))
    for sid, info in smap.items():
        try:
            days.setdefault(sid, datetime.fromtimestamp(Path(info['path']).stat().st_mtime).date().isoformat())
        except (OSError, KeyError, TypeError):
            pass
    targets = set()
    for r in records:
        if str(r.get('source') or '').startswith(OWNED) and r.get('source') in live:
            try:
                targets.update(re.findall(r'\[\[memrove/projects/([^\]|#]+)\]\]', engine._path(r['source']).read_text(encoding='utf-8')))
            except OSError:
                pass
    wiki = engine._path('wiki/_kavramlar')
    wiki_concepts = {f.stem for f in wiki.glob('*.md')} if wiki.is_dir() else set()
    return {'concepts': concepts, 'edges': edges, 'cards': cards, 'session_days': days, 'hub_targets': targets, 'wiki_concepts': wiki_concepts}


def _retire_graph_views(engine, db, desired, _hash, tracked):
    """Delete generated graph files that are no longer projected (renamed nodes, dropped junk hubs); an edited file is kept."""
    gone, kept = 0, []
    for relative, old_hash in tracked.items():
        if relative in desired:
            continue
        try:
            path = engine._path(relative)
            if path.is_file() and _hash(path.read_bytes()) != old_hash:
                kept.append(relative)  # edited by hand (or a stale hash): stays, reported
                continue
            path.unlink(missing_ok=True)
        except (ValueError, OSError):
            continue
        db.execute('DELETE FROM graph_views WHERE path=?', (relative,))
        gone += 1
    return {'removed': gone, 'kept': kept}
