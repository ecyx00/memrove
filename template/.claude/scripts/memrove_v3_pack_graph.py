#!/usr/bin/env python3
"""Graph arm of the task-start pack: spreading activation (personalized PageRank, two hops) over graph_edges.

graph_pack(state, project, prompt, as_of, exclude_session, cap) -> text block, '' when nothing stands out.
Pure SQL + Python, read-only, no model calls. Nodes created after as_of and anything from exclude_session are invisible."""
import re
import sqlite3
from pathlib import Path

KINDS = ('decision', 'rejected', 'lesson', 'preference', 'papercut')
# typed edge weights: supersedes, contradicts (and anything touching a rejected record) carry the most
W = {'supersedes': 1.0, 'contradicts': 1.0, 'answers': 0.6, 'relates': 0.5, 'part_of': 0.4}
DAMP = 0.6
FLAT = 1.5  # abstain unless the top activation is this many times the mean
STOP = set('one out its our them see the and for with that this from into have are was but not you your how why what when use using nin ile bir için ama gibi daha çok değil mi mı'.split())


def _stem(w):
    return w[:5]


def _terms(text):
    return {_stem(w): w for w in re.findall(r'\w{4,}', (text or '').lower()) if w not in STOP}


def _day(ts):
    return (ts or '')[:10]


def _visible(created, as_of):
    # date-only stamps count as the whole day: strictly before as_of's day, so a same-day correction cannot leak
    if not created:
        return False
    return created < as_of if 'T' in created else created < _day(as_of)


def _load(state, project, as_of, exclude):
    db = sqlite3.connect(f'file:{Path(state) / "memory.sqlite3"}?mode=ro', uri=True)
    q = ','.join('?' * len(KINDS))
    nodes = {}
    for r in db.execute(
        f"SELECT id, json_extract(payload,'$.kind'), json_extract(payload,'$.project'), json_extract(payload,'$.session'),"
        f" coalesce(json_extract(payload,'$.recorded_at'), json_extract(payload,'$.valid_from')),"
        f" json_extract(payload,'$.text'), json_extract(payload,'$.key'), json_extract(payload,'$.seen_sessions')"
        f" FROM records WHERE json_extract(payload,'$.status')='active' AND json_extract(payload,'$.kind') IN ({q})", KINDS):
        rid, kind, proj, sess, created, text, key, seen = r
        if not _visible(created, as_of) or sess == exclude or (exclude and exclude in (seen or '')):
            continue
        nodes[rid] = {'id': rid, 'kind': kind, 'project': proj, 'created': created or '', 'text': (text or '').strip(), 'key': key or ''}
    for r in db.execute('SELECT id, name, ts FROM graph_nodes'):
        if _visible(r[2], as_of):
            nodes[r[0]] = {'id': r[0], 'kind': 'concept', 'project': None, 'created': r[2], 'text': r[1], 'key': r[1]}
    edges = [e for e in db.execute('SELECT src, dst, type, ts FROM graph_edges')
             if e[0] in nodes and e[1] in nodes and _visible(e[3], as_of)]
    db.close()
    return nodes, edges


def graph_pack(state, project, prompt, as_of, exclude_session, cap=1500):
    nodes, edges = _load(state, project, as_of, exclude_session)
    adj = {}
    for s, d, t, _ in edges:
        w = W.get(t, 0.3)
        for a, b in ((s, d), (d, s)):
            ww = w * (1.5 if nodes[a]['kind'] == 'rejected' or nodes[b]['kind'] == 'rejected' else 1.0)
            adj.setdefault(a, []).append((b, ww, t))
    # successor of a supersedes pair = the newer record, regardless of how the edge was written
    old = set()
    for s, d, t, _ in edges:
        if t == 'supersedes':
            old.add(s if nodes[s]['created'] < nodes[d]['created'] else d)

    # seeds: prompt-term matches (idf-weighted), the project's topic heads (concepts its records belong to), its latest records
    pw = _terms(prompt)
    pt = set(pw)
    toks = {i: set(_terms(f"{n['text']} {n['key']}")) for i, n in nodes.items()}
    df = {}
    for ts in toks.values():
        for x in ts & pt:
            df[x] = df.get(x, 0) + 1
    pt = {x for x in pt if df.get(x, 0) <= max(8, len(nodes) // 20)}  # terms in a twentieth of memory say nothing
    seed, why = {}, {}
    for i, ts in toks.items():
        hit = ts & pt
        if hit:
            seed[i] = sum(1 / df[x] ** 0.5 for x in hit)
            why[i] = 'prompt term ' + pw[min(hit, key=df.get)]
    tot = sum(seed.values()) or 1.0
    seed = {i: v / tot for i, v in seed.items()}
    mine = sorted((n for n in nodes.values() if n['project'] == project), key=lambda n: n['created'], reverse=True)
    for n in mine[:12]:
        seed[n['id']] = seed.get(n['id'], 0) + 0.04
        why.setdefault(n['id'], 'recent in project')
    heads = {}
    for s, d, t, _ in edges:
        if t == 'part_of' and nodes[d]['kind'] == 'concept' and nodes[s]['project'] == project:
            heads[d] = heads.get(d, 0) + 1
    for c, k in sorted(heads.items(), key=lambda x: -x[1])[:5]:
        seed[c] = seed.get(c, 0) + 0.05
        why.setdefault(c, 'project topic')

    act, path = dict(seed), {i: why[i] for i in seed}
    front = seed
    for _ in range(2):
        nxt, via = {}, {}
        for n, a in front.items():
            deg = sum(w for _, w, _ in adj.get(n, ()))
            for m, w, t in adj.get(n, ()):
                v = DAMP * a * w / deg
                nxt[m] = nxt.get(m, 0) + v
                if v > via.get(m, (0,))[0]:
                    via[m] = (v, n, t)
        for m, (_, n, t) in via.items():
            if m not in path or (m not in seed and m not in act):
                path[m] = f'{path[n]} > {t} > {_label(nodes[n])}'
        for m, v in nxt.items():
            act[m] = act.get(m, 0) + v
        front = nxt

    score = {i: v * (0.25 if i in old else 1.0) for i, v in act.items() if nodes[i]['kind'] != 'concept'}
    if len(score) < 2:
        return ''
    mean = sum(score.values()) / len(score)
    if max(score.values()) < FLAT * mean:
        return ''

    def pick(pred, k):
        c = sorted((i for i in score if pred(nodes[i])), key=lambda i: (-score[i], nodes[i]['created']))
        return c[:k]
    ours = lambda n: n['project'] == project
    ids = (pick(lambda n: ours(n) and n['kind'] == 'decision', 3) + pick(lambda n: ours(n) and n['kind'] == 'rejected', 2)
           + pick(lambda n: ours(n) and n['kind'] == 'papercut', 2)
           + [i for i in pick(lambda n: not ours(n) and n['kind'] in ('decision', 'rejected', 'lesson') and n['project'], 20) if '>' in path[i]][:2])
    out, used = [], 0
    for i in ids:
        n = nodes[i]
        lab = 'open' if n['kind'] == 'papercut' else n['kind']
        if not ours(n):
            lab = f"{n['project']} {lab}"
        line = f"- {lab}: {_short(n['text'], 140)} (why: {path[i]})"
        if used + len(line) + 1 > cap:
            break
        out.append(line)
        used += len(line) + 1
    return '\n'.join(out)


def _short(t, n):
    t = ' '.join(t.split())
    return t if len(t) <= n else t[:n - 1].rstrip() + '…'


def _label(n):
    return n['key'] if n['kind'] == 'concept' else _short(n['text'], 40)
