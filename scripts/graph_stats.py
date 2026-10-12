#!/usr/bin/env python3
"""Link statistics of a vault view (read-only): python3 graph_stats.py [vault] [--json]."""
import collections, json, re, sys
from pathlib import Path

LINK = re.compile(r'\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]')
CODE = re.compile(r'```.*?```|`[^`\n]*`', re.S)


def scan(vault):
    vault = Path(vault).expanduser()
    files = [p for p in vault.rglob('*.md') if not any(x.startswith('.') for x in p.relative_to(vault).parts)]
    rel = {p.relative_to(vault).with_suffix('').as_posix(): p for p in files}
    byname = collections.defaultdict(list)
    for r in rel:
        byname[r.rsplit('/', 1)[-1].lower()].append(r)
    low = {r.lower(): r for r in rel}
    inb, out, broken = collections.Counter(), collections.Counter(), []
    scan.edges = []
    for r, p in rel.items():
        text = CODE.sub('', p.read_text(encoding='utf-8', errors='replace'))
        for t in LINK.findall(text):
            t = t.strip().removesuffix('.md')
            dst = low.get(t.lower()) or (byname[t.lower()][0] if len(byname.get(t.lower(), [])) == 1 else None)
            if dst is None:
                broken.append((r, t))
                continue
            if dst != r:
                inb[dst] += 1; out[r] += 1; scan.edges.append((r, dst))
    return rel, inb, out, broken


def stats(vault):
    rel, inb, out, broken = scan(vault)
    edges = scan.edges
    nodes = [r for r in rel]
    top = [(r, n) for r, n in inb.most_common(8)]
    to_index = sum(n for r, n in inb.items() if r == 'memrove/index')
    node_links_to_index = to_index
    orphan = [r for r in rel if not inb[r]]
    def under(prefix): return [r for r in rel if r.startswith(prefix)]
    rec_prefixes = ('memrove/decisions/', 'memrove/preferences/', 'memrove/rejected/', 'memrove/lessons/', 'notes/inferred/', 'notes/captured/', 'notes/observed/', 'wiki/_sources/grab/')
    rec_nodes = [r for r in rel if r.startswith(rec_prefixes)]
    rec_orph = [r for r in rec_nodes if not inb[r] and not out[r]]
    peers = ('memrove/decisions/', 'memrove/preferences/', 'memrove/rejected/', 'memrove/lessons/', 'memrove/concepts/', 'wiki/_sources/grab/')
    # unbridged: no link to or from a concept or another record/card (only hubs, sessions and its own source note)
    nb = collections.defaultdict(set)
    for a, b in edges:
        nb[a].add(b); nb[b].add(a)
    peer = lambda r, x: x != r and x.startswith(peers)
    graph_nodes = [r for r in rec_nodes if r.startswith(('memrove/', 'wiki/_sources/grab/'))]  # source notes under notes/ only carry the record's text
    unbridged = [r for r in graph_nodes if not any(peer(r, x) for x in nb[r])]
    return {
        'notes': len(rel), 'links': sum(inb.values()), 'broken': len(broken), 'orphans': len(orphan),
        'orphans_notes': len([r for r in orphan if r.startswith('notes/')]),
        'links_to_index': node_links_to_index,
        'zero_date_names': len([r for r in rel if '0000-00-00' in r]),
        'obs_only_names': len([r for r in rel if re.fullmatch(r'obs-[0-9a-f]+', r.rsplit('/', 1)[-1])]),
        'record_nodes': len(rec_nodes), 'record_orphans': len(rec_orph),
        'record_orphan_pct': round(100 * len(rec_orph) / max(1, len(rec_nodes)), 1), 'record_unbridged': len(unbridged), 'record_unbridged_pct': round(100 * len(unbridged) / max(1, len(graph_nodes)), 1),
        'top_targets': top, 'broken_sample': broken[:5],
        'junk_hubs': [r for r in rel if r in ('memrove/projects/unknown', 'memrove/projects/claude')],
    }


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    print(json.dumps(stats(args[0] if args else '~/Vault'), ensure_ascii=False, indent=1))
