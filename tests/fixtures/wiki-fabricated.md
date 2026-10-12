---
project: indexer
root: /srv/dev/Desktop/indexer
roots: {"indexer": "/srv/dev/Desktop/indexer"}
title: Fabricated topic
group: Concepts
updated: 2026-10-10
---
# Fabricated topic
Hybrid search merges two rankings.

Analogy: two juries give one verdict.

Spine: query → BM25 → embedding → RRF

## Problem
- A single signal misses. [general]
- A record claim that was never actually measured. [memrove:obs-ffffffffffffffffffffffff]
- An attempt to hide a number in backticks: `1024` times faster. [general]
## Skeleton
- Two rankers and one merger. [general]
## Flow
- The query goes to two rankers. [general]
- Breaks: one returns empty → the list is left with one → the result weakens. [general]
## Solution and cost
- Break → merge → extra latency. [general]
## Key distinctions
- Difference between BM25 and embeddings: one is words, the other meaning. [general]
## What we researched
### Uydurma kaynak (repo)
- A line that is not in the snapshot. [src:semble:nope.py:3]
## Ours
- Uydurma commit. [commit:deadbeef1234]
Our measurement:
| measurement | source | what theory says |
|---|---|---|
| gecikme | yok [general] | x |
## Where you see it
- [f](code:hm/yok.py#yok)
## Subtopics to know
- [[missing-page]] — neighbouring topic. [general]
## Compression
- One sentence. [general]
<details><summary>Test yourself</summary>
1. soru?
</details>
<details><summary>Layer 4</summary>
- 40% [doc:docs/state/yok.md]
</details>
