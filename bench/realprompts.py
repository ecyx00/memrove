#!/usr/bin/env python3
"""Real-prompt injection rate of the auto-recall chat lane. Reads a COPY of a state dir (path argument) and the Claude transcripts; prints counts only,
never message text. For each of the last N owner prompts: the log as it was then (ids below the prompt), the prompt's own transcript as the in-context set.

  python3 -I bench/realprompts.py /tmp/lme/livecopy [--n 200] [--projects ~/.claude/projects]
"""
import argparse
import itertools
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "template/.claude/scripts"))
sys.dont_write_bytecode = True
import memrove_v3_chatlog as chatlog  # noqa: E402
import memrove_v3_chatsearch as cs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dfs", default="36", help="rare-word document-frequency caps (messages) to sweep")
    ap.add_argument("--projects", default=str(Path.home() / ".claude/projects"))
    a = ap.parse_args(argv)
    top = chatlog.count(a.state)
    prompts = [r for r in chatlog.messages(a.state, max(0, top - 4000)) if r["kind"] == "user" and str(r.get("k", "")).startswith("c:") and "#" not in r["k"]][-a.n:]
    files = sorted(Path(a.projects).glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:400]
    where = {}
    for f in files:
        if len(where) == len(prompts):
            break
        data = f.read_text(errors="replace")
        for r in prompts:
            if r["i"] not in where and r["k"][2:] in data:
                where[r["i"]] = str(f)
    print(f"prompts {len(prompts)}, transcript found for {len(where)}", file=sys.stderr)
    prompts = [r for r in prompts if r["i"] in where]
    short = [r for r in prompts if len(r["text"].split()) <= 4]
    print(f"n={len(prompts)} short(<=4 words)={len(short)}")
    print("| rare df | min rare | min words | min strength | injected % | short injected | lines/prompt (injected) | tok med | session hits | within 2 msgs |\n|---|---|---|---|---|---|---|---|---|---|")
    data, kc = {}, {}
    cache = {}   # frac -> prompt id -> candidates
    for frac in tuple(int(x) for x in a.dfs.split(",")):
        cs.LANE_RARE_DF = frac
        cache[frac] = {r["i"]: cs.lane_candidates(a.state, r["text"][:500], where[r["i"]], upto=r["i"]) for r in prompts}
    for frac, mr, mw, ms in itertools.product(tuple(int(x) for x in a.dfs.split(",")), (1, 2, 3), (3,), (14, 18, 22, 26, 30)):
        inj = shortn = sess = near = 0
        toks, nl = [], []
        for r in prompts:
            if len({w for w in cs.WORD.findall(cs.fold(r["text"])) if len(w) > 3}) < mw:
                continue
            lines = cs.lane_pack(cache[frac][r["i"]], r["text"], min_strength=ms, min_rare=mr, budget=cs.LANE_TOKENS - cs._tokens(cs.LANE_HEADER + "\n"))
            if not lines:
                continue
            inj += 1
            shortn += r in short
            nl.append(len(lines))
            toks.append(cs._tokens("\n".join(l for _, l in lines)))
            if where[r["i"]] not in data:
                data[where[r["i"]]] = Path(where[r["i"]]).read_text(errors="replace")
            t = data[where[r["i"]]]
            for i, _ in lines:
                if i not in kc:
                    kc[i] = str(((chatlog.get(a.state, i) if isinstance(i, int) else None) or {}).get("k", "")).split("#")[0]
                k = kc[i]
                sess += k.startswith("c:") and k[2:] in t
                near += isinstance(i, int) and abs(i - r["i"]) <= 2
        n = len(prompts)
        print(f"| {frac} | {mr} | {mw} | {ms} | {100 * inj / n:.1f} | {shortn}/{len(short)} | {statistics.mean(nl) if nl else 0:.1f} | {statistics.median(toks) if toks else 0:.0f} | {sess} | {near} |")


if __name__ == "__main__":
    main()
