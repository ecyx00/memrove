"""Build and label the past-reference set. Owner text goes only to `claude -p` (haiku) and /tmp/e71-out.
usage: pastref_label.py STATE sample|label|eval"""
import json, random, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "template/.claude/scripts"))
import memrove_v3_pastref as pr

OUT = Path("/tmp/e71-out")
# Stems mix Turkish and English (onc/gecen = before/last time, karar = decision, dedi = said, konus = talk, bahset = mention, nerede = where, hatirl = remember).
LOOSE = re.compile(r"onc|gecen|hani|karar|dedi|konus|bahset|last|earlier|decid|remember|nerede|hatirl|yapmist|soylem|agree", re.I)
PROMPT = ("Label developer messages typed to a coding agent. 1 = the message EXPLICITLY points back at an earlier session, conversation or "
          "decision and asks to recall, continue or rely on it (e.g. 'what did we decide', 'last time', 'where are we', 'as we said', "
          "'that thing we discussed', 'hani şu ...', 'geçen sefer', 'daha önce konuştuk'). 0 = everything else, including ordinary "
          "follow-ups, complaints, tasks, or messages that merely need project context. Be strict. Reply with only a JSON list of "
          "the numbers of the messages labelled 1 (e.g. [3, 17]; [] if none), no other text.\n\n")


def sample(state):
    rng = random.Random(71)
    ps = [p for p in pr.owner_prompts(state) if len(p[3]) <= 600]
    loose = [p for p in ps if LOOSE.search(pr.fold(p[3]))]
    rest = [p for p in ps if p not in loose]
    pick = rng.sample(rest, 200) + rng.sample(loose, min(100, len(loose)))
    rng.shuffle(pick)
    with open(OUT / "pastref-sample.jsonl", "w") as f:
        for i, s, d, t in pick:
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    print(len(pick), "sampled; loose pool", len(loose), "of", len(ps))


def label():
    rows = [json.loads(l) for l in open(OUT / "pastref-sample.jsonl")]
    done = {json.loads(l)["id"] for l in open(OUT / "pastref-labels.jsonl")} if (OUT / "pastref-labels.jsonl").exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    for n in range(0, len(rows), 100):
        b = rows[n:n + 100]
        body = PROMPT + "\n".join(f"{k + 1}. {r['text'][:160]!r}" for k, r in enumerate(b))
        p = subprocess.run(["nice", "claude", "-p", "--model", "haiku", "--output-format", "json", "--tools", "", "--system-prompt",
                            "You are a strict labeler.", "--setting-sources", "project", "--disable-slash-commands", "--strict-mcp-config", body],
                           capture_output=True, text=True, cwd="/tmp", env={**__import__("os").environ, "MAX_THINKING_TOKENS": "0"})
        d = json.loads(p.stdout)
        u = d["usage"]
        with open(OUT / "tokens.log", "a") as f:
            f.write(json.dumps({"task": "pastref-label", "in": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"], "out": u["output_tokens"]}) + "\n")
        pos = set(json.loads(re.search(r"\[[\d,\s]*\]", d["result"]).group(0)))
        with open(OUT / "pastref-labels.jsonl", "a") as f:
            for k, r in enumerate(b):
                f.write(json.dumps({"id": r["id"], "label": int(k + 1 in pos)}) + "\n")


def evaluate():
    rows = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(OUT / "pastref-sample.jsonl")}
    lab = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(OUT / "pastref-labels.jsonl")}
    ids = sorted(lab)
    tune, held = ids[0::2], ids[1::2]  # fixed split by sorted id parity; rules are tuned on `tune` only
    for name, part in (("tune", tune), ("heldout", held)):
        tp = sum(lab[i] and pr.is_past_reference(rows[i]) for i in part)
        fp = sum(not lab[i] and pr.is_past_reference(rows[i]) for i in part)
        fn = sum(lab[i] and not pr.is_past_reference(rows[i]) for i in part)
        print(name, "n", len(part), "pos", tp + fn, "tp", tp, "fp", fp, "fn", fn,
              "precision", round(tp / (tp + fp), 3) if tp + fp else None, "recall", round(tp / (tp + fn), 3) if tp + fn else None)
    if len(sys.argv) > 3:
        for i in tune:
            m = pr.is_past_reference(rows[i])
            if m != bool(lab[i]):
                print("FP" if m else "FN", repr(rows[i][:160]))


if __name__ == "__main__":
    {"sample": lambda: sample(sys.argv[1]), "label": label, "eval": evaluate}[sys.argv[2]]()
