#!/usr/bin/env python3
"""End-to-end answer accuracy: retrieve top 10 -> Haiku reads -> Haiku judges. Public benchmark text only.

  python3 bench/answer.py run --set lme --arm memrove-chatlog|bm25 [--limit N] [--jobs 4]
  python3 bench/answer.py run --set locomo --arm memrove-chatlog            # conversations 5-9
  python3 bench/answer.py report

Every reader and judge output is cached in /tmp/lme/answer/<set>-<arm>/<qid>.json, so a crash resumes without new calls.
Reader and judge: `claude -p --model haiku` from an empty cwd with no tools and no user/project settings (no CLAUDE.md reaches the call). One reader prompt for every arm.
Judge prompts: LongMemEval src/evaluation/evaluate_qa.py get_anscheck_prompt (copied verbatim; only the model changed), also used for LoCoMo category 5.
LoCoMo categories 1-4 use the Mem0 repo's ACCURACY_PROMPT (evaluation/metrics/llm_judge.py, copied verbatim; judge model Haiku instead of gpt-4o-mini).
LoCoMo category 5 (Mem0 skips it) uses the LongMemEval abstention prompt.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lme  # noqa: E402

REAL_HOME = os.environ["HOME"]   # lme.setup() repoints HOME to a temp dir; claude needs the real one for its login
CACHE = Path("/tmp/lme/answer")
LME_DATA = "/tmp/lme/data/longmemeval_s_cleaned.json"
LOCO_DATA = "/tmp/lme/locomo/locomo10.json"
K = 10
LOCO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "unanswerable"}


def get_anscheck_prompt(task, question, answer, response, abstention=False):   # verbatim from LongMemEval evaluate_qa.py
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


LOCO_JUDGE = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {q}
Gold answer: {a}
Generated answer: {r}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

READER = """You answer a question using snippets retrieved from past conversations (the memory). Each snippet has its date and speaker.
{now}
Snippets:
{hits}

Question: {q}

Answer the question concisely using only the snippets. If the snippets do not contain the information needed to answer, answer exactly: I don't know"""


def haiku(prompt):
    """One `claude -p --model haiku` call, retried; -> (text, model). Empty cwd, no tools, no user/project settings, no session saved."""
    cwd = CACHE / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    last = ""
    for n in range(5):
        time.sleep(3 * n)   # empty 0-token replies were transient; back off
        p =subprocess.run(["claude", "-p", "--model", "haiku", "--tools", "", "--setting-sources", "project", "--disable-slash-commands", "--no-session-persistence",
                            "--system-prompt", "You are a careful assistant.", "--output-format", "json"], input=prompt, capture_output=True, text=True, cwd=cwd, env={**os.environ, "HOME": REAL_HOME})
        try:
            j = json.loads(p.stdout)
            if j.get("result", "").strip() and not j.get("is_error"):
                return j["result"].strip(), next(iter(j.get("modelUsage", {"?": 0})))
            last = p.stdout[:300]
        except ValueError:
            last = (p.stdout + p.stderr)[:300]
    raise RuntimeError("haiku failed on %r: %s" % (prompt[-200:], last))


def idk(text):
    return "don't know" in text.lower().replace("’", "'") or "do not know" in text.lower()


def judge_prompt(rec):
    if rec["set"] == "lme":
        return get_anscheck_prompt(rec["type"], rec["question"], rec["gold"], rec["reader"], abstention=rec["abs"])
    if rec["abs"]:
        return get_anscheck_prompt("", rec["question"], "The information needed is not available in the conversation.", rec["reader"], abstention=True)
    return LOCO_JUDGE.format(q=rec["question"], a=rec["gold"], r=rec["reader"])


def verdict(rec, text):
    if rec["set"] == "lme" or rec["abs"]:
        return "yes" in text.lower()   # as evaluate_qa.py
    labels = re.findall(r"\b(CORRECT|WRONG)\b", text.upper())
    return bool(labels) and labels[-1] == "CORRECT"


def answer_one(path, rec):
    """Fill reader and judge into the cache file; each step is cached, so a crash resumes mid-question."""
    if "reader" not in rec:
        rec["reader"], rec["model"] = haiku(READER.format(now=f"Current date: {rec['qdate']}\n" if rec["qdate"] else "", hits="\n".join(rec["hits"]), q=rec["question"]))
        path.write_text(json.dumps(rec))
    if "judge" not in rec:
        rec["judge"], _ = haiku(judge_prompt(rec))
        rec["correct"] = verdict(rec, rec["judge"])
        path.write_text(json.dumps(rec))


def fmt_hit(i, date, role, text):
    return f"[{i}] ({date or 'date unknown'}) {role}: {text}"


def lme_records(arm, limit):
    work = lme.setup("/tmp/lme/work/ans")
    import memrove_v3_chatlog as chatlog
    import memrove_v3_chatsearch as cs
    for it in json.load(open(LME_DATA))[:limit or None]:
        rec = {"set": "lme", "arm": arm, "qid": it["question_id"], "type": it["question_type"], "abs": it["question_id"].endswith("_abs"), "question": it["question"],
               "gold": it["answer"], "qdate": it["question_date"]}
        path = CACHE / f"lme-{arm}" / f"{rec['qid']}.json"
        if path.exists():
            yield path, json.loads(path.read_text())
            continue
        meta = [(d, t["role"]) for d, s in zip(it["haystack_dates"], it["haystack_sessions"]) for t in s]
        if arm == "bm25":   # bm25-turn of bench/lme.py: user turns only
            turns = [(d, t["content"]) for d, s in zip(it["haystack_dates"], it["haystack_sessions"]) for t in s if t["role"] == "user"]
            order = lme.bm25_order([t[1].split(" ") for t in turns], it["question"].split(" "))[:K]
            hits = [fmt_hit(n, turns[i][0], "user", turns[i][1]) for n, i in enumerate(order, 1)]
        else:
            items = [{"kind": "user" if t["role"] == "user" else "agent", "text": t["content"], "date": d} for d, s in zip(it["haystack_dates"], it["haystack_sessions"]) for t in s]
            with tempfile.TemporaryDirectory(dir=work, prefix="state-") as state:
                chatlog.append(state, items, today="2023-01-01")
                rows = [r for r in cs.merged(state, it["question"], top=lme.TOP, ref=lme.qdate(it)) if not str(r[0]).startswith("rec:")][:K]
            hits = [fmt_hit(n, meta[r[0]][0], meta[r[0]][1], r[3]) for n, r in enumerate(rows, 1)]
        path.parent.mkdir(parents=True, exist_ok=True)
        rec["hits"] = hits
        path.write_text(json.dumps(rec))
        yield path, rec


def locomo_records(arm, limit):
    if arm != "memrove-chatlog":  # only the memrove arm is wired for LoCoMo; any other label would mislabel a memrove run
        sys.exit(f"--set locomo supports only --arm memrove-chatlog, got {arm}")
    work = lme.setup("/tmp/lme/work/ans")
    import memrove_v3_chatlog as chatlog
    import memrove_v3_chatsearch as cs
    for ci, conv in list(enumerate(json.load(open(LOCO_DATA))))[5:10]:
        c, ts, n = conv["conversation"], [], 1
        while f"session_{n}" in c:
            ts += [(t["speaker"], t["text"], c.get(f"session_{n}_date_time", "")) for t in c[f"session_{n}"]]
            n += 1
        state = None
        for qi, q in enumerate(conv["qa"][:limit or None]):
            rec = {"set": "locomo", "arm": arm, "qid": f"{conv['sample_id']}-{qi}", "type": LOCO_CATS[q["category"]], "abs": q["category"] == 5, "question": q["question"],
                   "gold": str(q.get("answer", "")), "qdate": ""}
            path = CACHE / f"locomo-{arm}" / f"{rec['qid']}.json"
            if path.exists():
                yield path, json.loads(path.read_text())
                continue
            if state is None:
                state = tempfile.mkdtemp(dir=work, prefix="state-")
                first = ts[0][0]
                chatlog.append(state, [{"kind": "user" if t[0] == first else "agent", "text": t[1], "date": ""} for t in ts], today="2023-01-01")   # as bench/mem.py
            rows = cs._scored(state, q["question"], K, None, None, None)
            rec["hits"] = [fmt_hit(i, ts[r[0]][2], ts[r[0]][0], r[3]) for i, (sc, r) in enumerate(rows, 1)]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec))
            yield path, rec


def run(a):
    gen = (lme_records if a.set == "lme" else locomo_records)(a.arm, a.limit)
    with ThreadPoolExecutor(min(a.jobs, 4)) as ex:   # owner cap: 4 concurrent claude -p
        futs = []
        for path, rec in gen:   # retrieval happens in this thread while workers call Haiku
            futs.append(ex.submit(answer_one, path, rec))
        for f in futs:
            f.result()
    print(f"done {len(futs)} questions", file=sys.stderr)


def pct(xs):
    return f"{100 * sum(xs) / len(xs):5.1f} ({sum(xs)}/{len(xs)})" if xs else "   - "


def report():
    ok = True
    print("| set | arm | group | accuracy % (correct/n) |\n|---|---|---|---|")
    calls = 0
    for d in sorted(CACHE.glob("*-*")):
        if not d.is_dir():
            continue
        recs = [json.loads(p.read_text()) for p in d.glob("*.json")]
        calls += sum(("reader" in r) + ("judge" in r) for r in recs)
        recs = [r for r in recs if "correct" in r]
        if not recs:
            continue
        s, arm = recs[0]["set"], recs[0]["arm"]
        for t in sorted({r["type"] for r in recs}):
            print(f"| {s} | {arm} | {t} | {pct([r['correct'] for r in recs if r['type'] == t and not r['abs']])} |")
        ans = [r for r in recs if not r["abs"]]
        print(f"| {s} | {arm} | **overall answerable** | {pct([r['correct'] for r in ans])} |")
        print(f"| {s} | {arm} | abstention (unanswerable) | {pct([r['correct'] for r in recs if r['abs']])} |")
        print(f"| {s} | {arm} | **overall all questions** | {pct([r['correct'] for r in recs])} |")
        print(f"| {s} | {arm} | reader says I don't know, answerable | {pct([idk(r['reader']) for r in ans])} |")
        print(f"| {s} | {arm} | reader says I don't know, unanswerable | {pct([idk(r['reader']) for r in recs if r['abs']])} |")
    print(f"\nHaiku calls in cache: {calls}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=("lme", "locomo"), required=True)
    r.add_argument("--arm", choices=("memrove-chatlog", "bm25"), required=True)
    r.add_argument("--limit", type=int, default=0, help="first N questions (per conversation for locomo)")
    r.add_argument("--jobs", type=int, default=4)
    sub.add_parser("report")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report()
