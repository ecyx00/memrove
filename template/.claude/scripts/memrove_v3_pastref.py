"""Owner-prompt classifiers over the chat log (stdlib only).

1. Past-reference trigger: does the prompt point back at earlier work? If so (and MEMROVE_PASTREF=1) the prompt hook adds one line,
   no content: search before answering, then open the hits.
2. Re-brief detector (metric 0): owner prompts that ask for the same thing again in another session (BM25 over the chat log).
"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

LINE = "This prompt may build on earlier work: search first; a hit or a doc line is a lead, not evidence; open it with zoom or memrove doc and answer from what you opened."

# Turkish fold for past-reference detection (dotted/dotless i, diacritics).
_FOLD = str.maketrans({"İ": "i", "I": "ı", "ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c"})


def fold(text):
    return text.replace("İ", "i").replace("I", "ı").lower().translate(_FOLD)


# Rules run on folded text (Turkish diacritics dropped, so "geçen" and "gecen" both match).
RULES = [re.compile(p) for p in (
    r"\bdaha once\b.*\b(dedik|demistik|konustuk|yaptik|karar|bahsett|soyle|konus|yazdi|ettik|olusturdu)",
    r"\b(gecen|onceki|evvelki) (sefer|seans|oturum|session|konusma|hafta|gun|mesaj)",
    r"\bne (demistik|konusmustuk|kararlastirmistik|yapmistik|karar vermistik|yaptik|dedik)\b",
    r"\bneredeydik\b|\bneredeyiz\b|\bnerede kalmistik\b|\bnerede kaldik\b",
    r"\bhani (su|o|bir|bu|ya)\b",
    r"\b\w+[iu]yordu\b.*\bdegil mi\b",  # "X bunu yaziyordu degil mi?": asks to recall what an agent did
    r"\b(karar|konu)\w* (vermistik|almistik|etmistik|acmistik)\b",
    r"\b(konus|bahset|soyle|iste|paylas)\w*m[iu]st[iu]m\b|\baz once\b.*\b(paylas|soyle|bahset|yaz)|\b(session|oturum)\w* (bahset|konus)",
    r"\b(hatirla|hatirliyor mu)\w*\b.*\b(ne|neydi|nasildi|daha once|gecen)",
    r"\bneydi (o|su|bizim|karar)\b",
    r"\b\w+m[iu]st[iu]k\b",  # -mistik (we had done): past perfect, first person plural
    r"\b\w+(iyorduk|iyoduk|yorduk)\b|\bbiz\b.*\b\w+(tik|dik|duk|tuk) (ya|hani)\b",
    r"\bhatirliyorum\b|\bne durumdayiz\b|\bnoldu\b.*\?|\bnerede (o|onlar|bunlar)\b|\bnereye (gitti|kayboldu)\b",
    r"\b(neler|ne) konustugumuz|\bhakkinda yaptiklarimiz\b|\bonceki (session|oturum)",
    r"\bwhat (did|had) we (decide|say|agree|discuss|conclude|do|talk)",
    r"\b(last time|previously|earlier) (we|you|i)\b",
    r"\bwe (decided|discussed|agreed|talked about|said|settled)\b",
    r"\b(as|like) (we|i) (said|discussed|decided|agreed)\b",
    r"\bwhere (were we|did we leave)\b|\bwhere are we\b",
    r"\b(remember|recall) (when|what|the|how|that)\b",
    r"\bcontinue (from|where)\b.*\b(left|last|before|earlier)\b",
    # references to earlier work or sessions inside work orders
    r"\b(onceki|evvelki)\w*\b",
    r"\bdiger (\w+ )?\w*(session|oturum|chat|konusma)\w*",
    r"\b\w*(session|oturum)\w* (bahs|konus)\w*|\b(yapilan|yaptigimiz|yaptiklarimiz|yazdigimiz|egittigimiz|konustugumuz)\b.*\b(task|session|oturum|model|karar)",
    r"\b(task|session|oturum)\w* (ve|ya da) .*\b(bak|incele|oku)\w*|\b(tasklar|yapilan task)\w*",
    r"\bsen (yaptin|ettin|dedin|soyledin|yazdin|baktin)\b|\bsana (soyledim|demistim|verdim|gonderdim)\b",
    r"\b(paylastim|listeledim|anlattim|yazdirdim|istemistim|istemiistim)\b",
    # continue / resume an earlier task: the agent must recall the earlier work first
    r"\bdevam (et|edelim|edin|ediyoruz|edecegiz)\b|^\W*devam\W*$|\bdevam\w* (task|is|gorev)|\btask\w* devam|\bkaldigi\w* yer\w*",
    r"\b(implement|execute) the plan\b|\b(resume|pick up|carry on)\b.*\b(task|work|where)\b",
    r"\bcontinue (the |with )?(task|work)|\b(that|the) (task|model|work) (we|you)\b|\bwe (just )?(trained|built|made|wrote|did)\b",
)]


def is_past_reference(text):
    t = fold(text or "")[:1500]
    return any(r.search(t) for r in RULES)


def hint(text, env=None):
    """The one line to add to a UserPromptSubmit context, or '' (flag off, or no past reference)."""
    env = os.environ if env is None else env
    return LINE if env.get("MEMROVE_PASTREF") == "1" and is_past_reference(text) else ""


# ---- owner prompts + re-brief detector -----------------------------------------------------------------------------

SESSION_GAP = 3600  # seconds
MAX_OWNER = 1200  # longer "user" texts are pasted briefs or agent hand-offs, not the owner's own ask
TOKEN = re.compile(r"[a-z0-9_]{3,}")


def owner_prompts(state, since=None):
    """[(id, session, date, text)] of the owner's own prompts from the chat log (claude and codex user messages)."""
    from memrove_v3_chatlog import messages
    out, last, sess = [], {}, {}
    for r in messages(state):
        t = r.get("text") or ""
        # The log keeps no session id (`k` is one id per message), so a session is a run of one source's messages with gaps < SESSION_GAP.
        # ponytail: concurrent sessions of one source merge into one; store a session id at capture if that matters.
        src, ts = r.get("src"), datetime.fromisoformat(r["date"]).timestamp()
        if src not in last or ts - last[src] > SESSION_GAP:
            sess[src] = sess.get(src, 0) + 1
        last[src] = ts
        if r.get("probe"):  # a developer test run (sdk-cli / headless / cwd under /tmp), not the owner asking again
            continue
        if r.get("kind") != "user" or r.get("src") not in ("claude", "codex") or not 8 <= len(t) <= MAX_OWNER:
            continue
        if t.lstrip()[:1] in ("!", "<") or t.startswith(("claude ", "codex ")):  # shell lines, pasted harness blocks: not an ask
            continue
        if since and r["date"][:10] < since:
            continue
        out.append((r["i"], f"{src}:{sess[src]}", r["date"], t))
    return out


def tokens(text):
    return TOKEN.findall(fold(text))


def rebrief_pairs(prompts, min_score=0.5, min_tokens=6, ):
    """For each prompt, its best earlier prompt from a DIFFERENT session. Score = BM25 of the new prompt against the earlier one,
    divided by the earlier one's score against itself (so 1.0 = same words). Returns [(new_id, earlier_id, score)]."""
    docs = [(i, s, d, Counter(tokens(t))) for i, s, d, t in prompts]
    docs = [x for x in docs if sum(x[3].values()) >= min_tokens]
    n = len(docs)
    df = Counter(w for x in docs for w in x[3])
    idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}
    avg = sum(sum(x[3].values()) for x in docs) / max(n, 1)
    k1, b = 1.2, .75
    post = defaultdict(list)
    for j, x in enumerate(docs):
        ln = sum(x[3].values())
        for w, f in x[3].items():
            post[w].append((j, idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / avg))))
    pairs = []
    for j, (i, s, d, tf) in enumerate(docs):
        acc = defaultdict(float)
        for w in tf:
            for e, sc in post[w]:
                if e < j and docs[e][1] != s:
                    acc[e] += sc
        # normalise by the earlier doc's self-score and the new doc's, geometric mean (symmetric, bounded by ~1)
        cands = []
        for e, sc in acc.items():
            cands.append((sc / math.sqrt(_self(docs[j][3], idf, k1, b, avg) * _self(docs[e][3], idf, k1, b, avg)), e))
        if cands:
            sc, e = max(cands, key=lambda c: (c[0], -c[1]))  # ties: the earliest
            if sc >= min_score:
                pairs.append((i, docs[e][0], round(sc, 3)))
    return pairs


def _self(tf, idf, k1, b, avg):
    ln = sum(tf.values())
    return sum(idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / avg)) for w, f in tf.items()) or 1e-9


def weekly_rate(prompts, pairs):
    """{iso week: [prompts, re-briefs, rate]} over prompts long enough to be compared."""
    week = {i: datetime.fromisoformat(d).strftime("%G-W%V") for i, s, d, t in prompts}
    tot = Counter(week[i] for i, s, d, t in prompts if len(tokens(t)) >= 6)
    hit = Counter(week[p[0]] for p in pairs)
    return {w: [tot[w], hit[w], round(hit[w] / tot[w], 3)] for w in sorted(tot)}


def rebrief_command(state, out="/tmp/e71-out/rebrief-pairs.jsonl", min_score=0.5, since=None):
    prompts = owner_prompts(state, since)
    pairs = rebrief_pairs(prompts, min_score)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for new, old, sc in pairs:
            f.write(json.dumps({"new": new, "earlier": old, "score": sc}) + "\n")
    return json.dumps({"prompts": len(prompts), "pairs": len(pairs), "pairs_file": out,
                       "weekly": weekly_rate(prompts, pairs)}, indent=2)
