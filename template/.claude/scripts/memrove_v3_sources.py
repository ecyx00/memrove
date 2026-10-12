"""Source snapshots: fetch a pinned copy of a repo, paper or model (data only, never executed) under <vault>/wiki/_sources/snap/<slug>/,
rank its files for a prompt, and ground [src:slug:relpath:line] tags and the numbers of a src line in it (0 tokens).
API for topic pages: fetch(vault, kind, ident, slug) -> meta (sha); source_evidence(snap_dir(vault, slug), kind, budget) -> prompt text;
check_src(body, slug, vault) -> errors. No import of memrove_v3_wiki, so memrove_v3_wiki may import this module."""
from datetime import date
from html.parser import HTMLParser
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

UA = "memrove-wiki-fetch/1"  # no personal data goes out: no email, no user name, no token
SRC = re.compile(r"\[src:([a-z0-9-]+):([^\]\s:]+):(\d+)\]")
ANYTAG = re.compile(r"\[(?:src|doc|commit|memrove|m|grab):[^\]]*\]")
NUM = re.compile(r"\d+(?:[.,]\d+)*")
NAME_WITH_DIGIT = re.compile(r"\b[A-Za-z][A-Za-z-]*\d+[A-Za-z\d]*\b")  # BM25, Qwen3, k1: a name, not a quantity
MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.S)
SNAP_META = "SNAPSHOT.json"
MAX_FILE = 2_000_000  # a snapshot file above this is not read for the number check or the prompt


def _run(cmd, cwd=None, timeout=180):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""


def _get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(30_000_000)
    return data if binary else data.decode("utf-8", "replace")


def snap_dir(vault, slug):
    return Path(vault) / "wiki" / "_sources" / "snap" / slug


def _finish(dest, meta):
    """Drop symlinks (a link out of the snapshot would let a plain read leave it), then record the pin."""
    for p in list(Path(dest).rglob("*")):
        if p.is_symlink():
            p.unlink()
    meta["fetched"] = date.today().isoformat()
    meta["files"] = sum(1 for p in Path(dest).rglob("*") if p.is_file())
    meta["bytes"] = sum(p.stat().st_size for p in Path(dest).rglob("*") if p.is_file())
    (Path(dest) / SNAP_META).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


# ---- fetch (deterministic, no LLM, nothing in the snapshot is run) -------------------------------------------------------

def fetch_repo(url, dest):
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"snapshot dir not empty: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    rc, out = _run(["git", "clone", "--depth", "1", "--no-tags", "--quiet", url, str(dest)], cwd=str(dest.parent))
    if rc:
        raise ValueError(f"git clone failed: {out.strip()[:200]}")
    rc, sha = _run(["git", "-C", str(dest), "rev-parse", "HEAD"])
    if rc or not re.fullmatch(r"[0-9a-f]{40}", sha.strip()):
        raise ValueError("cannot read the cloned sha")
    shutil.rmtree(dest / ".git")  # snapshot is plain files only: no hooks, no config
    return _finish(dest, {"kind": "repo", "url": url, "sha": sha.strip()})


class _Text(HTMLParser):
    """arXiv HTML (LaTeXML) -> plain text; <math> becomes $alttext$ so formulas stay readable."""
    BLOCK = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "li", "tr", "br", "table", "figcaption", "blockquote", "pre"}
    SKIP = {"script", "style", "nav", "header", "footer", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "math":
            self.math += 1
            if self.math == 1:
                alt = " ".join((a.get("alttext") or "").split())
                if alt:
                    self.out.append(f"${alt}$")
        elif tag in self.BLOCK:
            self.out.append("\n")
            if tag.startswith("h"):
                self.out.append("## ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "math":
            self.math = max(0, self.math - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip and not self.math:
            self.out.append(data)


def html_to_text(html):
    p = _Text()
    p.feed(html)
    lines = [" ".join(l.split()) for l in "".join(p.out).splitlines()]
    return "\n".join(l for l in lines if l) + "\n"


def fetch_paper(arxiv_id, dest, get=_get):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError(f"snapshot dir not empty: {dest}")
    meta = {"kind": "paper", "arxiv": arxiv_id, "url": f"https://arxiv.org/abs/{arxiv_id}"}
    atom = get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    (dest / "abstract.xml").write_text(atom, encoding="utf-8")
    text = ""
    try:
        text = html_to_text(get(f"https://arxiv.org/html/{arxiv_id}"))
        meta["source"] = "arxiv-html"
    except OSError:
        pass
    if len(text) < 5000:  # no HTML version: PDF -> text with pdftotext (reads the file, executes nothing in it)
        (dest / "paper.pdf").write_bytes(get(f"https://arxiv.org/pdf/{arxiv_id}", binary=True))
        rc, _ = _run(["pdftotext", "-layout", str(dest / "paper.pdf"), str(dest / "paper.txt")])
        if rc:
            raise ValueError("no arXiv HTML and pdftotext failed")
        meta["source"] = "pdf"
    else:
        (dest / "paper.txt").write_text(text, encoding="utf-8")
    return _finish(dest, meta)


SMALL_MODEL_FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json", "modules.json",
                     "config_sentence_transformers.json", "generation_config.json", "preprocessor_config.json")


def fetch_model(repo_id, dest, get=_get):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError(f"snapshot dir not empty: {dest}")
    info = json.loads(get(f"https://huggingface.co/api/models/{repo_id}?blobs=true"))
    sha = info.get("sha") or ""
    files = {s["rfilename"]: s.get("size") or 0 for s in info.get("siblings", [])}
    keep = {k: info.get(k) for k in ("id", "sha", "pipeline_tag", "library_name", "tags", "safetensors", "license", "lastModified") if k in info}
    keep["files"] = files
    (dest / "model_info.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in files:
        base = name.rsplit("/", 1)[-1]
        if name == "README.md" or (base in SMALL_MODEL_FILES and files[name] <= 65536):
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(get(f"https://huggingface.co/{repo_id}/resolve/{sha}/{name}"), encoding="utf-8")
    return _finish(dest, {"kind": "model", "url": f"https://huggingface.co/{repo_id}", "sha": sha, "id": repo_id})


def fetch(vault, kind, ident, slug):
    dest = snap_dir(vault, slug)
    if kind == "repo":
        return fetch_repo(ident if "://" in ident or ident.startswith("/") else f"https://github.com/{ident}", dest)
    return (fetch_paper if kind == "paper" else fetch_model)(ident, dest)


# ---- check (0 tokens) ----------------------------------------------------------------------------------------------------

def _canon(n):
    """'1,000' / '1.000' / '0,5' -> ('1.000', '1000') style forms; a page number matches a source number in either form."""
    return n.replace(",", "."), re.sub(r"[.,]", "", n)


def _numbers(text):
    return NUM.findall(text)


def _file_numbers(path, cache):
    if path not in cache:
        a, b = set(), set()
        for n in _numbers(path.read_text(encoding="utf-8", errors="replace")):
            x, y = _canon(n)
            a.add(x)
            b.add(y)
        cache[path] = (a, b)
    return cache[path]


def _strip_for_numbers(line):
    s = ANYTAG.sub("", line)
    s = re.sub(r"\bN\d+(?:\.\d+)*\b|\bT\d+\b|#\d+\b|\b[A-Za-z]+@\d+\b", "", s)  # node ids (N1.2), technique ids (T1), issue numbers, NDCG@10
    s = re.sub(r"\]\(code:[^)]*\)", "]", s)
    return NAME_WITH_DIGIT.sub("", s)


def check_src(body, slug, vault):
    """Every [src:slug:relpath:line] in body must name an existing file and line of the snapshot, and every number on a line with
    src tags must appear in the cited file(s) (math included; 0,5 = 0.5, 1.000 = 1,000). Node ids (N1.2), #123 and NDCG@10 are not numbers.
    -> list of error strings. Cannot catch a correct number attached to the wrong claim."""
    sdir = snap_dir(vault, slug)
    if not sdir.is_dir():
        return [f"no snapshot at {sdir}"]
    errors, nums, base = [], {}, sdir.resolve()
    for n, line in enumerate(body.splitlines(), 1):
        srcs = SRC.findall(line)
        if not srcs:
            continue
        short = line.strip()[:70]
        files = []
        for s_slug, rel, ln in srcs:
            f = (sdir / rel).resolve()
            if s_slug != slug:
                errors.append(f"line {n}: src tag names slug {s_slug}, page uses {slug}")
            elif base not in f.parents or not f.is_file() or f.stat().st_size > MAX_FILE:
                errors.append(f"line {n}: src file not in snapshot: {rel}")
            elif not 1 <= int(ln) <= len(f.read_text(encoding="utf-8", errors="replace").splitlines()):
                errors.append(f"line {n}: src line out of range: {rel}:{ln}")
            else:
                files.append(f)
        if files and len(files) == len(srcs):
            sets = [_file_numbers(p, nums) for p in files]
            for num in _numbers(_strip_for_numbers(line.replace("\\$", ""))):
                a, b = _canon(num)
                if not any(a in st[0] or b in st[1] for st in sets):
                    errors.append(f"line {n}: number {num} not in the cited snapshot file(s): {short}")
    return list(dict.fromkeys(errors))


# ---- evidence ------------------------------------------------------------------------------------------------------------

CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".java", ".kt", ".swift", ".c", ".cc", ".cpp", ".h", ".hpp", ".rb", ".cs", ".scala"}
SKIP_DIRS = {"node_modules", "vendor", "dist", "build", "tests", "test", "__tests__", "examples", "benchmarks", "assets", "third_party", ".github"}
README_CAP, DOCS_CAP, FILE_CAP = 6000, 2500, 5000


def _numbered(path, rel, cap):
    out, n = [], 0
    for i, l in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        l = l.rstrip()[:300]
        if n + len(l) > cap:
            out.append(f"... (truncated at line {i}; the file continues)")
            break
        out.append(f"{i}| {l}")
        n += len(l) + 1
    return f"=== FILE {rel} ===\n" + "\n".join(out)


def rank_repo(sdir):
    """Deterministic: README first, then docs/*.md by size, then code files by (1 + files that import the stem) x size (capped at the per-file cap)."""
    files = [p for p in sorted(Path(sdir).rglob("*")) if p.is_file() and p.stat().st_size <= MAX_FILE
             and not any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(sdir).parts[:-1])]
    rel = lambda p: str(p.relative_to(sdir))
    readmes = [p for p in files if p.parent == Path(sdir) and p.name.lower().startswith("readme")]
    docs = sorted([p for p in files if p.suffix == ".md" and p not in readmes and ("docs" in p.relative_to(sdir).parts or p.parent == Path(sdir))
                   and "changelog" not in p.name.lower() and "license" not in p.name.lower()], key=lambda p: (-p.stat().st_size, rel(p)))
    code = [p for p in files if p.suffix in CODE_EXT and not p.name.startswith("test_") and not p.name.endswith(("_test.py", ".test.ts", ".d.ts", ".min.js"))]
    imp = re.compile(r"^\s*(?:from|import|use|mod|require|#include|package)\b.*", re.M)
    heads = {p: imp.findall(p.read_text(encoding="utf-8", errors="replace")) for p in code}
    def inbound(p):
        stem = p.stem if p.stem != "__init__" else p.parent.name
        pat = re.compile(r"\b" + re.escape(stem) + r"\b")
        return sum(1 for q, hs in heads.items() if q != p and any(pat.search(h) for h in hs))
    score = {p: inbound(p) for p in code}
    code = [p for p in code if p.stat().st_size >= 600]  # stubs and re-export files teach nothing
    code.sort(key=lambda p: (-(score[p] + 1) * min(p.stat().st_size, FILE_CAP), rel(p)))
    return readmes, docs, code, score


def source_evidence(sdir, kind, budget):
    sdir = Path(sdir)
    rel = lambda p: str(p.relative_to(sdir))
    blocks = []
    if kind == "repo":
        readmes, docs, code, score = rank_repo(sdir)
        listing = "\n".join(f"{rel(p)} (imported by {score[p]})" for p in code[:40])
        blocks.append("FILE RANKING ((1 + imported-by) x size):\n" + listing)
        picks = [(p, README_CAP) for p in readmes[:1]] + [(p, DOCS_CAP) for p in docs[:4]] + [(p, FILE_CAP) for p in code]
    elif kind == "paper":
        p = sdir / "paper.txt"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        cut = next((i for i, l in enumerate(lines) if re.match(r"(## )?(References|Bibliography)\s*$", l.strip())), len(lines))
        start = next((i for i, l in enumerate(lines[:cut]) if l.strip().lower().startswith("abstract")), 0)  # skip the page chrome
        tmp = "\n".join(f"{i}| {l[:400]}" for i, l in enumerate(lines[:cut], 1) if i > start)
        blocks.append(f"=== FILE paper.txt ({len(lines)} lines; references cut) ===\n" + tmp[:budget])
        picks = []
    else:
        picks = [(p, 12000) for p in sorted(sdir.rglob("*")) if p.is_file() and p.name != SNAP_META and p.name != "model_info.json"]
        picks.sort(key=lambda t: (t[0].name != "README.md", rel(t[0])))
        picks.append((sdir / "model_info.json", 3000))
    used = sum(len(b) for b in blocks)
    for p, cap in picks:
        if used >= budget:
            break
        if p.is_file():
            b = _numbered(p, rel(p), min(cap, budget - used))
            blocks.append(b)
            used += len(b)
    return "\n\n".join(blocks)
