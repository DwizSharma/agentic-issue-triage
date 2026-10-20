"""Triage pipeline used by the standalone API. Synthetic data is for demos only."""
import os, re, json, math, ast, random, itertools, operator, subprocess, tempfile, threading, time
import datetime as dt
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, TypedDict, Annotated

import numpy as np
import pandas as pd

CFG = dict(
    MODE="synthetic",                      # "synthetic" | "github"
    GITHUB_REPO="pallets/flask",           # used when MODE == "github"
    GITHUB_TOKEN=os.getenv("GITHUB_TOKEN"),
    MAX_ISSUES=400,
    MAX_COMMITS=3000,
    EMBED_MODEL="BAAI/bge-small-en-v1.5",  # or "sentence-transformers/all-MiniLM-L6-v2"
    DUP_THRESHOLD=0.86,                    # auto-calibrated below when ground truth exists
    TOP_K=5,
    RRF_K=60,
    TRACE_BOOST=0.05,
    # LLM: "stub" (no model, template test) | "hf" (local Qwen) | "openai_compat" (Ollama/vLLM server)
    LLM_BACKEND="stub",
    HF_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct",     # use 7B-Instruct on a bigger GPU
    OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    OPENAI_MODEL="qwen2.5-coder:7b",
    DB_URL=os.getenv("DATABASE_URL", "sqlite:///triage.db"),   # e.g. postgresql+psycopg2://user:pw@host/triage
    SEED=42,
)
CFG["MODE"] = os.getenv("TRIAGE_MODE", "synthetic").lower()
CFG["GITHUB_REPO"] = os.getenv("GITHUB_REPOSITORY", CFG["GITHUB_REPO"])
CFG["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")
CFG["EMBED_MODEL"] = os.getenv("EMBED_MODEL", CFG["EMBED_MODEL"])
CFG["DB_URL"] = os.getenv("DATABASE_URL", CFG["DB_URL"])
CFG["DUP_THRESHOLD"] = float(os.getenv("DUP_THRESHOLD", CFG["DUP_THRESHOLD"]))
CFG["MAX_ISSUES"] = int(os.getenv("MAX_ISSUES", CFG["MAX_ISSUES"]))
CFG["MAX_COMMITS"] = int(os.getenv("MAX_COMMITS", CFG["MAX_COMMITS"]))
CFG["LLM_BACKEND"] = os.getenv("LLM_BACKEND", "stub")
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])

class Corpus:
    def __init__(self, files, commits, issues, repo_dir=None):
        self.files, self.commits, self.repo_dir = files, commits, repo_dir
        self.issues = sorted(issues, key=lambda i: i["created"])
        self.by_sha = {c["sha"]: c for c in commits}
        self.issue_by_id = {i["id"]: i for i in self.issues}

    def diff(self, sha):
        c = self.by_sha.get(sha)
        if c is None:
            return ""
        if c.get("diff") is None and self.repo_dir:
            out = subprocess.run(["git", "-C", self.repo_dir, "show", "--no-color", "--patch", "--format=", sha],
                                 capture_output=True, text=True, errors="ignore").stdout
            c["diff"] = out[:4000]
        return c.get("diff") or ""

# ---------- Synthetic repo (smoke test) ----------
MODULES = {
 "src/auth/session.py": dict(owners=["alice","bob"], funcs=["create_session","refresh_token","validate_cookie","expire_session"],
   vocab="login session token cookie logout authentication", symptoms=[
   (["Users get logged out randomly right after the token refresh","Session token refresh fails and customer is signed out unexpectedly","Random sign-outs: refreshing the auth token kills the login session"],"high"),
   (["Security issue: session cookie still accepted after logout","Expired session cookie validates successfully, auth bypass","Cookie from a logged-out user is not invalidated"],"critical")]),
 "src/db/connection_pool.py": dict(owners=["carol","dave"], funcs=["acquire_connection","release_connection","health_check","resize_pool"],
   vocab="database connection pool timeout leak", symptoms=[
   (["Connection pool exhausted under load, all requests hang","Database connections leak and the pool runs dry, service outage","App freezes: no free DB connection available in the pool"],"critical"),
   (["Health check marks healthy database as down after idle period","Idle database connection reported unhealthy by health check","False negative from pool health_check on idle connections"],"medium")]),
 "src/api/rate_limiter.py": dict(owners=["erin","frank"], funcs=["check_quota","consume_token","refill_bucket"],
   vocab="rate limit quota token bucket throttle 429", symptoms=[
   (["Rate limiter returns 429 for first request of the day","Users throttled immediately although quota was reset","429 Too Many Requests on fresh quota window"],"medium"),
   (["Retry-After header shows negative seconds","Negative value in Retry-After header when throttled","Throttle response advertises a wrong retry delay"],"low")]),
 "src/parser/json_parser.py": dict(owners=["grace","heidi"], funcs=["parse_document","decode_unicode","parse_nested"],
   vocab="json parser unicode decode nested escape", symptoms=[
   (["JSON parser crashes on emoji in unicode strings","UnicodeDecodeError when document contains surrogate pairs","Parsing payloads with emoji throws decode exception"],"high"),
   (["Deeply nested arrays cause RecursionError in parser","Parser hits recursion limit on nested JSON","RecursionError while parsing deeply nested document"],"medium")]),
 "src/cache/lru_cache.py": dict(owners=["ivan","judy"], funcs=["get_entry","evict_oldest","set_ttl","purge_expired"],
   vocab="cache eviction ttl stale expiry", symptoms=[
   (["Cache returns stale values after the TTL has elapsed","Expired cache entries are still served to users","TTL ignored: outdated data is served from cache"],"high"),
   (["LRU evicts the most recently used entry instead of the oldest","Wrong eviction order, newest cache item removed first","Eviction policy removes hot keys"],"medium")]),
 "src/payments/invoice.py": dict(owners=["alice","karl"], funcs=["compute_total","apply_tax","convert_currency","round_amount"],
   vocab="invoice payment tax currency rounding total", symptoms=[
   (["Invoice totals off by one cent due to rounding, customers overcharged","Tax rounding error produces wrong invoice amount, billing incident","Customers billed incorrect totals after currency conversion rounding"],"critical"),
   (["VAT is applied twice for EU customers","Double tax on invoices for European accounts","Duplicate VAT line on EU invoice"],"medium")]),
 "src/upload/file_handler.py": dict(owners=["liam","mallory"], funcs=["save_upload","check_mime","parse_multipart"],
   vocab="upload file multipart mime size", symptoms=[
   (["Uploads larger than 2GB silently truncated, data loss","Large file upload is cut off without error","Big multipart uploads lose their tail, file corrupted"],"high"),
   (["PNG files rejected with wrong MIME type error","Valid PNG upload fails MIME validation","check_mime refuses legit png images"],"low")]),
 "src/search/indexer.py": dict(owners=["nina","oscar"], funcs=["tokenize_text","stem_word","build_index","merge_segments"],
   vocab="search index tokenizer stemming segment", symptoms=[
   (["Search index corrupted after segment merge, queries return nothing","Index corruption when merging segments, search is down","After merge_segments the search index is broken"],"critical"),
   (["Stemmer turns running into runn reducing recall","Incorrect stemming of words ending in ing","Search misses results because of over-aggressive stemming"],"low")]),
 "src/notify/email_sender.py": dict(owners=["peggy","quinn"], funcs=["send_email","render_template","retry_send"],
   vocab="email smtp notification template retry", symptoms=[
   (["Emails are sent twice when SMTP retry kicks in","Duplicate notification emails after smtp timeout retry","Customers receive two copies of every email on retry"],"medium"),
   (["Email template shows raw placeholder instead of the user name","Unrendered name variable in notification email","Template placeholder not substituted in emails"],"low")]),
 "src/scheduler/cron.py": dict(owners=["rupert","sybil"], funcs=["next_run","handle_dst","missed_jobs"],
   vocab="scheduler cron timezone dst job", symptoms=[
   (["Jobs skipped during daylight saving time change","Cron jobs do not run on the DST transition night","Scheduled job missed when clocks go forward"],"high"),
   (["Cron schedule uses server timezone instead of the user timezone","Wrong timezone applied to scheduled jobs","Jobs fire at UTC instead of configured timezone"],"medium")]),
}
COMMIT_MSGS = ["Refactor {fn}", "Add logging to {fn}", "Update {w} handling in {fn}", "Optimize {fn}", "Clean up {fn} and tests"]
NOISE = ["Happens on Python 3.11 on Ubuntu.", "Started after the last release.", "Cannot reproduce locally every time.",
         "Customer escalated this via support.", "Seen in production and staging.", "Please advise, blocking our deploy."]

def _make_code(path, m):
    L = [f'"""{path}: {m["vocab"]}"""', "import time", ""]
    for f in m["funcs"]:
        L += [f"def {f}(ctx, *args, **kwargs):", f'    """Handle {f.replace("_", " ")} ({m["vocab"]})."""',
              "    state = ctx.get('state', {})", f"    # {m['vocab']}", "    return state", ""]
    return "\n".join(L)

def load_synthetic(seed=42, variants=(3, 6)):
    rng = random.Random(seed); day = 86400; t0 = 1_650_000_000
    files, commits, issues = {}, [], []
    pool = sorted({o for m in MODULES.values() for o in m["owners"]})
    ctr = itertools.count(1)
    new_sha = lambda: f"{next(ctr):07x}" + "".join(rng.choice("0123456789abcdef") for _ in range(33))
    files["src/common/utils.py"] = '"""helpers: retry format string time"""\ndef retry(fn, n=3):\n    return fn()\n'
    for path, m in MODULES.items():
        files[path] = _make_code(path, m)
        for _ in range(14):
            fn = rng.choice(m["funcs"]); a = rng.choice(m["owners"]) if rng.random() < .8 else rng.choice(pool)
            commits.append(dict(sha=new_sha(), author=a, ts=t0 + rng.randint(0, 200) * day, files=[path],
                                message=rng.choice(COMMIT_MSGS).format(fn=fn, w=rng.choice(m["vocab"].split())),
                                diff=f"@@ {path} @@\n- old {fn}\n+ new {fn} logic"))
    groups = []
    for path, m in MODULES.items():
        for phr, sev in m["symptoms"]:
            bug = rng.choice([c for c in commits if c["files"] == [path]])
            bug["diff"] += f"\n+ # behaviour change: {phr[0].lower()}"       # tiny signal so blame is learnable
            groups.append((path, m, phr, sev, bug, rng.randint(*variants)))
    events = []
    for gi, g in enumerate(groups):
        base = t0 + 210 * day + rng.randint(0, 120) * day
        for j in range(g[5]):
            events.append((base + j * rng.randint(1, 15) * day + rng.randint(0, day - 1), gi))
    events.sort()
    canon, fixsha = {}, {}
    for iid, (ts, gi) in enumerate(events, 1):
        path, m, phr, sev, bug, _ = groups[gi]
        a, b = rng.sample(phr, 2)
        body = f"{b}.\n\n{rng.choice(NOISE)} {rng.choice(NOISE)}"
        if rng.random() < .3:
            body += f'\n\nTraceback (most recent call last):\n  File "{path}", line {rng.randint(10, 60)}, in {rng.choice(m["funcs"])}\n    raise RuntimeError("unexpected state")'
        fixed = [path] + (["src/common/utils.py"] if rng.random() < .15 else [])
        if gi not in canon:
            canon[gi] = iid
            sha = new_sha(); fixsha[gi] = sha
            commits.append(dict(sha=sha, author=rng.choice(m["owners"]), ts=ts + 2 * day, files=fixed,
                                message=f"Fix #{iid}: address reported problem", diff=f"@@ {path} @@\n+ fix"))
        issues.append(dict(id=iid, title=a, body=body, created=ts, severity=sev, fixed_files=fixed, bug_sha=bug["sha"],
                           fix_sha=fixsha[gi], group=canon[gi], duplicate_of=None if canon[gi] == iid else canon[gi]))
    return Corpus(files, commits, issues)

CODE_EXT = (".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".rb", ".php")
SEV_PATTERNS = [("critical", r"critical|blocker|p0|security"), ("high", r"high|major|p1"),
                ("medium", r"medium|p2"), ("low", r"low|minor|trivial|p3")]

def load_github(repo, token=None, max_issues=400, max_commits=3000):
    import requests
    hdr = {"Accept": "application/vnd.github+json"}
    if token: hdr["Authorization"] = f"Bearer {token}"
    clone_cmd = ["git"]
    git_env = os.environ.copy()
    if token:
        import base64
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        auth_header = f"AUTHORIZATION: basic {auth}"
        git_env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraheader",
                        "GIT_CONFIG_VALUE_0": auth_header})
    repo_cache = os.getenv("REPO_CACHE_DIR")
    d = os.path.join(repo_cache, repo.replace("/", "__")) if repo_cache else tempfile.mkdtemp(prefix="repo_")
    repo_url = f"https://github.com/{repo}.git"
    if os.path.isdir(os.path.join(d, ".git")):
        subprocess.run(["git", "-C", d, "fetch", "--quiet", "--all", "--prune"], check=True,
                       capture_output=True, text=True, env=git_env)
        subprocess.run(["git", "-C", d, "pull", "--quiet", "--ff-only"], check=True,
                       capture_output=True, text=True, env=git_env)
    else:
        if repo_cache:
            os.makedirs(repo_cache, exist_ok=True)
        clone_cmd += ["clone", "--quiet", repo_url, d]
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True, env=git_env)
    raw = subprocess.run(["git", "-C", d, "log", "--no-merges", "--name-only", "-n", str(max_commits),
                          "--pretty=format:@@@%H|%an|%at|%s"], capture_output=True, text=True, errors="ignore").stdout
    commits = []
    for block in raw.split("@@@")[1:]:
        lines = block.strip().split("\n"); sha, author, ts, msg = lines[0].split("|", 3)
        commits.append(dict(sha=sha, author=author, ts=float(ts), message=msg, files=[f for f in lines[1:] if f], diff=None))
    files = {}
    for f in subprocess.run(["git", "-C", d, "ls-files"], capture_output=True, text=True).stdout.split("\n"):
        if f.endswith(CODE_EXT):
            try:
                t = open(os.path.join(d, f), errors="ignore").read()
                if len(t) < 200_000: files[f] = t
            except OSError: pass
    fix_re = re.compile(r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.I)
    fix_map = {}
    for c in commits:
        for n in fix_re.findall(c["message"]): fix_map.setdefault(int(n), c)
    issues, page = [], 1
    while len(issues) < max_issues and page <= 10:
        r = requests.get(f"https://api.github.com/repos/{repo}/issues", headers=hdr,
                         params=dict(state="closed", labels="bug", per_page=100, page=page))
        r.raise_for_status(); batch = r.json()
        if not batch: break
        for it in batch:
            if "pull_request" in it: continue
            fx = fix_map.get(it["number"])
            labels = " ".join(l["name"].lower() for l in it.get("labels", []))
            sev = next((s for s, p in SEV_PATTERNS if re.search(p, labels)), None)
            issues.append(dict(id=it["number"], title=it["title"], body=it.get("body") or "",
                               created=dt.datetime.fromisoformat(it["created_at"].replace("Z", "+00:00")).timestamp(),
                               severity=sev, fixed_files=[f for f in fx["files"] if f in files] if fx else [],
                               fix_sha=fx["sha"] if fx else None, bug_sha=None, group=None, duplicate_of=None))
        page += 1
    return Corpus(files, commits, issues, repo_dir=d)


corpus = load_synthetic(CFG["SEED"]) if CFG["MODE"] == "synthetic" else load_github(
    CFG["GITHUB_REPO"], CFG["GITHUB_TOKEN"], CFG["MAX_ISSUES"], CFG["MAX_COMMITS"])
FIX_SHAS = {i["fix_sha"] for i in corpus.issues if i.get("fix_sha")}
split = int(0.6 * len(corpus.issues))
train_issues = corpus.issues[:split]
test_issues = corpus.issues[split:]


from qdrant_client import QdrantClient, models

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hashing").lower()
if EMBEDDING_BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(CFG["EMBED_MODEL"])
    BGE_Q = "Represent this sentence for searching relevant passages: " if "bge" in CFG["EMBED_MODEL"].lower() else ""
    def embed(texts, query=False):
        prefix = BGE_Q if query else ""
        return np.asarray(embedder.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=64, show_progress_bar=False), dtype="float32")
else:
    from sklearn.feature_extraction.text import HashingVectorizer
    embedder = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    def embed(texts, query=False):
        return np.asarray(embedder.transform(texts).toarray(), dtype="float32")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdr = (QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
       if QDRANT_URL else QdrantClient(":memory:"))
DIM = embed(["probe"]).shape[1]
def make_collection(name):
    if qdr.collection_exists(name): qdr.delete_collection(name)
    qdr.create_collection(name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
def upsert(name, vecs, payloads, start=0, bs=256):
    for i in range(0, len(vecs), bs):
        qdr.upsert(name, points=[models.PointStruct(id=start+i+j, vector=v.tolist(), payload=p) for j,(v,p) in enumerate(zip(vecs[i:i+bs], payloads[i:i+bs]))])
issue_text = lambda i: f"{i['title']}\n{i.get('body', '')}"[:2000]


class IssueStore:
    def __init__(self, name="issues"):
        self.name = name
        if not qdr.collection_exists(name):
            make_collection(name)
        self.n = qdr.count(name, exact=True).count
    def add(self, issue):
        upsert(self.name, embed([issue_text(issue)]), [dict(issue_id=issue["id"], title=issue["title"])], start=self.n); self.n += 1
    def add_many(self, issues):
        if issues:
            upsert(self.name, embed([issue_text(i) for i in issues]), [dict(issue_id=i["id"], title=i["title"]) for i in issues], start=self.n)
            self.n += len(issues)
    def nearest(self, issue, k=3):
        if self.n == 0: return []
        return qdr.query_points(self.name, query=embed([issue_text(issue)])[0].tolist(), limit=k).points



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
SEV = ["low", "medium", "high", "critical"]
CRIT_RE = re.compile(r"\b(security|data loss|corrupt\w*|auth(?:entication)? bypass|vulnerab\w*|rce|segfault|outage)\b", re.I)

class SeverityModel:
    def __init__(self): self.clf = None
    def fit(self, issues):
        lab = [i for i in issues if i.get("severity") in SEV]
        if len(lab) >= 20 and len({i["severity"] for i in lab}) > 1:
            self.clf = LogisticRegression(max_iter=3000, C=4, class_weight="balanced").fit(
                embed([issue_text(i) for i in lab]), [i["severity"] for i in lab])
        return self
    def predict(self, issue):
        if CRIT_RE.search(issue_text(issue)): return "critical"
        if self.clf is None:
            return "high" if re.search(r"regression|traceback|exception|crash|fails?", issue_text(issue), re.I) else "medium"
        return self.clf.predict(embed([issue_text(issue)]))[0]

sev_model = SeverityModel().fit(train_issues)



from rank_bm25 import BM25Okapi

def tokenize(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w.lower() for w in re.findall(r"[A-Za-z]+|\d+", text.replace("_", " ")) if len(w) > 1]

def chunk_lines(text, size=40, stride=30):
    L = text.split("\n")
    return ["\n".join(L[i:i + size]) for i in range(0, max(len(L) - size + stride, 1), stride)] or [text]

def extract_trace_paths(text, known):
    cands = set(re.findall(r'File "([^"]+)"', text)) | set(re.findall(r"([\w./-]+\.(?:py|js|ts|tsx|java|go|rs|cpp|cc|c|h|rb|php))", text))
    out = []
    for c in cands:
        c = c.lstrip("./")
        out += [f for f in known if f == c or f.endswith("/" + c)]
    return list(dict.fromkeys(out))

class CodeIndex:
    def __init__(self, corpus, exclude=frozenset()):
        self.corpus, self.chunks = corpus, []
        hist = defaultdict(list)
        for c in corpus.commits:
            if c["sha"] in exclude: continue
            for f in c["files"]: hist[f].append(c["message"])
        for path, text in corpus.files.items():
            self.chunks += [dict(file=path, kind="code", text=f"{path}\n{ch}") for ch in chunk_lines(text)]
            if hist.get(path): self.chunks.append(dict(file=path, kind="history", text=f"{path}\n" + "\n".join(hist[path][:60])))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        vecs = embed([c["text"][:2000] for c in self.chunks])
        make_collection("code"); upsert("code", vecs, [dict(file=c["file"], kind=c["kind"]) for c in self.chunks])

    def search(self, query, k=5, mode="hybrid", use_trace=True):
        scores = defaultdict(float)
        def fuse(ranks):
            for r, idx in enumerate(ranks): scores[int(idx)] += 1 / (CFG["RRF_K"] + r + 1)
        if mode in ("bm25", "hybrid"): fuse(np.argsort(-self.bm25.get_scores(tokenize(query)))[:100])
        if mode in ("dense", "hybrid"):
            fuse([h.id for h in qdr.query_points("code", query=embed([query[:2000]], query=True)[0].tolist(), limit=100).points])
        fs = defaultdict(float)
        for idx, s in scores.items():
            f = self.chunks[idx]["file"]; fs[f] = max(fs[f], s)
        if use_trace and mode == "hybrid":
            for p in extract_trace_paths(query, self.corpus.files): fs[p] += CFG["TRACE_BOOST"]
        return sorted(fs.items(), key=lambda x: -x[1])[:k]




code_index = CodeIndex(corpus, exclude=FIX_SHAS)


_commit_vec = {}
def _cvec(sha):
    if sha not in _commit_vec:
        c = corpus.by_sha[sha]; _commit_vec[sha] = embed([f"{c['message']}\n{corpus.diff(sha)[:1500]}"])[0]
    return _commit_vec[sha]

def suspect_commits(issue, files, topn=3, max_cands=80):
    before = issue["created"]
    cands = [c for c in corpus.commits if c["sha"] not in FIX_SHAS and c["ts"] < before and set(c["files"]) & set(files)]
    cands = sorted(cands, key=lambda c: -c["ts"])[:max_cands]
    if not cands: return []
    q = embed([issue_text(issue)], query=True)[0]
    out = []
    for c in cands:
        age = (before - c["ts"]) / 86400
        out.append(dict(sha=c["sha"], author=c["author"], message=c["message"], file_hits=sorted(set(c["files"]) & set(files)),
                        score=float(q @ _cvec(c["sha"])) + 0.15 * math.exp(-age / 90)))
    return sorted(out, key=lambda x: -x["score"])[:topn]

def suggest_assignees(issue, files, suspects, topn=3, half_life_days=90):
    sc = defaultdict(float)
    for c in corpus.commits:
        if c["sha"] in FIX_SHAS or c["ts"] >= issue["created"] or not (set(c["files"]) & set(files)): continue
        sc[c["author"]] += math.exp(-(issue["created"] - c["ts"]) / (half_life_days * 86400))
    if suspects: sc[suspects[0]["author"]] += 2.0
    return [a for a, _ in sorted(sc.items(), key=lambda x: -x[1])[:topn]]

_hf_pipe = None
def llm_generate(system, user, max_new_tokens=700):
    global _hf_pipe
    b = CFG["LLM_BACKEND"]
    if b == "hf":
        if _hf_pipe is None:
            from transformers import pipeline
            _hf_pipe = pipeline("text-generation", model=CFG["HF_MODEL"], device_map="auto", torch_dtype="auto")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return _hf_pipe(msgs, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"][-1]["content"]
    if b == "openai_compat":
        from openai import OpenAI
        cl = OpenAI(base_url=CFG["OPENAI_BASE_URL"], api_key=os.getenv("OPENAI_API_KEY", "none"))
        r = cl.chat.completions.create(model=CFG["OPENAI_MODEL"], temperature=0, max_tokens=max_new_tokens,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content
    return None

def stub_repro(issue, path, text):
    funcs = re.findall(r"^def (\w+)\(", text, re.M) or ["target_function"]
    mod = path[:-3].replace("/", ".") if path.endswith(".py") else path
    return f'''```python
import pytest
from {mod} import {funcs[0]}

def test_regression_issue_{issue["id"]}():
    """Repro for: {issue["title"][:80].replace('"', "'")}"""
    ctx = {{"state": {{}}}}
    result = {funcs[0]}(ctx)
    # TODO: replace with the expected behaviour described in the report
    assert result is not None
```'''

REPRO_SYS = ("You are a senior engineer. Write ONE minimal, deterministic pytest test that reproduces the reported bug "
             "against the given source file. The test must FAIL on the buggy behaviour. Import only from the repo and the "
             "standard library. Output only a single ```python code block.")

def extract_code(txt):
    m = re.search(r"```(?:python)?\n(.*?)```", txt or "", re.S)
    return (m.group(1) if m else (txt or "")).strip()

def validate_test(code):
    try: tree = ast.parse(code)
    except SyntaxError as e: return False, f"SyntaxError: {e}"
    if not any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in ast.walk(tree)):
        return False, "no test_* function found"
    return True, ""

from langgraph.graph import StateGraph, START, END

class TriageState(TypedDict, total=False):
    issue: dict; severity: str; duplicate: Optional[dict]; files: list; suspects: list; assignees: list
    repro_test: str; repro_ok: bool; attempts: int; error: str; report: str
    log: Annotated[list, operator.add]

store = IssueStore("issues_live")
if store.n == 0:
    store.add_many(train_issues)

def n_classify(s):
    sev = sev_model.predict(s["issue"]); return dict(severity=sev, log=[f"severity={sev}"])

def n_dedup(s):
    hits = store.nearest(s["issue"], 3); top = hits[0] if hits else None
    dup = dict(issue_id=top.payload["issue_id"], title=top.payload["title"], score=round(top.score, 3)) \
          if top and top.score >= CFG["DUP_THRESHOLD"] else None
    store.add(s["issue"])
    return dict(duplicate=dup, log=[f"duplicate={dup['issue_id'] if dup else None}"])

def route_after_dedup(s): return "report" if s.get("duplicate") else "localize"

def n_localize(s):
    q = issue_text(s["issue"]); res = code_index.search(q, k=CFG["TOP_K"])
    return dict(files=[dict(path=f, score=round(sc, 4)) for f, sc in res], log=[f"top file={res[0][0] if res else None}"])

def n_blame(s):
    sus = suspect_commits(s["issue"], [f["path"] for f in s["files"][:3]]); return dict(suspects=sus, log=[f"suspect={sus[0]['sha'][:8] if sus else None}"])

def n_assign(s):
    a = suggest_assignees(s["issue"], [f["path"] for f in s["files"][:3]], s.get("suspects", [])); return dict(assignees=a, log=[f"assignees={a}"])

def n_repro(s):
    top = s["files"][0]["path"]; text = corpus.files[top]
    sus = s["suspects"][0] if s.get("suspects") else None
    user = (f"BUG REPORT\nTitle: {s['issue']['title']}\n{s['issue']['body']}\n\nSOURCE FILE: {top}\n{text[:6000]}\n\n"
            + (f"SUSPECT COMMIT {sus['sha'][:8]} ({sus['message']}):\n{corpus.diff(sus['sha'])[:2000]}\n" if sus else "")
            + (f"\nYour previous attempt failed validation: {s['error']}. Fix it.\n" if s.get("error") else ""))
    raw = llm_generate(REPRO_SYS, user) or stub_repro(s["issue"], top, text)
    return dict(repro_test=extract_code(raw), attempts=s.get("attempts", 0) + 1, log=[f"repro attempt {s.get('attempts', 0) + 1}"])

def n_validate(s):
    ok, err = validate_test(s["repro_test"]); return dict(repro_ok=ok, error=err, log=[f"repro valid={ok}"])

def route_after_validate(s): return "report" if s["repro_ok"] or s["attempts"] >= 3 else "repro"

def n_report(s):
    i = s["issue"]; L = [f"### 🤖 Automated triage for #{i['id']}", f"**Severity:** `{s.get('severity')}`"]
    if s.get("duplicate"):
        d = s["duplicate"]; L.append(f"**Likely duplicate of** #{d['issue_id']} — _{d['title']}_ (similarity {d['score']})")
    else:
        L.append("**Top suspect files:** " + ", ".join(f"`{f['path']}`" for f in s.get("files", [])))
        if s.get("suspects"): x = s["suspects"][0]; L.append(f"**Suspect commit:** `{x['sha'][:8]}` by {x['author']} — {x['message']}")
        L.append("**Suggested assignees:** " + ", ".join(f"@{a}" for a in s.get("assignees", [])))
        if s.get("repro_test"): L.append("<details><summary>Generated repro test</summary>\n\n```python\n" + s["repro_test"] + "\n```\n</details>")
    return dict(report="\n\n".join(L))

g = StateGraph(TriageState)
for name, fn in [("classify", n_classify), ("dedup", n_dedup), ("localize", n_localize), ("blame", n_blame),
                 ("assign", n_assign), ("repro", n_repro), ("validate", n_validate), ("report", n_report)]:
    g.add_node(name, fn)
g.add_edge(START, "classify"); g.add_edge("classify", "dedup")
g.add_conditional_edges("dedup", route_after_dedup, {"report": "report", "localize": "localize"})
g.add_edge("localize", "blame"); g.add_edge("blame", "assign"); g.add_edge("assign", "repro"); g.add_edge("repro", "validate")
g.add_conditional_edges("validate", route_after_validate, {"report": "report", "repro": "repro"})
g.add_edge("report", END)
triage_app = g.compile()

def triage_issue(issue): return triage_app.invoke({"issue": issue, "attempts": 0, "log": []})


