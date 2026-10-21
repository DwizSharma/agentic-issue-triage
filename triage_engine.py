"""Triage pipeline used by the standalone API. Synthetic data is for demos only."""
import os
import re
import math
import ast
import random
import itertools
import operator
import subprocess
import tempfile
import threading
import time
import logging
import datetime as dt
from collections import defaultdict
from functools import lru_cache
from typing import Optional, TypedDict, Annotated

import numpy as np

LOG = logging.getLogger("triage.engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG = dict(
    MODE="synthetic",                       # "synthetic" | "github"
    GITHUB_REPO="pallets/flask",            # used when MODE == "github"
    GITHUB_TOKEN=os.getenv("GITHUB_TOKEN"),
    MAX_ISSUES=400,
    MAX_COMMITS=3000,
    EMBED_MODEL="BAAI/bge-small-en-v1.5",   # or "sentence-transformers/all-MiniLM-L6-v2"
    DUP_THRESHOLD=0.86,                     # auto-calibrated when ground truth exists
    TOP_K=5,
    RRF_K=60,
    TRACE_BOOST=0.05,
    # LLM: "stub" | "hf" (local Qwen) | "openai_compat" (Ollama/vLLM)
    LLM_BACKEND="stub",
    HF_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    OPENAI_MODEL="qwen2.5-coder:7b",
    DB_URL=os.getenv("DATABASE_URL", "sqlite:///triage.db"),
    SEED=42,
)

# Environment overrides — applied once at module load
CFG["MODE"] = os.getenv("TRIAGE_MODE", "synthetic").lower()
CFG["GITHUB_REPO"] = os.getenv("GITHUB_REPOSITORY", CFG["GITHUB_REPO"])
CFG["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")
CFG["EMBED_MODEL"] = os.getenv("EMBED_MODEL", CFG["EMBED_MODEL"])
CFG["DB_URL"] = os.getenv("DATABASE_URL", CFG["DB_URL"])
CFG["DUP_THRESHOLD"] = float(os.getenv("DUP_THRESHOLD", CFG["DUP_THRESHOLD"]))
CFG["MAX_ISSUES"] = int(os.getenv("MAX_ISSUES", CFG["MAX_ISSUES"]))
CFG["MAX_COMMITS"] = int(os.getenv("MAX_COMMITS", CFG["MAX_COMMITS"]))
CFG["LLM_BACKEND"] = os.getenv("LLM_BACKEND", "stub")

# Seed only the module-level RNG instances; do not touch the global state
_rng = random.Random(CFG["SEED"])
np.random.seed(CFG["SEED"])


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class Corpus:
    """Container for source files, commit history, and issues."""

    def __init__(self, files, commits, issues, repo_dir=None):
        self.files = files
        self.commits = commits
        self.repo_dir = repo_dir
        self.issues = sorted(issues, key=lambda i: i["created"])
        self.by_sha = {c["sha"]: c for c in commits}
        self.issue_by_id = {i["id"]: i for i in self.issues}

    def diff(self, sha):
        """Return the patch for *sha*, lazily fetching it from git when needed."""
        c = self.by_sha.get(sha)
        if c is None:
            return ""
        if c.get("diff") is None and self.repo_dir:
            result = subprocess.run(
                ["git", "-C", self.repo_dir, "show",
                 "--no-color", "--patch", "--format=", sha],
                capture_output=True, text=True, errors="ignore",
            )
            c["diff"] = result.stdout[:4000]
        return c.get("diff") or ""


# ---------------------------------------------------------------------------
# Synthetic corpus (demo / smoke-test)
# ---------------------------------------------------------------------------

MODULES = {
    "src/auth/session.py": dict(
        owners=["alice", "bob"],
        funcs=["create_session", "refresh_token", "validate_cookie", "expire_session"],
        vocab="login session token cookie logout authentication",
        symptoms=[
            (["Users get logged out randomly right after the token refresh",
              "Session token refresh fails and customer is signed out unexpectedly",
              "Random sign-outs: refreshing the auth token kills the login session"], "high"),
            (["Security issue: session cookie still accepted after logout",
              "Expired session cookie validates successfully, auth bypass",
              "Cookie from a logged-out user is not invalidated"], "critical"),
        ],
    ),
    "src/db/connection_pool.py": dict(
        owners=["carol", "dave"],
        funcs=["acquire_connection", "release_connection", "health_check", "resize_pool"],
        vocab="database connection pool timeout leak",
        symptoms=[
            (["Connection pool exhausted under load, all requests hang",
              "Database connections leak and the pool runs dry, service outage",
              "App freezes: no free DB connection available in the pool"], "critical"),
            (["Health check marks healthy database as down after idle period",
              "Idle database connection reported unhealthy by health check",
              "False negative from pool health_check on idle connections"], "medium"),
        ],
    ),
    "src/api/rate_limiter.py": dict(
        owners=["erin", "frank"],
        funcs=["check_quota", "consume_token", "refill_bucket"],
        vocab="rate limit quota token bucket throttle 429",
        symptoms=[
            (["Rate limiter returns 429 for first request of the day",
              "Users throttled immediately although quota was reset",
              "429 Too Many Requests on fresh quota window"], "medium"),
            (["Retry-After header shows negative seconds",
              "Negative value in Retry-After header when throttled",
              "Throttle response advertises a wrong retry delay"], "low"),
        ],
    ),
    "src/parser/json_parser.py": dict(
        owners=["grace", "heidi"],
        funcs=["parse_document", "decode_unicode", "parse_nested"],
        vocab="json parser unicode decode nested escape",
        symptoms=[
            (["JSON parser crashes on emoji in unicode strings",
              "UnicodeDecodeError when document contains surrogate pairs",
              "Parsing payloads with emoji throws decode exception"], "high"),
            (["Deeply nested arrays cause RecursionError in parser",
              "Parser hits recursion limit on nested JSON",
              "RecursionError while parsing deeply nested document"], "medium"),
        ],
    ),
    "src/cache/lru_cache.py": dict(
        owners=["ivan", "judy"],
        funcs=["get_entry", "evict_oldest", "set_ttl", "purge_expired"],
        vocab="cache eviction ttl stale expiry",
        symptoms=[
            (["Cache returns stale values after the TTL has elapsed",
              "Expired cache entries are still served to users",
              "TTL ignored: outdated data is served from cache"], "high"),
            (["LRU evicts the most recently used entry instead of the oldest",
              "Wrong eviction order, newest cache item removed first",
              "Eviction policy removes hot keys"], "medium"),
        ],
    ),
    "src/payments/invoice.py": dict(
        owners=["alice", "karl"],
        funcs=["compute_total", "apply_tax", "convert_currency", "round_amount"],
        vocab="invoice payment tax currency rounding total",
        symptoms=[
            (["Invoice totals off by one cent due to rounding, customers overcharged",
              "Tax rounding error produces wrong invoice amount, billing incident",
              "Customers billed incorrect totals after currency conversion rounding"], "critical"),
            (["VAT is applied twice for EU customers",
              "Double tax on invoices for European accounts",
              "Duplicate VAT line on EU invoice"], "medium"),
        ],
    ),
    "src/upload/file_handler.py": dict(
        owners=["liam", "mallory"],
        funcs=["save_upload", "check_mime", "parse_multipart"],
        vocab="upload file multipart mime size",
        symptoms=[
            (["Uploads larger than 2GB silently truncated, data loss",
              "Large file upload is cut off without error",
              "Big multipart uploads lose their tail, file corrupted"], "high"),
            (["PNG files rejected with wrong MIME type error",
              "Valid PNG upload fails MIME validation",
              "check_mime refuses legit png images"], "low"),
        ],
    ),
    "src/search/indexer.py": dict(
        owners=["nina", "oscar"],
        funcs=["tokenize_text", "stem_word", "build_index", "merge_segments"],
        vocab="search index tokenizer stemming segment",
        symptoms=[
            (["Search index corrupted after segment merge, queries return nothing",
              "Index corruption when merging segments, search is down",
              "After merge_segments the search index is broken"], "critical"),
            (["Stemmer turns running into runn reducing recall",
              "Incorrect stemming of words ending in ing",
              "Search misses results because of over-aggressive stemming"], "low"),
        ],
    ),
    "src/notify/email_sender.py": dict(
        owners=["peggy", "quinn"],
        funcs=["send_email", "render_template", "retry_send"],
        vocab="email smtp notification template retry",
        symptoms=[
            (["Emails are sent twice when SMTP retry kicks in",
              "Duplicate notification emails after smtp timeout retry",
              "Customers receive two copies of every email on retry"], "medium"),
            (["Email template shows raw placeholder instead of the user name",
              "Unrendered name variable in notification email",
              "Template placeholder not substituted in emails"], "low"),
        ],
    ),
    "src/scheduler/cron.py": dict(
        owners=["rupert", "sybil"],
        funcs=["next_run", "handle_dst", "missed_jobs"],
        vocab="scheduler cron timezone dst job",
        symptoms=[
            (["Jobs skipped during daylight saving time change",
              "Cron jobs do not run on the DST transition night",
              "Scheduled job missed when clocks go forward"], "high"),
            (["Cron schedule uses server timezone instead of the user timezone",
              "Wrong timezone applied to scheduled jobs",
              "Jobs fire at UTC instead of configured timezone"], "medium"),
        ],
    ),
}

COMMIT_MSGS = [
    "Refactor {fn}",
    "Add logging to {fn}",
    "Update {w} handling in {fn}",
    "Optimize {fn}",
    "Clean up {fn} and tests",
]
NOISE = [
    "Happens on Python 3.11 on Ubuntu.",
    "Started after the last release.",
    "Cannot reproduce locally every time.",
    "Customer escalated this via support.",
    "Seen in production and staging.",
    "Please advise, blocking our deploy.",
]


def _make_code(path, m):
    lines = [f'"""{path}: {m["vocab"]}"""', "import time", ""]
    for fn in m["funcs"]:
        lines += [
            f"def {fn}(ctx, *args, **kwargs):",
            f'    """Handle {fn.replace("_", " ")} ({m["vocab"]})."""',
            "    state = ctx.get('state', {})",
            f"    # {m['vocab']}",
            "    return state",
            "",
        ]
    return "\n".join(lines)


def load_synthetic(seed=42, variants=(3, 6)):
    """Build a fully synthetic corpus for demos and smoke tests."""
    rng = random.Random(seed)
    day = 86400
    t0 = 1_650_000_000
    files, commits, issues = {}, [], []
    pool = sorted({o for m in MODULES.values() for o in m["owners"]})
    ctr = itertools.count(1)

    def new_sha():
        return f"{next(ctr):07x}" + "".join(
            rng.choice("0123456789abcdef") for _ in range(33)
        )

    files["src/common/utils.py"] = (
        '"""helpers: retry format string time"""\ndef retry(fn, n=3):\n    return fn()\n'
    )

    for path, m in MODULES.items():
        files[path] = _make_code(path, m)
        for _ in range(14):
            fn = rng.choice(m["funcs"])
            author = rng.choice(m["owners"]) if rng.random() < 0.8 else rng.choice(pool)
            commits.append(dict(
                sha=new_sha(),
                author=author,
                ts=t0 + rng.randint(0, 200) * day,
                files=[path],
                message=rng.choice(COMMIT_MSGS).format(fn=fn, w=rng.choice(m["vocab"].split())),
                diff=f"@@ {path} @@\n- old {fn}\n+ new {fn} logic",
            ))

    groups = []
    for path, m in MODULES.items():
        for phr, sev in m["symptoms"]:
            bug = rng.choice([c for c in commits if c["files"] == [path]])
            bug["diff"] += f"\n+ # behaviour change: {phr[0].lower()}"
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
        if rng.random() < 0.3:
            body += (
                f'\n\nTraceback (most recent call last):\n'
                f'  File "{path}", line {rng.randint(10, 60)}, '
                f'in {rng.choice(m["funcs"])}\n'
                f'    raise RuntimeError("unexpected state")'
            )
        fixed = [path] + (["src/common/utils.py"] if rng.random() < 0.15 else [])
        if gi not in canon:
            canon[gi] = iid
            sha = new_sha()
            fixsha[gi] = sha
            commits.append(dict(
                sha=sha,
                author=rng.choice(m["owners"]),
                ts=ts + 2 * day,
                files=fixed,
                message=f"Fix #{iid}: address reported problem",
                diff=f"@@ {path} @@\n+ fix",
            ))
        issues.append(dict(
            id=iid,
            title=a,
            body=body,
            created=ts,
            severity=sev,
            fixed_files=fixed,
            bug_sha=bug["sha"],
            fix_sha=fixsha[gi],
            group=canon[gi],
            duplicate_of=None if canon[gi] == iid else canon[gi],
        ))

    return Corpus(files, commits, issues)


# ---------------------------------------------------------------------------
# GitHub corpus loader
# ---------------------------------------------------------------------------

CODE_EXT = (".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs",
            ".c", ".cc", ".cpp", ".h", ".rb", ".php")

SEV_PATTERNS = [
    ("critical", r"critical|blocker|p0|security"),
    ("high", r"high|major|p1"),
    ("medium", r"medium|p2"),
    ("low", r"low|minor|trivial|p3"),
]


def load_github(repo, token=None, max_issues=400, max_commits=3000):
    """Clone *repo* and load its closed bug issues from the GitHub API."""
    import requests as _requests

    hdr = {"Accept": "application/vnd.github+json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"

    # Pass the token via git environment rather than embedding it in the URL
    clone_cmd = ["git"]
    git_env = os.environ.copy()
    if token:
        import base64
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        git_env.update({
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        })

    repo_cache = os.getenv("REPO_CACHE_DIR")
    if repo_cache:
        d = os.path.join(repo_cache, repo.replace("/", "__"))
    else:
        d = tempfile.mkdtemp(prefix="repo_")

    repo_url = f"https://github.com/{repo}.git"

    if os.path.isdir(os.path.join(d, ".git")):
        LOG.info("Updating cached repo %s", repo)
        subprocess.run(
            ["git", "-C", d, "fetch", "--quiet", "--all", "--prune"],
            check=True, capture_output=True, text=True, env=git_env,
        )
        subprocess.run(
            ["git", "-C", d, "pull", "--quiet", "--ff-only"],
            check=True, capture_output=True, text=True, env=git_env,
        )
    else:
        if repo_cache:
            os.makedirs(repo_cache, exist_ok=True)
        LOG.info("Cloning %s into %s", repo_url, d)
        subprocess.run(
            clone_cmd + ["clone", "--quiet", repo_url, d],
            check=True, capture_output=True, text=True, env=git_env,
        )

    # Parse commit log
    raw = subprocess.run(
        ["git", "-C", d, "log", "--no-merges", "--name-only",
         "-n", str(max_commits), "--pretty=format:@@@%H|%an|%at|%s"],
        capture_output=True, text=True, errors="ignore",
    ).stdout

    commits = []
    for block in raw.split("@@@")[1:]:
        lines = block.strip().split("\n")
        sha, author, ts, msg = lines[0].split("|", 3)
        commits.append(dict(
            sha=sha,
            author=author,
            ts=float(ts),
            message=msg,
            files=[f for f in lines[1:] if f],
            diff=None,
        ))

    # Read source files — use context manager to avoid file descriptor leaks
    files = {}
    for fname in subprocess.run(
        ["git", "-C", d, "ls-files"], capture_output=True, text=True
    ).stdout.split("\n"):
        if not fname.endswith(CODE_EXT):
            continue
        try:
            with open(os.path.join(d, fname), errors="ignore") as fh:
                text = fh.read()
            if len(text) < 200_000:
                files[fname] = text
        except OSError:
            pass

    # Map fix-commits to issue numbers
    fix_re = re.compile(r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.I)
    fix_map = {}
    for c in commits:
        for n in fix_re.findall(c["message"]):
            fix_map.setdefault(int(n), c)

    # Fetch closed bug issues from the API
    issues, page = [], 1
    while len(issues) < max_issues and page <= 10:
        resp = _requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            headers=hdr,
            params=dict(state="closed", labels="bug", per_page=100, page=page),
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for it in batch:
            if "pull_request" in it:
                continue
            fx = fix_map.get(it["number"])
            labels = " ".join(lbl["name"].lower() for lbl in it.get("labels", []))
            sev = next((s for s, p in SEV_PATTERNS if re.search(p, labels)), None)
            issues.append(dict(
                id=it["number"],
                title=it["title"],
                body=it.get("body") or "",
                created=dt.datetime.fromisoformat(
                    it["created_at"].replace("Z", "+00:00")
                ).timestamp(),
                severity=sev,
                fixed_files=[f for f in fx["files"] if f in files] if fx else [],
                fix_sha=fx["sha"] if fx else None,
                bug_sha=None,
                group=None,
                duplicate_of=None,
            ))
        page += 1

    LOG.info("Loaded %d issues and %d commits from %s", len(issues), len(commits), repo)
    return Corpus(files, commits, issues, repo_dir=d)


# ---------------------------------------------------------------------------
# Load corpus and split
# ---------------------------------------------------------------------------

if CFG["MODE"] == "synthetic":
    corpus = load_synthetic(CFG["SEED"])
else:
    corpus = load_github(
        CFG["GITHUB_REPO"],
        CFG["GITHUB_TOKEN"],
        CFG["MAX_ISSUES"],
        CFG["MAX_COMMITS"],
    )

FIX_SHAS = {i["fix_sha"] for i in corpus.issues if i.get("fix_sha")}
split = int(0.6 * len(corpus.issues))
train_issues = corpus.issues[:split]
test_issues = corpus.issues[split:]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

from qdrant_client import QdrantClient, models  # noqa: E402

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hashing").lower()

if EMBEDDING_BACKEND == "sentence-transformers":
    from sentence_transformers import SentenceTransformer  # noqa: E402
    embedder = SentenceTransformer(CFG["EMBED_MODEL"])
    BGE_Q = (
        "Represent this sentence for searching relevant passages: "
        if "bge" in CFG["EMBED_MODEL"].lower()
        else ""
    )

    def embed(texts, query=False):
        prefix = BGE_Q if query else ""
        return np.asarray(
            embedder.encode(
                [prefix + t for t in texts],
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

else:
    from sklearn.feature_extraction.text import HashingVectorizer  # noqa: E402
    embedder = HashingVectorizer(
        n_features=384,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )

    def embed(texts, query=False):  # noqa: F811
        return np.asarray(embedder.transform(texts).toarray(), dtype="float32")


# ---------------------------------------------------------------------------
# Qdrant client and helpers
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdr = (
    QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if QDRANT_URL
    else QdrantClient(":memory:")
)

DIM = embed(["probe"]).shape[1]


def make_collection(name):
    """Recreate a Qdrant collection, dropping it first if it already exists."""
    if qdr.collection_exists(name):
        qdr.delete_collection(name)
    qdr.create_collection(
        name,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    )


def upsert(name, vecs, payloads, start=0, batch_size=256):
    """Batch-upsert vectors into *name* collection."""
    for i in range(0, len(vecs), batch_size):
        chunk_vecs = vecs[i:i + batch_size]
        chunk_pay = payloads[i:i + batch_size]
        qdr.upsert(
            name,
            points=[
                models.PointStruct(
                    id=start + i + j,
                    vector=v.tolist(),
                    payload=p,
                )
                for j, (v, p) in enumerate(zip(chunk_vecs, chunk_pay))
            ],
        )


def issue_text(issue):
    """Return a single string representation of an issue for embedding."""
    return f"{issue['title']}\n{issue.get('body', '')}"[:2000]


# ---------------------------------------------------------------------------
# Issue store (deduplication)
# ---------------------------------------------------------------------------

class IssueStore:
    """Thin Qdrant wrapper that tracks the live issue backlog for dedup."""

    def __init__(self, name="issues"):
        self.name = name
        if not qdr.collection_exists(name):
            make_collection(name)
        self.n = qdr.count(name, exact=True).count

    def add(self, issue):
        upsert(
            self.name,
            embed([issue_text(issue)]),
            [dict(issue_id=issue["id"], title=issue["title"])],
            start=self.n,
        )
        self.n += 1

    def add_many(self, issues):
        if not issues:
            return
        upsert(
            self.name,
            embed([issue_text(i) for i in issues]),
            [dict(issue_id=i["id"], title=i["title"]) for i in issues],
            start=self.n,
        )
        self.n += len(issues)

    def nearest(self, issue, k=3):
        if self.n == 0:
            return []
        try:
            return qdr.query_points(
                self.name,
                query=embed([issue_text(issue)])[0].tolist(),
                limit=k,
            ).points
        except Exception:
            LOG.warning("Qdrant nearest-neighbor query failed", exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Severity classifier
# ---------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression  # noqa: E402

SEV = ["low", "medium", "high", "critical"]
CRIT_RE = re.compile(
    r"\b(security|data loss|corrupt\w*|auth(?:entication)? bypass"
    r"|vulnerab\w*|rce|segfault|outage)\b",
    re.I,
)


class SeverityModel:
    """Logistic-regression severity classifier with keyword fallback."""

    def __init__(self):
        self.clf = None

    def fit(self, issues):
        labeled = [i for i in issues if i.get("severity") in SEV]
        if len(labeled) >= 20 and len({i["severity"] for i in labeled}) > 1:
            self.clf = LogisticRegression(
                max_iter=3000, C=4, class_weight="balanced"
            ).fit(
                embed([issue_text(i) for i in labeled]),
                [i["severity"] for i in labeled],
            )
        return self

    def predict(self, issue):
        text = issue_text(issue)
        if CRIT_RE.search(text):
            return "critical"
        if self.clf is None:
            return (
                "high"
                if re.search(r"regression|traceback|exception|crash|fails?", text, re.I)
                else "medium"
            )
        return self.clf.predict(embed([text]))[0]


sev_model = SeverityModel().fit(train_issues)


# ---------------------------------------------------------------------------
# DUP_THRESHOLD auto-calibration
# ---------------------------------------------------------------------------

def _calibrate_dup_threshold(issues, thresholds=None):
    """
    Sweep similarity thresholds on labeled duplicate pairs and pick the one
    that maximises pair-level F1.  Only runs when ground-truth labels exist.
    Returns the best threshold found (or the configured default).
    """
    if thresholds is None:
        thresholds = [round(0.60 + 0.02 * i, 2) for i in range(20)]  # 0.60 – 0.98

    labeled = [i for i in issues if i.get("duplicate_of") is not None]
    if len(labeled) < 10:
        LOG.debug("Not enough labeled duplicates for threshold calibration; using default.")
        return CFG["DUP_THRESHOLD"]

    # Build a quick lookup: issue_id → embedding index
    vecs = embed([issue_text(i) for i in issues])
    idx_by_id = {i["id"]: n for n, i in enumerate(issues)}

    best_thr, best_f1 = CFG["DUP_THRESHOLD"], 0.0
    for thr in thresholds:
        tp = fp = fn = 0
        for issue in labeled:
            canon_id = issue["duplicate_of"]
            if canon_id not in idx_by_id:
                continue
            q_vec = vecs[idx_by_id[issue["id"]]]
            c_vec = vecs[idx_by_id[canon_id]]
            # Cosine similarity (vectors are already L2-normalised by embed())
            sim = float(np.dot(q_vec, c_vec))
            predicted_dup = sim >= thr
            if predicted_dup:
                tp += 1
            else:
                fn += 1
        # False positives: non-dup pairs scored above threshold
        non_dup = [i for i in issues if i.get("duplicate_of") is None and i["id"] in idx_by_id]
        for issue in non_dup[:200]:  # sample to keep calibration fast
            q_vec = vecs[idx_by_id[issue["id"]]]
            for other in issues[:50]:
                if other["id"] == issue["id"]:
                    continue
                c_vec = vecs[idx_by_id[other["id"]]] if other["id"] in idx_by_id else None
                if c_vec is not None and float(np.dot(q_vec, c_vec)) >= thr:
                    fp += 1
                    break  # one false positive per issue is enough
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    LOG.info("DUP_THRESHOLD calibrated to %.2f (F1=%.3f)", best_thr, best_f1)
    return best_thr


# Run calibration only if env var is not explicitly set, and ground truth exists
if not os.getenv("DUP_THRESHOLD"):
    CFG["DUP_THRESHOLD"] = _calibrate_dup_threshold(train_issues)


# ---------------------------------------------------------------------------
# Code indexing
# ---------------------------------------------------------------------------

from rank_bm25 import BM25Okapi  # noqa: E402


def tokenize(text):
    """Tokenise *text* for BM25, splitting camelCase and lowercasing."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [
        w.lower()
        for w in re.findall(r"[A-Za-z]+|\d+", text.replace("_", " "))
        if len(w) > 1
    ]


def chunk_lines(text, size=40, stride=30):
    """Split *text* into overlapping windows of *size* lines."""
    lines = text.split("\n")
    chunks = [
        "\n".join(lines[i:i + size])
        for i in range(0, max(len(lines) - size + stride, 1), stride)
    ]
    return chunks or [text]


def extract_trace_paths(text, known):
    """Extract file paths from Python tracebacks and match them against *known*."""
    cands = (
        set(re.findall(r'File "([^"]+)"', text))
        | set(re.findall(
            r"([\w./-]+\.(?:py|js|ts|tsx|java|go|rs|cpp|cc|c|h|rb|php))", text
        ))
    )
    out = []
    for cand in cands:
        cand = cand.lstrip("./")
        out += [f for f in known if f == cand or f.endswith("/" + cand)]
    return list(dict.fromkeys(out))


class CodeIndex:
    """Hybrid BM25 + dense code search index with RRF fusion."""

    def __init__(self, corpus, exclude=frozenset()):
        self.corpus = corpus
        self.chunks = []

        # Build per-file commit-message history (excluding fix commits)
        hist = defaultdict(list)
        for c in corpus.commits:
            if c["sha"] in exclude:
                continue
            for f in c["files"]:
                hist[f].append(c["message"])

        # Create text chunks for every source file
        for path, text in corpus.files.items():
            for ch in chunk_lines(text):
                self.chunks.append(dict(file=path, kind="code", text=f"{path}\n{ch}"))
            if hist.get(path):
                history_text = f"{path}\n" + "\n".join(hist[path][:60])
                self.chunks.append(dict(file=path, kind="history", text=history_text))

        # BM25 index
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

        # Dense index in Qdrant
        vecs = embed([c["text"][:2000] for c in self.chunks])
        make_collection("code")
        upsert("code", vecs, [dict(file=c["file"], kind=c["kind"]) for c in self.chunks])

        LOG.info("CodeIndex built: %d chunks from %d files", len(self.chunks), len(corpus.files))

    def search(self, query, k=5, mode="hybrid", use_trace=True):
        """Return the top-k (file, score) pairs using BM25, dense, or hybrid RRF."""
        scores = defaultdict(float)

        def fuse(ranks):
            for r, idx in enumerate(ranks):
                scores[int(idx)] += 1 / (CFG["RRF_K"] + r + 1)

        if mode in ("bm25", "hybrid"):
            fuse(np.argsort(-self.bm25.get_scores(tokenize(query)))[:100])

        if mode in ("dense", "hybrid"):
            try:
                hits = qdr.query_points(
                    "code",
                    query=embed([query[:2000]], query=True)[0].tolist(),
                    limit=100,
                ).points
                fuse([h.id for h in hits])
            except Exception:
                LOG.warning("Qdrant dense search failed; falling back to BM25 only", exc_info=True)

        # Aggregate to file level (take max chunk score per file)
        fs = defaultdict(float)
        for idx, s in scores.items():
            f = self.chunks[idx]["file"]
            fs[f] = max(fs[f], s)

        # Boost files mentioned in stack traces
        if use_trace and mode == "hybrid":
            for p in extract_trace_paths(query, self.corpus.files):
                fs[p] += CFG["TRACE_BOOST"]

        return sorted(fs.items(), key=lambda x: -x[1])[:k]


code_index = CodeIndex(corpus, exclude=FIX_SHAS)


# ---------------------------------------------------------------------------
# Commit vector cache (bounded LRU to avoid unbounded memory growth)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _cvec(sha):
    """Return the embedding for commit *sha*, cached with an LRU of 4 096 entries."""
    c = corpus.by_sha[sha]
    return embed([f"{c['message']}\n{corpus.diff(sha)[:1500]}"])[0]


# ---------------------------------------------------------------------------
# Blame and assignee suggestion
# ---------------------------------------------------------------------------

def suspect_commits(issue, files, topn=3, max_cands=80):
    """Return the top-*topn* commits most likely to have introduced the bug."""
    before = issue["created"]
    cands = [
        c for c in corpus.commits
        if c["sha"] not in FIX_SHAS
        and c["ts"] < before
        and set(c["files"]) & set(files)
    ]
    cands = sorted(cands, key=lambda c: -c["ts"])[:max_cands]
    if not cands:
        return []

    q = embed([issue_text(issue)], query=True)[0]
    scored = []
    for c in cands:
        age = (before - c["ts"]) / 86400
        similarity = float(q @ _cvec(c["sha"]))
        recency_bonus = 0.15 * math.exp(-age / 90)
        scored.append(dict(
            sha=c["sha"],
            author=c["author"],
            message=c["message"],
            file_hits=sorted(set(c["files"]) & set(files)),
            score=similarity + recency_bonus,
        ))

    return sorted(scored, key=lambda x: -x["score"])[:topn]


def suggest_assignees(issue, files, suspects, topn=3, half_life_days=90):
    """Rank contributors by recency-weighted ownership of the affected files."""
    sc = defaultdict(float)
    for c in corpus.commits:
        if c["sha"] in FIX_SHAS:
            continue
        if c["ts"] >= issue["created"]:
            continue
        if not (set(c["files"]) & set(files)):
            continue
        decay = math.exp(-(issue["created"] - c["ts"]) / (half_life_days * 86400))
        sc[c["author"]] += decay

    # Give the top suspect's author an extra nudge
    if suspects:
        sc[suspects[0]["author"]] += 2.0

    return [a for a, _ in sorted(sc.items(), key=lambda x: -x[1])[:topn]]


# ---------------------------------------------------------------------------
# LLM repro-test generation
# ---------------------------------------------------------------------------

# Thread-safety: _hf_pipe is only written under the pipeline_lock in app.py
_hf_pipe = None
_hf_pipe_lock = threading.Lock()


def llm_generate(system, user, max_new_tokens=700):
    """
    Call the configured LLM backend and return the generated text, or None
    if the backend is 'stub'.
    """
    global _hf_pipe
    backend = CFG["LLM_BACKEND"]

    if backend == "hf":
        with _hf_pipe_lock:
            if _hf_pipe is None:
                from transformers import pipeline as hf_pipeline  # noqa: E402
                _hf_pipe = hf_pipeline(
                    "text-generation",
                    model=CFG["HF_MODEL"],
                    device_map="auto",
                    torch_dtype="auto",
                )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return _hf_pipe(
            msgs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )[0]["generated_text"][-1]["content"]

    if backend == "openai_compat":
        from openai import OpenAI  # noqa: E402
        client = OpenAI(
            base_url=CFG["OPENAI_BASE_URL"],
            api_key=os.getenv("OPENAI_API_KEY", "none"),
        )
        resp = client.chat.completions.create(
            model=CFG["OPENAI_MODEL"],
            temperature=0,
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    # "stub" — no LLM; template is generated in n_repro
    return None


def stub_repro(issue, path, text):
    """Generate a syntactically valid (but hollow) pytest stub."""
    funcs = re.findall(r"^def (\w+)\(", text, re.M) or ["target_function"]
    mod = path[:-3].replace("/", ".") if path.endswith(".py") else path
    return (
        f"```python\n"
        f"import pytest\n"
        f"from {mod} import {funcs[0]}\n\n"
        f"def test_regression_issue_{issue['id']}():\n"
        f'    """Repro for: {issue["title"][:80].replace(chr(34), chr(39))}"""\n'
        f"    ctx = {{\"state\": {{}}}}\n"
        f"    result = {funcs[0]}(ctx)\n"
        f"    # TODO: replace with the expected behaviour described in the report\n"
        f"    assert result is not None\n"
        f"```"
    )


REPRO_SYS = (
    "You are a senior engineer. Write ONE minimal, deterministic pytest test that "
    "reproduces the reported bug against the given source file. The test must FAIL "
    "on the buggy behaviour. Import only from the repo and the standard library. "
    "Output only a single ```python code block."
)


def extract_code(txt):
    """Extract the first Python code block from an LLM response."""
    m = re.search(r"```(?:python)?\n(.*?)```", txt or "", re.S)
    return (m.group(1) if m else (txt or "")).strip()


def validate_test(code):
    """Return (ok, error_message) after checking syntax and test-function presence."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    has_test_fn = any(
        isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        for n in ast.walk(tree)
    )
    if not has_test_fn:
        return False, "no test_* function found"
    return True, ""


# ---------------------------------------------------------------------------
# LangGraph state machine
# ---------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END  # noqa: E402


class TriageState(TypedDict, total=False):
    issue: dict
    severity: str
    duplicate: Optional[dict]
    files: list
    suspects: list
    assignees: list
    repro_test: str
    repro_ok: bool
    attempts: int
    error: str
    report: str
    log: Annotated[list, operator.add]


# Populate the issue store with training issues on first startup
store = IssueStore("issues_live")
if store.n == 0:
    store.add_many(train_issues)


def n_classify(s):
    sev = sev_model.predict(s["issue"])
    LOG.debug("classify: issue=%s severity=%s", s["issue"].get("id"), sev)
    return dict(severity=sev, log=[f"severity={sev}"])


def n_dedup(s):
    hits = store.nearest(s["issue"], 3)
    top = hits[0] if hits else None
    if top and top.score >= CFG["DUP_THRESHOLD"]:
        dup = dict(
            issue_id=top.payload["issue_id"],
            title=top.payload["title"],
            score=round(top.score, 3),
        )
    else:
        dup = None
    store.add(s["issue"])
    LOG.debug("dedup: issue=%s duplicate=%s", s["issue"].get("id"), dup)
    return dict(duplicate=dup, log=[f"duplicate={dup['issue_id'] if dup else None}"])


def route_after_dedup(s):
    return "report" if s.get("duplicate") else "localize"


def n_localize(s):
    query = issue_text(s["issue"])
    results = code_index.search(query, k=CFG["TOP_K"])
    files = [dict(path=f, score=round(sc, 4)) for f, sc in results]
    top_file = results[0][0] if results else None
    LOG.debug("localize: top_file=%s", top_file)
    return dict(files=files, log=[f"top file={top_file}"])


def n_blame(s):
    suspects = suspect_commits(s["issue"], [f["path"] for f in s["files"][:3]])
    top_sha = suspects[0]["sha"][:8] if suspects else None
    LOG.debug("blame: suspect=%s", top_sha)
    return dict(suspects=suspects, log=[f"suspect={top_sha}"])


def n_assign(s):
    assignees = suggest_assignees(
        s["issue"], [f["path"] for f in s["files"][:3]], s.get("suspects", [])
    )
    LOG.debug("assign: assignees=%s", assignees)
    return dict(assignees=assignees, log=[f"assignees={assignees}"])


def n_repro(s):
    top_file = s["files"][0]["path"]
    source_text = corpus.files[top_file]
    suspect = s["suspects"][0] if s.get("suspects") else None

    user_prompt = (
        f"BUG REPORT\nTitle: {s['issue']['title']}\n{s['issue']['body']}\n\n"
        f"SOURCE FILE: {top_file}\n{source_text[:6000]}\n\n"
    )
    if suspect:
        user_prompt += (
            f"SUSPECT COMMIT {suspect['sha'][:8]} ({suspect['message']}):\n"
            f"{corpus.diff(suspect['sha'])[:2000]}\n"
        )
    if s.get("error"):
        user_prompt += f"\nYour previous attempt failed validation: {s['error']}. Fix it.\n"

    raw = llm_generate(REPRO_SYS, user_prompt) or stub_repro(s["issue"], top_file, source_text)
    attempt = s.get("attempts", 0) + 1
    LOG.debug("repro: attempt=%d", attempt)
    return dict(repro_test=extract_code(raw), attempts=attempt, log=[f"repro attempt {attempt}"])


def n_validate(s):
    ok, err = validate_test(s["repro_test"])
    LOG.debug("validate: ok=%s error=%s", ok, err)
    return dict(repro_ok=ok, error=err, log=[f"repro valid={ok}"])


def route_after_validate(s):
    return "report" if s["repro_ok"] or s["attempts"] >= 3 else "repro"


def n_report(s):
    issue = s["issue"]
    lines = [
        f"### 🤖 Automated triage for #{issue['id']}",
        f"**Severity:** `{s.get('severity')}`",
    ]

    if s.get("duplicate"):
        d = s["duplicate"]
        lines.append(
            f"**Likely duplicate of** #{d['issue_id']} — _{d['title']}_ "
            f"(similarity {d['score']})"
        )
    else:
        lines.append(
            "**Top suspect files:** "
            + ", ".join(f"`{f['path']}`" for f in s.get("files", []))
        )
        if s.get("suspects"):
            x = s["suspects"][0]
            lines.append(
                f"**Suspect commit:** `{x['sha'][:8]}` by {x['author']} — {x['message']}"
            )
        lines.append(
            "**Suggested assignees:** "
            + ", ".join(f"@{a}" for a in s.get("assignees", []))
        )
        if s.get("repro_test"):
            lines.append(
                "<details><summary>Generated repro test</summary>\n\n"
                "```python\n"
                + s["repro_test"]
                + "\n```\n</details>"
            )

    return dict(report="\n\n".join(lines))


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_graph = StateGraph(TriageState)
for _name, _fn in [
    ("classify", n_classify),
    ("dedup", n_dedup),
    ("localize", n_localize),
    ("blame", n_blame),
    ("assign", n_assign),
    ("repro", n_repro),
    ("validate", n_validate),
    ("report", n_report),
]:
    _graph.add_node(_name, _fn)

_graph.add_edge(START, "classify")
_graph.add_edge("classify", "dedup")
_graph.add_conditional_edges(
    "dedup", route_after_dedup, {"report": "report", "localize": "localize"}
)
_graph.add_edge("localize", "blame")
_graph.add_edge("blame", "assign")
_graph.add_edge("assign", "repro")
_graph.add_edge("repro", "validate")
_graph.add_conditional_edges(
    "validate", route_after_validate, {"report": "report", "repro": "repro"}
)
_graph.add_edge("report", END)

triage_app = _graph.compile()


def triage_issue(issue):
    """Entry point: run the full triage pipeline for one issue dict."""
    return triage_app.invoke({"issue": issue, "attempts": 0, "log": []})
