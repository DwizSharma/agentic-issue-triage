/* ─── Agentic Issue Triage — UI JavaScript ─────────────────────────────── */
'use strict';

// ── Quick-fill examples ────────────────────────────────────────────────────
const EXAMPLES = {
  auth: {
    number: 9001,
    title: 'Session cookie stays valid after logout',
    body: 'Security issue: logout does not invalidate the session cookie.\n\nSteps to reproduce:\n1. Log in\n2. Call /logout\n3. Reuse the old cookie — request succeeds\n\nAuth bypass confirmed on production.',
  },
  db: {
    number: 9002,
    title: 'Connection pool exhausted under load',
    body: 'All requests hang when traffic spikes above ~200 req/s.\nDatabase connections leak and the pool runs dry causing a full service outage.\n\nTraceback (most recent call last):\n  File "src/db/connection_pool.py", line 34, in acquire_connection\n    raise RuntimeError("unexpected state")',
  },
  invoice: {
    number: 9003,
    title: 'Invoice totals off by one cent for EU customers',
    body: 'Tax rounding error produces wrong invoice amount.\nCustomers are being overcharged by €0.01 on every EU transaction.\n\nBilling incident filed. Seen in production and staging.',
  },
  cache: {
    number: 9004,
    title: 'Cache returns stale values after TTL has elapsed',
    body: 'Expired cache entries are still served to users.\nTTL is set to 300s but outdated data is returned indefinitely.\n\nStarted after the last release. Cannot reproduce locally every time.',
  },
  email: {
    number: 9005,
    title: 'Emails sent twice when SMTP retry kicks in',
    body: 'Customers receive two copies of every notification email.\nDuplicate emails after SMTP timeout retry.\n\nCustomer escalated this via support. Please advise, blocking our deploy.',
  },
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const form         = document.getElementById('triage-form');
const btnSubmit    = document.getElementById('btn-submit');
const btnText      = btnSubmit.querySelector('.btn-text');
const btnSpinner   = btnSubmit.querySelector('.btn-spinner');

const emptyState   = document.getElementById('empty-state');
const progress     = document.getElementById('pipeline-progress');
const resultsEl    = document.getElementById('results-content');
const errorState   = document.getElementById('error-state');
const errorMsg     = document.getElementById('error-msg');

const badgeMode    = document.getElementById('badge-mode');
const badgeLLM     = document.getElementById('badge-llm');
const badgeStatus  = document.getElementById('badge-status');

const historyList  = document.getElementById('history-list');
const historyCount = document.getElementById('history-count');
const btnHistTog   = document.getElementById('btn-history-toggle');

// pipeline step elements
const STEPS = ['classify','dedup','localize','blame','assign','repro','report'];

// ── History store ──────────────────────────────────────────────────────────
let historyItems = [];
let lastResult   = null;

// ── Check server health on load ────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/health/ready');
    if (!r.ok) throw new Error('not ready');
    const d = await r.json();
    badgeStatus.textContent = '● ready';
    badgeStatus.className   = 'badge badge-status ready';
    badgeMode.textContent   = `mode: ${d.mode}`;
  } catch {
    badgeStatus.textContent = '● offline';
    badgeStatus.className   = 'badge badge-status error';
  }
}

async function checkLLMBadge() {
  try {
    const r = await fetch('/config');
    if (!r.ok) return;
    const d = await r.json();
    badgeLLM.textContent = `llm: ${d.llm_backend}${d.llm_backend === 'openai_compat' ? ' / ' + d.openai_model : ''}`;
  } catch { /* config endpoint optional */ }
}

checkHealth();
checkLLMBadge();

// ── Quick example buttons ──────────────────────────────────────────────────
document.querySelectorAll('.btn-example').forEach(btn => {
  btn.addEventListener('click', () => {
    const ex = EXAMPLES[btn.dataset.example];
    if (!ex) return;
    document.getElementById('issue-number').value = ex.number;
    document.getElementById('issue-title').value  = ex.title;
    document.getElementById('issue-body').value   = ex.body;
  });
});

// ── History toggle ─────────────────────────────────────────────────────────
btnHistTog.addEventListener('click', () => {
  historyList.classList.toggle('open');
});

// ── Form submit ────────────────────────────────────────────────────────────
form.addEventListener('submit', async e => {
  e.preventDefault();
  await runTriage();
});

async function runTriage() {
  const number  = parseInt(document.getElementById('issue-number').value, 10);
  const title   = document.getElementById('issue-title').value.trim();
  const body    = document.getElementById('issue-body').value.trim();
  const apiKey  = document.getElementById('api-key').value.trim();

  if (!title) return;

  // UI state — show progress
  setUIState('loading');
  animateSteps();

  try {
    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers['X-API-Key'] = apiKey;

    const res = await fetch('/triage', {
      method:  'POST',
      headers,
      body: JSON.stringify({ number, title, body }),
    });

    if (res.status === 401) throw new Error('Invalid or missing API key — paste your TRIAGE_API_KEY');
    if (res.status === 429) throw new Error('Rate limit hit — wait a minute and try again');
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    lastResult = { number, title, data };

    renderResults(number, title, data);
    addToHistory(number, title, data);
    setUIState('results');

  } catch (err) {
    errorMsg.textContent = err.message || 'Unknown error';
    setUIState('error');
  }
}

// ── UI state machine ───────────────────────────────────────────────────────
function setUIState(state) {
  emptyState.hidden   = state !== 'empty';
  progress.hidden     = state !== 'loading';
  resultsEl.hidden    = state !== 'results';
  errorState.hidden   = state !== 'error';

  const busy = state === 'loading';
  btnSubmit.disabled  = busy;
  btnText.hidden      = busy;
  btnSpinner.hidden   = !busy;
}

// ── Animate pipeline steps (fake progressive reveal) ──────────────────────
function animateSteps() {
  // reset
  STEPS.forEach(id => {
    const el = document.getElementById(`step-${id}`);
    el.className = 'step';
    el.querySelector('.step-icon').textContent = '○';
  });

  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) {
      const prev = document.getElementById(`step-${STEPS[i-1]}`);
      prev.className = 'step done';
    }
    if (i < STEPS.length) {
      const cur = document.getElementById(`step-${STEPS[i]}`);
      cur.className = 'step active';
      i++;
    } else {
      clearInterval(interval);
    }
  }, 280);
}

// ── Render full results ────────────────────────────────────────────────────
function renderResults(number, title, d) {
  // header
  document.getElementById('r-issue-num').textContent = `#${number}`;
  document.getElementById('r-title').textContent     = title;

  const sevEl = document.getElementById('r-severity');
  const sev   = (d.severity || 'unknown').toLowerCase();
  sevEl.textContent = sev.toUpperCase();
  sevEl.className   = `severity-badge sev-${sev}`;

  // duplicate banner
  const dupBanner = document.getElementById('r-dup-banner');
  if (d.duplicate) {
    document.getElementById('r-dup-id').textContent    = `#${d.duplicate.issue_id}`;
    document.getElementById('r-dup-title').textContent  = d.duplicate.title;
    document.getElementById('r-dup-score').textContent  = `(similarity: ${(d.duplicate.score * 100).toFixed(0)}%)`;
    dupBanner.hidden = false;
  } else {
    dupBanner.hidden = true;
  }

  // show/hide the detail cards (hidden when it's a duplicate)
  const cards = document.getElementById('r-cards');
  const cardCode   = document.getElementById('card-code');
  const cardRepro  = document.getElementById('card-repro');
  const cardReport = document.getElementById('card-report');

  if (d.duplicate) {
    cards.hidden    = true;
    cardCode.hidden = true;
    cardRepro.hidden = true;
  } else {
    cards.hidden = false;
    renderFiles(d.files   || []);
    renderCommit(d.suspects ? d.suspects[0] : null);
    renderAssignees(d.assignees || []);
    renderCodeBlock(d.files || [], d.suspects || []);
    renderReproTest(d.repro_test);
    cardCode.hidden  = false;
    cardRepro.hidden = false;
  }

  // full report always shown
  cardReport.hidden = false;
  document.getElementById('r-report').textContent = d.report || '';
}

// ── Files card ─────────────────────────────────────────────────────────────
function renderFiles(files) {
  const ul = document.getElementById('r-files');
  document.getElementById('r-files-count').textContent = files.length;
  ul.innerHTML = '';
  files.forEach((f, i) => {
    const li = document.createElement('li');
    const rank = document.createElement('span');
    rank.className = `file-rank${i === 0 ? ' rank-1' : ''}`;
    rank.textContent = i + 1;

    const name = document.createElement('span');
    name.className   = 'file-name';
    name.textContent = f.path;
    name.title       = f.path;

    const score = document.createElement('span');
    score.className   = 'file-score';
    score.textContent = f.score;

    li.append(rank, name, score);
    ul.appendChild(li);
  });
}

// ── Commit card ────────────────────────────────────────────────────────────
function renderCommit(commit) {
  const box = document.getElementById('r-commit');
  if (!commit) {
    box.innerHTML = '<span style="color:var(--text-muted);font-size:.82rem;padding:12px 0;display:block">No suspect commit found</span>';
    return;
  }
  const score = Math.min(commit.score, 1);
  box.innerHTML = `
    <div>
      <span class="commit-sha">${commit.sha.slice(0, 8)}</span>
    </div>
    <div class="commit-author">by ${commit.author}</div>
    <div class="commit-msg">"${commit.message}"</div>
    ${commit.file_hits && commit.file_hits.length ? `
      <div style="font-size:.75rem;color:var(--text-muted);margin-top:6px">
        Touched: ${commit.file_hits.map(f => `<code style="font-family:var(--mono);background:var(--bg);padding:1px 4px;border-radius:3px">${f}</code>`).join(', ')}
      </div>` : ''}
    <div class="commit-score">
      Confidence
      <div class="score-bar-wrap">
        <div class="score-bar" style="width:${Math.round(score * 100)}%"></div>
      </div>
      ${(score * 100).toFixed(0)}%
    </div>`;
}

// ── Assignees card ─────────────────────────────────────────────────────────
function renderAssignees(assignees) {
  const el = document.getElementById('r-assignees');
  el.innerHTML = '';
  if (!assignees.length) {
    el.innerHTML = '<span style="color:var(--text-muted);font-size:.82rem">No assignees found</span>';
    return;
  }
  assignees.forEach((a, i) => {
    const chip = document.createElement('div');
    chip.className = `assignee-chip${i === 0 ? ' assignee-rank-1' : ''}`;
    chip.innerHTML = `
      <div class="assignee-avatar">${a.slice(0,2)}</div>
      <span>@${a}</span>
      ${i === 0 ? '<span style="font-size:.65rem;color:var(--accent)">top pick</span>' : ''}`;
    el.appendChild(chip);
  });
}

// ── Source code block with bug highlighting ────────────────────────────────
function renderCodeBlock(files, suspects) {
  if (!files.length) {
    document.getElementById('card-code').hidden = true;
    return;
  }

  const topFile    = files[0].path;
  const issueTitle = document.getElementById('issue-title').value.trim();
  const issueBody  = document.getElementById('issue-body').value.trim();
  document.getElementById('r-code-path').textContent = topFile;

  // Fetch the source file from the server, passing issue context for bug-line hints
  const params = new URLSearchParams({
    path:        topFile,
    issue_title: issueTitle,
    issue_body:  issueBody.slice(0, 500),
  });
  fetch(`/source?${params}`)
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data || !data.source) {
        document.getElementById('card-code').hidden = true;
        return;
      }

      const source = data.source;
      const bugKeywords = data.bug_keywords || [];

      // Split into lines and mark suspect ones
      const lines = source.split('\n');
      const rendered = lines.map((line, idx) => {
        const lineNum = idx + 1;
        const isSuspect = bugKeywords.some(kw =>
          line.toLowerCase().includes(kw.toLowerCase())
        ) || (data.suspect_lines && data.suspect_lines.includes(lineNum));

        const escapedLine = line
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');

        if (isSuspect) {
          return `<span class="hljs-bug-line">${escapedLine}</span>`;
        }
        return escapedLine;
      }).join('\n');

      const codeEl = document.getElementById('r-code-block');
      codeEl.innerHTML = rendered;

      // Run highlight.js over the block (preserves our span markup)
      hljs.highlightElement(codeEl);

      // Scroll to first bug line
      const bugLine = codeEl.querySelector('.hljs-bug-line');
      if (bugLine) {
        setTimeout(() => bugLine.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
      }
    })
    .catch(() => {
      document.getElementById('card-code').hidden = true;
    });
}

// ── Repro test ─────────────────────────────────────────────────────────────
function renderReproTest(reproTest) {
  const cardRepro = document.getElementById('card-repro');
  const badgeSrc  = document.getElementById('r-repro-source');

  if (!reproTest) {
    cardRepro.hidden = true;
    return;
  }

  cardRepro.hidden = false;
  const codeEl = document.getElementById('r-repro-block');
  codeEl.textContent = reproTest;
  hljs.highlightElement(codeEl);

  // Detect if it was LLM-generated or a stub
  const isStub = reproTest.includes('# TODO: replace with the expected behaviour');
  badgeSrc.textContent = isStub ? 'stub template' : '✨ LLM generated';
  if (!isStub) {
    badgeSrc.style.background     = 'rgba(88,166,255,.15)';
    badgeSrc.style.borderColor    = 'var(--accent)';
    badgeSrc.style.color          = 'var(--accent)';
  }
}

// ── Copy buttons ───────────────────────────────────────────────────────────
function setupCopy(btnId, getContent) {
  document.getElementById(btnId).addEventListener('click', async () => {
    const text = getContent();
    if (!text) return;
    await navigator.clipboard.writeText(text).catch(() => {});
    const btn = document.getElementById(btnId);
    const orig = btn.textContent;
    btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
}

setupCopy('btn-copy-code',   () => document.getElementById('r-code-block').textContent);
setupCopy('btn-copy-repro',  () => document.getElementById('r-repro-block').textContent);
setupCopy('btn-copy-report', () => document.getElementById('r-report').textContent);

// ── History ────────────────────────────────────────────────────────────────
function addToHistory(number, title, data) {
  historyItems.unshift({ number, title, data });
  if (historyItems.length > 20) historyItems.pop();
  renderHistory();
}

function renderHistory() {
  historyCount.textContent = historyItems.length;
  historyList.innerHTML = '';
  historyItems.forEach((item, idx) => {
    const chip = document.createElement('div');
    chip.className = 'history-chip';
    const sev = (item.data.severity || '').toLowerCase();
    chip.innerHTML = `
      <span class="h-sev" style="color:var(--sev-${sev})">${sev || '?'}</span>
      <span>#${item.number} ${item.title.slice(0, 40)}${item.title.length > 40 ? '…' : ''}</span>`;
    chip.addEventListener('click', () => {
      document.getElementById('issue-number').value = item.number;
      document.getElementById('issue-title').value  = item.title;
      renderResults(item.number, item.title, item.data);
      setUIState('results');
      historyList.classList.remove('open');
    });
    historyList.appendChild(chip);
  });
}

// ── Initial state ──────────────────────────────────────────────────────────
setUIState('empty');
