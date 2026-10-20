# Git Ticket Bug Project

This repository has two ways to run the triage system:

- **Deployable API:** standalone FastAPI service in `app.py`, with PostgreSQL
  for triage records, Qdrant for vector search, API-key protection, and
  optional signed GitHub webhooks.
- **Notebook demo:** `agentic_issue_triage.ipynb` for exploring the synthetic
  workflow in VS Code.

## Start the deployment locally

Docker and Docker Compose are required. From this folder:

```bash
./deploy.sh
```

The script creates a private `.env` with random API and database secrets on
first run, builds the API image, starts PostgreSQL and Qdrant with persistent
volumes, and waits for the API health check. Compose keeps the service running
after terminal exit and restarts it after a machine reboot when Docker starts.
By default the API binds only to `127.0.0.1:8000`.

Check live and ready status:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

Send a triage request (read the generated API key from `.env`):

```bash
set -a; source .env; set +a
curl -s -X POST http://127.0.0.1:8000/triage \
  -H "X-API-Key: $TRIAGE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"number":9002,"title":"Session cookie stays valid after logout","body":"Security issue: logout does not invalidate the cookie."}'
```

The response includes severity, duplicate status, candidate files, suspect
commits, assignees, and a generated repro-test draft. With the default
`TRIAGE_MODE=synthetic`, results demonstrate the workflow only; they are not
real repository findings.

## Use a real GitHub repository

Edit `.env` and set:

```dotenv
TRIAGE_MODE=github
GITHUB_REPOSITORY=owner/repository
```

For private repositories or higher GitHub API limits, set a fine-grained
`GITHUB_TOKEN` with read access to repository contents and issues. Then run
`docker compose up -d --build --wait` again. The service clones the repository
and indexes its issue and commit history during startup, so first startup can
take several minutes. The checked-out source is cached in a persistent volume
and incrementally fetched on restarts. Keep `EMBEDDING_BACKEND=hashing` for a smaller deployment;
install the optional sentence-transformers dependency and choose
`EMBEDDING_BACKEND=sentence-transformers` to use BGE embeddings.

## GitHub webhooks and write-back

For automatic triage on newly opened issues, deploy the service behind an
HTTPS reverse proxy with a public URL. Set `API_BIND_ADDRESS=127.0.0.1` on the
host so only that proxy can reach the container port. Configure a GitHub
webhook for **Issues** events and set `GITHUB_WEBHOOK_ENABLED=true` and
`GITHUB_WEBHOOK_SECRET` in `.env`. The service verifies GitHub's HMAC signature
before processing payloads.

GitHub comments are disabled by default. To enable them, set
`GITHUB_WRITEBACK=true` and configure `GITHUB_TOKEN` with permission to write
issue comments. The API key protects direct `/triage` calls; GitHub webhook
requests use their webhook secret. Do not expose Qdrant or PostgreSQL ports to
the public internet.

## Manage the service

```bash
docker compose logs -f api
docker compose ps
docker compose down          # stop containers; keeps database and vector data
docker compose down -v       # permanently deletes persisted database/vector data
```

## Run the notebook in VS Code

See [RUN_IN_VSCODE.txt](RUN_IN_VSCODE.txt). The notebook starts its own
localhost-only demo API when all cells are run. Use the standalone Docker API
for deployment; do not run both APIs on port 8000 at the same time.
