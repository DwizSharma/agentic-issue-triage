#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
command -v openssl >/dev/null || { echo "OpenSSL is required to create deployment secrets." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required for the deployment health check." >&2; exit 1; }
docker compose version >/dev/null
docker info >/dev/null 2>&1 || {
  echo "Docker is installed, but its daemon is not accessible. Start Docker and ensure your user can access it." >&2
  exit 1
}

if [[ ! -f .env ]]; then
  cp .env.example .env
  api_key="$(openssl rand -hex 32)"
  db_password="$(openssl rand -hex 24)"
  qdrant_api_key="$(openssl rand -hex 32)"
  sed -i "s|^TRIAGE_API_KEY=.*|TRIAGE_API_KEY=${api_key}|" .env
  sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${db_password}|" .env
  sed -i "s|^QDRANT_API_KEY=.*|QDRANT_API_KEY=${qdrant_api_key}|" .env
  chmod 600 .env
  echo "Created a private .env with generated API and database secrets."
fi

docker compose up -d --build --wait --wait-timeout 900
echo
docker compose ps
echo
echo "Service health:"
curl --fail --silent "http://127.0.0.1:${API_PORT:-8000}/health/ready"
echo
