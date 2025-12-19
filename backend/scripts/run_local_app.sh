#!/usr/bin/env bash
set -euo pipefail

# Runs the full app locally over HTTP (no TLS/cert prompts):
# - builds the frontend (Vite -> dist/)
# - starts the FastAPI backend which serves dist/ at /

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"

cd "$ROOT_DIR"

if [[ ! -d "node_modules" ]]; then
  echo "Installing frontend dependencies (first run only)…"
  npm install
fi

echo "Building frontend…"
npm run build

echo "Starting backend (serving UI at http://127.0.0.1:8787)…"
(
  sleep 1
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:8787" || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:8787" || true
  fi
) &

cd "$BACKEND_DIR"
exec ./scripts/run_http.sh
