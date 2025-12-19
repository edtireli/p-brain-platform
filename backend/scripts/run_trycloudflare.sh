#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install it, then re-run:" >&2
  echo "  brew install cloudflared" >&2
  exit 1
fi

echo "[backend] starting on http://127.0.0.1:8787" >&2
uvicorn app:app --host 127.0.0.1 --port 8787 &
backend_pid=$!

cleanup() {
  kill "$backend_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[tunnel] starting Cloudflare tunnel (you'll get a public https://... URL)" >&2
echo "[tunnel] copy that URL into the UI as the Backend URL" >&2
exec cloudflared tunnel \
  --url http://127.0.0.1:8787 \
  --no-http2-origin
