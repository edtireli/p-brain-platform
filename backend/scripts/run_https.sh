#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/gen_dev_cert.sh

# Optional: set PBRAIN_MAIN_PY before running if needed.
exec uvicorn app:app \
  --host 127.0.0.1 \
  --port 8787 \
  --reload \
  --ssl-keyfile certs/dev.key \
  --ssl-certfile certs/dev.crt
