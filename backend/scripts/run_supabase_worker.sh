#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
	set -a
	source .env
	set +a
fi

missing=()
for v in SUPABASE_URL SUPABASE_SERVICE_ROLE_KEY PBRAIN_STORAGE_ROOT PBRAIN_MAIN_PY; do
	if [[ -z "${(P)v:-}" ]]; then
		missing+=("$v")
	fi
done

if (( ${#missing[@]} > 0 )); then
	echo "[run_supabase_worker] Missing required env vars: ${missing[*]}" >&2
	echo "[run_supabase_worker] Create backend/.env (see backend/.env.example) or export them in your shell." >&2
	exit 1
fi

PY="$PWD/../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
	PY="python3"
fi

"$PY" -m pip install -r requirements.txt

"$PY" supabase_worker.py
