#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pip install -r requirements.txt

python3 supabase_worker.py
