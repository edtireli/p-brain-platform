#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."

exec ./backend/scripts/run_local_app.sh
