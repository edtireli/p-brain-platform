#!/usr/bin/env bash
set -euo pipefail

# Builds the packaged backend bundle used by the Tauri launcher.
# Output: p-brain-web/tauri-launcher/src-tauri/resources/backend/pbrain-web-backend.zip

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_RES_BACKEND="$ROOT_DIR/../tauri-launcher/src-tauri/resources/backend"
SPEC_SRC="$ROOT_DIR/../tauri-launcher/src-tauri/resources/pyi-spec/pbrain-web-backend.spec"

if [[ ! -f "$SPEC_SRC" ]]; then
  echo "Missing PyInstaller spec: $SPEC_SRC" >&2
  exit 1
fi

python3 -m venv "$ROOT_DIR/.venv-build" >/dev/null 2>&1 || true
source "$ROOT_DIR/.venv-build/bin/activate"

python -m pip -q install --upgrade pip
python -m pip -q install -r "$ROOT_DIR/dev-requirements.txt"

WORKDIR="$ROOT_DIR/../tauri-launcher/src-tauri/resources/pyi-work"
DISTDIR="$WORKDIR/dist"
BUILDDIR="$WORKDIR/build"
mkdir -p "$WORKDIR" "$TAURI_RES_BACKEND"

# Use the canonical spec so hiddenimports/excludes stay in sync with the launcher.
pyinstaller -y --clean \
  --workpath "$BUILDDIR" \
  --distpath "$DISTDIR" \
  "$SPEC_SRC"

BUNDLE_DIR="$DISTDIR/pbrain-web-backend"
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "Expected bundle dir missing: $BUNDLE_DIR" >&2
  exit 1
fi

OUT_ZIP="$TAURI_RES_BACKEND/pbrain-web-backend.zip"
rm -f "$OUT_ZIP"

# Zip the onedir bundle.
( cd "$DISTDIR" && /usr/bin/zip -qr "$OUT_ZIP" "pbrain-web-backend" )

echo "Wrote: $OUT_ZIP"
