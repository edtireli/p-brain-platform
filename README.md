# p-brain Platform (formerly p-brain-web)

Cross-platform UI and desktop launcher for the p-brain neuroimaging pipeline. It brings subject browsing, diffusion/tractography QC, AI-assisted workflows (AIF extraction + CNN lesion/slice models), and project provisioning into a single experience. A pre-trained CNN bundle is published on Zenodo: https://doi.org/10.5281/zenodo.15655348

## What’s inside

- **Web UI (Vite/React)** – Supabase-backed subject/projects browser, tractography viewer, QC overlays, and pipeline status.
- **Desktop launcher (Tauri)** – Ships the web UI with a local FastAPI bridge and Python environment bootstrap.
- **Backend bridge (FastAPI)** – Serves tractography streamlines, AI outputs, and local file access for the app.
- **p-brain pipeline hooks** – Calls out to the Python pipeline (segmentation, diffusion, tractography, AIF, CNN inference) and surfaces the results in the UI.

## Requirements

- Node.js 18+ and npm
- Rust toolchain (for Tauri desktop builds)
- Python 3.10+ (pipeline + backend) with virtualenv
- Supabase project (URL + anon key) for remote data/metadata storage
- Optional system deps for full pipeline (FSL/FreeSurfer if you run the native pipeline locally)

## Quick start: web UI only

```zsh
git clone https://github.com/edtireli/p-brain-web.git
cd p-brain-web
npm install
cp .env.example .env.local  # if present; otherwise set vars below
```

Set build-time env vars (e.g. in `.env.local`):
- `VITE_SUPABASE_URL` – `https://<ref>.supabase.co`
- `VITE_SUPABASE_ANON_KEY` – Supabase anon key
- `VITE_SUPABASE_STORAGE_BUCKET` – optional, defaults to `pbrain`
- `VITE_API_BASE_URL` – optional FastAPI backend (http://127.0.0.1:8787)

Run the UI:

```zsh
npm run dev
# open http://localhost:5173
```

## Backend (FastAPI bridge)

```zsh
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r dev-requirements.txt
# optional: export PBRAIN_MAIN_PY to point to your p-brain main.py if not co-located
uvicorn app:app --host 127.0.0.1 --port 8787 --reload
```

## Desktop app (Tauri)

```zsh
cd tauri-launcher
npm install
npm run tauri:build   # or npm run tauri:dev for live dev
# macOS: the built app appears under src-tauri/target/release/bundle/macos/
```

The launcher bundles the UI, starts the FastAPI bridge, and can manage a Python virtualenv for the pipeline. After building, you can install locally (example):

```zsh
APP_SRC="$(pwd)/src-tauri/target/release/bundle/macos/p-brain.app"
rm -rf /Applications/p-brain.app && cp -R "$APP_SRC" /Applications/p-brain.app && xattr -dr com.apple.quarantine /Applications/p-brain.app
```

## AI models (CNN)

- Download the latest CNN model bundle from Zenodo: https://doi.org/10.5281/zenodo.15655348
- Place the model file where your pipeline expects it (e.g., alongside other `.keras` models in your p-brain AI directory) and configure the path in your pipeline config if needed.

## Supabase deployment (GitHub Pages)

Workflow: `.github/workflows/pages.yml`
- Repo variable/secret: `VITE_SUPABASE_URL`
- Repo secret: `VITE_SUPABASE_ANON_KEY`
- Optional: `VITE_SUPABASE_STORAGE_BUCKET`

## Artifact upload helper

To publish QC artifacts to Supabase Storage for the UI:

```zsh
cd p-brain-web
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"

node scripts/sync-artifacts.mjs \
  --project <PROJECT_ID> \
  --subject-dir "/Volumes/T5_EVO_EDT/data/20230403x3" \
  --bucket pbrain
```

Uploads:
- `Images/AI/Montages/*.png` (QC montages)
- `Images/Fit/*.png` (fit plots shown as “maps”)
- `curves.json` distilled from select `.npy` curve files

## Seed a sample project (optional)

```zsh
cd p-brain-web
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<sb_secret_...>"  # local only

node scripts/seed-supabase.mjs \
  --project-name "Test Project" \
  --subject-dir "/Volumes/T5_EVO_EDT/data/20230403x3" \
  --email "you@example.com" \
  --password "choose-a-password"
```

## Image gallery (chronological)

Screenshots live in `docs/images/platform/` and are embedded below from earliest to latest capture time.

![Screenshot 2026-01-08 17:43:51](docs/images/platform/Screenshot%202026-01-08%20at%2017.43.51.png)
*Snapshot from an early session showing the platform dashboard and initial subject context.*

![Screenshot 2026-01-08 17:45:21](docs/images/platform/Screenshot%202026-01-08%20at%2017.45.21.png)
*Progressing through workflow setup with project/subject details visible.*

![Screenshot 2026-01-08 17:45:41](docs/images/platform/Screenshot%202026-01-08%20at%2017.45.41.png)
*QC/visual context for imaging data at this stage of the pipeline.*

![Screenshot 2026-01-08 17:45:56](docs/images/platform/Screenshot%202026-01-08%20at%2017.45.56.png)
*Demonstrates navigation across imaging outputs and related metadata.*

![Screenshot 2026-01-08 17:46:02](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.02.png)
*Shows the interface transitioning between modalities/QC views.*

![Screenshot 2026-01-08 17:46:20](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.20.png)
*Highlights tractography/visual overlays during review.*

![Screenshot 2026-01-08 17:46:35](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.35.png)
*Pipeline status and dataset context in the UI.*

![Screenshot 2026-01-08 17:46:54](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.54.png)
*Another QC-focused view to inspect outputs.*

![Screenshot 2026-01-08 17:47:11](docs/images/platform/Screenshot%202026-01-08%20at%2017.47.11.png)
*Demonstrates navigation and action controls for a subject.*

![Screenshot 2026-01-08 17:47:18](docs/images/platform/Screenshot%202026-01-08%20at%2017.47.18.png)
*Additional visualization of imaging outputs.*

![Screenshot 2026-01-08 17:49:13](docs/images/platform/Screenshot%202026-01-08%20at%2017.49.13.png)
*Late-stage review of results, including overlays.*

![Screenshot 2026-01-08 17:49:45](docs/images/platform/Screenshot%202026-01-08%20at%2017.49.45.png)
*Wrap-up view summarizing processed data and controls.*

## Contributing

Issues and PRs welcome. For local changes:

```zsh
git checkout -b feature/my-change
# edit + test
git add ...
git commit -m "Describe your change"
git push origin feature/my-change
```

## Notes

- Keep secrets out of `.env` committed files. Use `.env.local` for local dev.
- When shipping desktop builds, ensure required Python deps and model files are present in the managed virtualenv or packaged resources.
