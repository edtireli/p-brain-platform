# p-brain Platform
<img width="2500" height="549" alt="pbrainplatform_banner" src="https://github.com/user-attachments/assets/a7ee49ac-b972-4e95-b3e1-d24b91c52a40" />

Cross-platform desktop app + web UI that turns the **p-brain** neuroimaging pipeline into a usable end-to-end product: project/subject management, job monitoring, and a fast QC/review workspace for **DCE-MRI** and **diffusion** outputs (including tractography).

Key links:
- p-brain (pipeline): https://github.com/edtireli/p-brain
- CNN bundle (Zenodo): https://doi.org/10.5281/zenodo.15655348
- Releases (macOS DMG): https://github.com/edtireli/p-brain-platform/releases

What this platform gives you:
- A single workspace to browse projects/subjects and review outputs.
- Fast QC for DCE maps/curves and segmentation-derived summaries.
- An interactive tractography viewer for diffusion outputs.
- A desktop app that bundles the UI and runs a local bridge for file access.

## Download (macOS)

We publish macOS builds as a **`.dmg`** in GitHub Releases.

- Download the latest `p-brain_*.dmg` from the Releases page.
- Open it, then drag `p-brain.app` into `Applications`.

Note: the `.app` is inside the DMG; you don’t need a separate `.app` download.

## Open source

This project is **open source** to encourage collaboration and make the platform better over time. That said, the primary path for most users is simply: **download the latest DMG and use the app**.

## What p-brain does (pipeline)

p-brain takes raw DCE-MRI (and optional diffusion MRI) and produces quantitative neuroimaging outputs with transparent QC artifacts:

- **Input functions (AIF/VIF)**: CNN-based rICA + SSS slice detection/ROI extraction.
- **Pharmacokinetics**: Patlak permeability model (Ki, vp) and Extended Tofts (Ktrans, kep, ve).
- **Perfusion/deconvolution**: model-free residue deconvolution outputs including CBF, MTT, and CTH.
- **Anatomy / parcellation**: FastSurfer-style segmentations/parcels propagated to DCE space for parcel-wise summaries.
- **Deliverables**: voxel-wise NIfTI maps, parcel tables, curve plots, fit diagnostics, and montages for QC.

## What’s inside

- **Web UI (Vite/React)** – Supabase-backed subject/projects browser, tractography viewer, QC overlays, and pipeline status.
- **Desktop app (Tauri)** – Bundles the UI with a local FastAPI bridge and Python environment bootstrap.
- **Backend bridge (FastAPI)** – Serves tractography streamlines, AI outputs, and local file access for the app.
- **p-brain pipeline hooks** – Calls out to the Python pipeline (segmentation, diffusion, tractography, AIF, CNN inference) and surfaces results for review.

## Image gallery

The following screenshots demonstrate the platform’s design and operability.

![Screenshot 2026-01-08 17:43:51](docs/images/platform/Screenshot%202026-01-08%20at%2017.43.51.png)
*Snapshot showing the platform dashboard with example project.*

![Screenshot 2026-01-08 17:45:21](docs/images/platform/Screenshot%202026-01-08%20at%2017.45.21.png)
*Tractography viewer.*

![Screenshot 2026-01-08 17:47:11](docs/images/platform/Screenshot%202026-01-08%20at%2017.47.11.png)
*Dynamic-contrast enhanced series with AI overlay of predicted AIF/VIF.*

![Screenshot 2026-01-08 17:45:41](docs/images/platform/Screenshot%202026-01-08%20at%2017.45.41.png)
*Display of a voxelwise map (Ki in this example).*

![Screenshot 2026-01-08 17:45:56](docs/images/platform/Screenshot%202026-01-08%20at%2017.45.56.png)
*Display of a parcelwise map (Ki in this example).*

![Screenshot 2026-01-08 17:46:02](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.02.png)
*Arterial input functions (AIF), venous input functions (VIF), and tissue functions displayed for a subject.*

![Screenshot 2026-01-08 17:46:20](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.20.png)
*Isolating desired functions for further inspection.*

![Screenshot 2026-01-08 17:46:35](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.35.png)
*Patlak analysis; dotted line indicates the window used for estimation.*

![Screenshot 2026-01-08 17:46:54](docs/images/platform/Screenshot%202026-01-08%20at%2017.46.54.png)
*Extended Tofts estimation.*

![Screenshot 2026-01-08 17:49:45](docs/images/platform/Screenshot%202026-01-08%20at%2017.49.45.png)
*Table of results from segmentation.*

![Screenshot 2026-01-08 17:47:18](docs/images/platform/Screenshot%202026-01-08%20at%2017.47.18.png)
*Additional visualization of imaging outputs.*

![Screenshot 2026-01-08 17:49:13](docs/images/platform/Screenshot%202026-01-08%20at%2017.49.13.png)
*Overview of subjects; blue indicates an active job; green indicates a completed job.*

## Requirements

- Node.js 18+ and npm
- Supabase project (URL + anon key) for remote data/metadata storage

For desktop builds:
- Rust toolchain (for Tauri)
- Python 3.10+ (pipeline + backend) with virtualenv
- Optional system deps for full pipeline (FSL/FreeSurfer if you run the native pipeline locally)

## Quick start: web UI only

```zsh
git clone https://github.com/edtireli/p-brain-platform.git
cd p-brain-platform
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
# macOS: outputs appear under src-tauri/target/release/bundle/macos/ (.app and .dmg)
```

The launcher bundles the UI, starts the FastAPI bridge, and can manage a Python virtualenv for the pipeline. After building, you can install locally (example):

```zsh
APP_SRC="$(pwd)/src-tauri/target/release/bundle/macos/p-brain.app"
rm -rf /Applications/p-brain.app && cp -R "$APP_SRC" /Applications/p-brain.app && xattr -dr com.apple.quarantine /Applications/p-brain.app
```

Find the most recent DMG produced by a build:

```zsh
ls -t src-tauri/target/release/bundle/macos/*.dmg | head -n 1
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
cd p-brain-platform
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
cd p-brain-platform
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<sb_secret_...>"  # local only

node scripts/seed-supabase.mjs \
  --project-name "Test Project" \
  --subject-dir "/Volumes/T5_EVO_EDT/data/20230403x3" \
  --email "you@example.com" \
  --password "choose-a-password"
```

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
