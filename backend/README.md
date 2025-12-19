# p-brain-web backend (local-only)

This is a **local** FastAPI server that the React UI talks to.

- No cloud services.
- No uploads (the server reads from local disk paths you provide).

## Run

## Recommended (no HTTPS, no cert prompts)

For day-to-day use (e.g. neuroscientists), do **not** use GitHub Pages.
Run the full app locally over plain HTTP so there are no TLS/certificate steps.

```zsh
cd /Users/edt/p-brain-web/backend

# Tell the backend where p-brain lives (points to the CLI entry file)
export PBRAIN_MAIN_PY="/Users/edt/Desktop/p-brain/main.py"

# Builds the frontend and starts the backend serving the UI at http://127.0.0.1:8787
./scripts/run_local_app.sh
```

```zsh
cd /Users/edt/p-brain-web/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tell the backend where p-brain lives (points to the CLI entry file)
export PBRAIN_MAIN_PY="/Users/edt/Desktop/p-brain/main.py"

# Start API
./scripts/run_http.sh

## Run (HTTPS, for GitHub Pages)

If you load the UI from GitHub Pages (HTTPS), browsers will block calls to an HTTP `localhost` API.
Run the backend over HTTPS instead:

```zsh
cd /Users/edt/p-brain-web/backend

# Generates a local self-signed cert (first run only), then starts uvicorn with TLS
./scripts/run_https.sh

# One-time browser step:
# open https://127.0.0.1:8787/health and accept the certificate warning
```
```

## Frontend

In another terminal:

```zsh
cd /Users/edt/p-brain-web
npm run dev
```

Notes:
- For now, `project.storagePath` is treated as the **data root** containing subject folders.
- `Run Full Pipeline` triggers `p-brain/main.py --mode auto` for each subject.

## Supabase worker (real pipeline runner)

If you want the **Supabase/GitHub Pages** UI to queue work and have a **real runner** execute the full p-brain pipeline, run the Supabase worker on a machine that can see the datasets on disk.

Requirements:
- The worker needs a Supabase **service role key** (so it can update `jobs` + `subjects` regardless of RLS).
- Each subject row must have `source_path` set to a path that exists on the worker machine (e.g. `/Volumes/T5_EVO_EDT/data/20250217x4`).
- The worker runs `p-brain/main.py --mode auto` (optionally with `--diffusion`).

Env vars:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `PBRAIN_MAIN_PY` (e.g. `/Users/edt/Desktop/p-brain/main.py`)
- Optional: `PBRAIN_PYTHON` (python executable that has all p-brain deps, e.g. TensorFlow)
- Optional: `PBRAIN_RUN_DIFFUSION=1` (default on)
- Optional: `PBRAIN_TURBO=1` (default on; suppresses plots)

Run:

```zsh
cd /Users/edt/p-brain-web/backend

export SUPABASE_URL="https://..."
export SUPABASE_SERVICE_ROLE_KEY="..."
export PBRAIN_MAIN_PY="/Users/edt/Desktop/p-brain/main.py"

./scripts/run_supabase_worker.sh
```
