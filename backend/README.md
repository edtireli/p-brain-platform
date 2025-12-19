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
python3 -m pip install -r requirements.txt

# Tell the backend where p-brain lives (points to the CLI entry file)
export PBRAIN_MAIN_PY="/Users/edt/Desktop/p-brain/main.py"

# Start API
./scripts/run_http.sh

## Troubleshooting

- `net::ERR_CONNECTION_REFUSED` means the backend is not running (or it's bound to a different port/host).
- For local HTTPS mode, make sure you have accepted the cert at `https://127.0.0.1:8787/health`.
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

## Supabase control-plane runner (no tunnels, no exposed ports)

The UI (hosted or local) writes a job row; a **local runner** sitting next to the datasets polls, claims, runs, uploads logs to Supabase Storage, and writes events/outputs rows. Nothing is exposed publicly.

Schema additions (apply in Supabase SQL editor):
- `jobs` gains: `payload jsonb`, `runner_id text`, `claimed_at timestamptz`, `finished_at timestamptz`
- New tables: `job_events`, `job_outputs`
- Function: `claim_job(p_worker_id text)` does `SELECT ... FOR UPDATE SKIP LOCKED` to atomically claim a queued job.

Runner expectations:
- UI inserts into `jobs` with `status='queued'` and `payload` containing at least `relative_path` (relative to `PBRAIN_STORAGE_ROOT`) and optional `subject_id`.
- Runner uses Supabase **service role key** only (treat like a password).
- Runner reads data from the local filesystem and never opens ports or tunnels.

Runner env vars:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `PBRAIN_STORAGE_ROOT` (local root that matches your payload.relative_path values)
- `PBRAIN_MAIN_PY` (e.g. `/Users/edt/Desktop/p-brain/main.py`)
- Optional: `PBRAIN_PYTHON` (python with TF, etc.)
- Optional: `PBRAIN_STORAGE_BUCKET` (default `pbrain`)
- Optional: `PBRAIN_RUN_DIFFUSION=1` (default on)
- Optional: `PBRAIN_TURBO=1` (default on; suppresses plots)
- Optional: `PBRAIN_AI_DIR` (override model paths)
- Optional: `PBRAIN_WORKER_POLL_INTERVAL` (seconds, default 2.5)
- Optional: `PBRAIN_WORKER_ID` (defaults to hostname)
- Optional: `PBRAIN_WORKER_LOG_DIR` (where local logs are written)

How jobs flow:
1) UI inserts a job row: `status='queued'`, `payload={'relative_path': '20230928x1/subjectA', 'subject_id': 'subjectA'}`.
2) Runner calls `claim_job()` (FOR UPDATE SKIP LOCKED) and atomically flips to `running` + stamps `runner_id/claimed_at/start_time`.
3) Runner executes `p-brain/main.py --mode auto --data-dir <PBRAIN_STORAGE_ROOT>/<relative_path>` and captures stdout/stderr to a local log file.
4) Runner updates the job row (`status` -> `completed` or `failed`, sets `finished_at/end_time`).
5) Runner uploads the log to Supabase Storage at `jobs/<job_id>/logs/runner.log` and inserts a `job_outputs` row; `job_events` holds claim/start/error messages.

Run the worker:

```zsh
cd /Users/edt/p-brain-web/backend
python3 -m pip install -r requirements.txt

export SUPABASE_URL="https://..."
export SUPABASE_SERVICE_ROLE_KEY="..."
export PBRAIN_STORAGE_ROOT="/Volumes/T5_EVO_EDT/data"
export PBRAIN_MAIN_PY="/Users/edt/Desktop/p-brain/main.py"

./scripts/run_supabase_worker.sh
```

Tip: you can run multiple workers (different machines) pointing at the same Supabase project and the same dataset root; `claim_job` uses `FOR UPDATE SKIP LOCKED` so workers will not double-claim.
