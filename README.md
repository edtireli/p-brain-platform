# p-brain-web

React UI + local FastAPI backend for running/inspecting `p-brain` outputs.

## Run (UI)

```zsh
cd /Users/edt/p-brain-web
npm install

# Local dev (defaults to http://127.0.0.1:8787)
npm run dev
```

## Backend URL

The frontend needs an API base URL:

- Set `VITE_API_BASE_URL` (preferred) or `VITE_BACKEND_URL` to a full `http(s)` URL.
- If you provide a value ending in `/api` (common convention), the UI will strip it.

Important: when the UI is served over **HTTPS** (e.g. GitHub Pages), it will **not** auto-fallback to localhost.
Use a proper public HTTPS backend URL.

## Run (backend)

```zsh
cd /Users/edt/p-brain-web
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Required so the backend can run the real pipeline
export PBRAIN_MAIN_PY=/Users/edt/Desktop/p-brain/main.py

uvicorn backend.app:app --host 127.0.0.1 --port 8787
```
