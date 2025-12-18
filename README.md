# p-brain-web

React UI + local FastAPI backend for running/inspecting `p-brain` outputs.

## Demo vs Real Mode

The UI supports two modes:

- **Demo mode** (no backend): uses the in-browser mock engine.
- **Backend mode**: talks to the local FastAPI backend (runs/reads real `p-brain` outputs).

### Which mode is used?

- Default is controlled by `VITE_ENGINE` (`demo` or `backend`).
- You can force demo via URL params **only if demo is allowed**:
	- `?demo=1` or `?engine=demo`
	- Force backend with `?engine=backend`
- Demo can be disabled for “real users” with `VITE_ALLOW_DEMO=false`.

This lets you keep a public demo link (e.g. GitHub Pages), while your real deployment can hard-disable demo.

## Run (UI)

```zsh
cd /Users/edt/p-brain-web
npm install

# Demo mode
VITE_ENGINE=demo npm run dev

# Backend mode
VITE_ENGINE=backend VITE_BACKEND_URL=http://127.0.0.1:8787 npm run dev
```

### Demo link

- Add `?demo=1` to the UI URL.

### Disable demo for real users

- Build/run with `VITE_ALLOW_DEMO=false`.

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
