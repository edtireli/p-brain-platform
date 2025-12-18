# p-brain-web backend (local-only)

This is a **local** FastAPI server that the React UI talks to.

- No cloud services.
- No uploads (the server reads from local disk paths you provide).

## Run

```zsh
cd /Users/edt/p-brain-web/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tell the backend where p-brain lives (points to the CLI entry file)
export PBRAIN_MAIN_PY="/Users/edt/Desktop/p-brain/main.py"

# Start API
uvicorn app:app --host 127.0.0.1 --port 8787 --reload
```

## Frontend (backend mode)

In another terminal:

```zsh
cd /Users/edt/p-brain-web
VITE_ENGINE=backend VITE_BACKEND_URL=http://127.0.0.1:8787 npm run dev
```

Notes:
- For now, `project.storagePath` is treated as the **data root** containing subject folders.
- `Run Full Pipeline` triggers `p-brain/main.py --mode auto` for each subject.
