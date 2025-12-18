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
