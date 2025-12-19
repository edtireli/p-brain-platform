# p-brain-web

GitHub Pages UI for browsing p-brain projects/subjects via Supabase.

## Supabase (required)

This UI reads these build-time env vars:

- `VITE_SUPABASE_URL` (e.g. `https://<ref>.supabase.co`)
- `VITE_SUPABASE_ANON_KEY` (Supabase anon key)
- (optional) `VITE_SUPABASE_STORAGE_BUCKET` (defaults to `pbrain`)

Important: `VITE_API_BASE_URL` is **not** your Supabase anon key. It is only for an optional separate HTTP backend.

## Run (UI)

```zsh
cd /Users/edt/p-brain-web
npm install
npm run dev
```

## GitHub Pages deploy env

The workflow is in `.github/workflows/pages.yml` and expects:

- Repo variable or secret: `VITE_SUPABASE_URL`
- Repo secret: `VITE_SUPABASE_ANON_KEY`
- (optional) Repo variable/secret: `VITE_SUPABASE_STORAGE_BUCKET`

## Uploading artifacts for the UI (maps/montages/curves)

GitHub Pages cannot read files from `/Volumes/...`. To show real outputs, upload artifacts to Supabase Storage.

Helper script:

```zsh
cd /Users/edt/p-brain-web
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"

node scripts/sync-artifacts.mjs \
	--project <PROJECT_ID> \
	--subject-dir "/Volumes/T5_EVO_EDT/data/20230403x3" \
	--bucket pbrain
```

It uploads a small, web-friendly subset:

- `Images/AI/Montages/*.png` (QC montages)
- `Images/Fit/*.png` (fit plots shown as “maps”)
- a minimal `curves.json` derived from a few `.npy` curve files

## Seeding a test project (optional)

If you want a fully automated local seed (user + project + subject rows):

```zsh
cd /Users/edt/p-brain-web
export SUPABASE_URL="https://<ref>.supabase.co"

# Preferred: use the Supabase Dashboard "Secret key" (service_role equivalent) locally only.
export SUPABASE_SERVICE_ROLE_KEY="<sb_secret_...>"

node scripts/seed-supabase.mjs \
	--project-name "Test Project" \
	--subject-dir "/Volumes/T5_EVO_EDT/data/20230403x3" \
	--email "you@example.com" \
	--password "choose-a-password"
```

Notes:
- The key shown as `sb_publishable_...` is the browser-safe key (use it for `VITE_SUPABASE_ANON_KEY`).
- The key shown as `sb_secret_...` is admin-only (never put it in GitHub Pages / VITE_*).
