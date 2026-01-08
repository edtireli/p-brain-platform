# p-brain Launcher (macOS)

This folder packages a **double-click** macOS app that:

- starts the local FastAPI backend on `http://127.0.0.1:8787`
- serves the built UI from the backend
- opens an embedded app window (Electron)

## Developer build (macOS)

From `p-brain-web/launcher/`:

```zsh
npm install
npm run dist:mac
```

Output:
- `p-brain-web/launcher/release/*.dmg`

## Dev run

```zsh
npm install
npm run dev
```

Notes:
- `npm run dev` builds the Vite UI and launches Electron.
- The backend runs from source in dev; packaged builds bundle the backend via PyInstaller.
