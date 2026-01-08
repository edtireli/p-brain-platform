function sanitizeHttpUrl(raw: string | undefined | null): string | null {
  const s = String(raw || '').trim();
  if (!s) return null;
  if (!/^https?:\/\//i.test(s)) return null;
  const noSlash = s.replace(/\/+$/, '');
  return noSlash.endsWith('/api') ? noSlash.slice(0, -4) : noSlash;
}

export function getLocalBackendBaseUrl(): string | null {
  // Launchers (Electron/Tauri) can inject a concrete backend base URL.
  // This is the most reliable option when the UI isn't served by the backend.
  try {
    const injected = (globalThis as any)?.window?.__PBRAIN_BACKEND_URL;
    if (typeof injected === 'string' && injected.trim().length > 0) {
      return sanitizeHttpUrl(injected);
    }
  } catch {
    // ignore
  }

  const env = (import.meta as any).env as Record<string, string | undefined> | undefined;
  const fromEnv = (
    sanitizeHttpUrl(env?.VITE_LOCAL_BACKEND_URL) ||
    sanitizeHttpUrl(env?.VITE_API_BASE_URL) ||
    sanitizeHttpUrl(env?.VITE_BACKEND_URL) ||
    null
  );

  if (fromEnv) return fromEnv;

  // If the UI is being served by the local backend, default to same-origin.
  try {
    if (typeof window !== 'undefined') {
      const host = window.location.hostname;
      if (host === '127.0.0.1' || host === 'localhost') {
        // If we're on the backend-served UI (port 8787), same-origin is correct.
        // If we're on a separate dev server (e.g. Vite :5173), default to backend :8787.
        const port = String(window.location.port || '');
        if (port === '8787') return sanitizeHttpUrl(window.location.origin);
        return 'http://127.0.0.1:8787';
      }
    }
  } catch {
    // ignore
  }

  return null;
}

export async function checkLocalBackendHealth(timeoutMs: number = 1200): Promise<boolean> {
  const base = getLocalBackendBaseUrl();
  if (!base) return false;
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/health`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

export async function resolveStoragePathFromDrop(
  folderName: string,
  sampleSubject?: string
): Promise<string | null> {
  const base = getLocalBackendBaseUrl();
  if (!base) return null;

  const u = new URL('/local/resolve-storage-path', base);
  u.searchParams.set('folderName', folderName);
  if (sampleSubject) u.searchParams.set('sampleSubject', sampleSubject);
  u.searchParams.set('limit', '5');

  try {
    const res = await fetch(u.toString());
    if (!res.ok) return null;
    const json = await res.json();
    const candidates = Array.isArray(json?.candidates) ? json.candidates : [];
    const best = candidates[0];
    const path = typeof best?.path === 'string' ? best.path : null;
    return path && path.trim().length > 0 ? path.trim() : null;
  } catch {
    return null;
  }
}

export async function pickFolderWithNativeDialog(): Promise<string | null> {
  const base = getLocalBackendBaseUrl();
  if (!base) return null;
  try {
    // If running inside the packaged Electron launcher, use its native picker.
    const electronPicker = (globalThis as any)?.window?.pbrainLauncher?.pickFolder;
    if (typeof electronPicker === 'function') {
      const picked = await electronPicker();
      const path = typeof picked === 'string' ? picked : null;
      const trimmed = path && path.trim().length > 0 ? path.trim() : null;
      if (!trimmed) return null;
      // Tell the local backend to allow reads under this directory.
      try {
        await fetch(`${base}/local/allow-root`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ path: trimmed }),
        });
      } catch {
        // Ignore; UI will still show a helpful error if reads are blocked.
      }
      return trimmed;
    }

    // If running inside the Tauri launcher, use its native picker via IPC.
    const tauriInvoke = (globalThis as any)?.window?.__TAURI__?.core?.invoke;
    if (typeof tauriInvoke === 'function') {
      const picked = await tauriInvoke('pick_folder');
      const path = typeof picked === 'string' ? picked : null;
      const trimmed = path && path.trim().length > 0 ? path.trim() : null;
      if (!trimmed) return null;
      try {
        await fetch(`${base}/local/allow-root`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ path: trimmed }),
        });
      } catch {
        // Ignore; UI will still show a helpful error if reads are blocked.
      }
      return trimmed;
    }

    const res = await fetch(`${base}/local/pick-folder`, { method: 'POST' });
    if (!res.ok) return null;
    const json = await res.json();
    if (json?.cancelled) return null;
    const path = typeof json?.path === 'string' ? json.path : null;
    return path && path.trim().length > 0 ? path.trim() : null;
  } catch {
    return null;
  }
}
