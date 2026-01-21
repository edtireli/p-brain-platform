import { BackendEngineAPI, backendConfigured, setBackendOverride } from '@/lib/backend-engine';

// Local-only app engine: UI talks to the bundled FastAPI backend.
export const engine = new BackendEngineAPI();

// Best-effort cleanup: prevent pending polling/fetches from blocking shutdown.
try {
  if (typeof window !== 'undefined') {
    const dispose = () => {
      try {
        (engine as any)?.dispose?.();
      } catch {
        // ignore
      }
    };
    window.addEventListener('pagehide', dispose);
    window.addEventListener('beforeunload', dispose);
  }
} catch {
  // ignore
}

// Used by UI to gate features that require a direct backend HTTP engine.
export const isBackendEngine = true;

export function assertBackendConfigured(): void {
  if (!backendConfigured()) throw new Error('BACKEND_NOT_CONFIGURED');
}

export { backendConfigured, setBackendOverride };
