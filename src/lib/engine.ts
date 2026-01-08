import { BackendEngineAPI, backendConfigured, setBackendOverride } from '@/lib/backend-engine';

// Local-only app engine: UI talks to the bundled FastAPI backend.
export const engine = new BackendEngineAPI();

// Used by UI to gate features that require a direct backend HTTP engine.
export const isBackendEngine = true;

export function assertBackendConfigured(): void {
  if (!backendConfigured()) throw new Error('BACKEND_NOT_CONFIGURED');
}

export { backendConfigured, setBackendOverride };
