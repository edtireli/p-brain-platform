import { BackendEngineAPI } from '@/lib/backend-engine';

// Backwards-compatible: existing UI imports expect `mockEngine`.
// Demo/engine switching has been removed; the app always uses the backend.
export const mockEngine = new BackendEngineAPI();

// Backwards-compatible exports (kept to avoid breaking any stale imports).
export const engineKind = 'backend' as const;
export const isBackendEngine = true;
