import { backendConfigured, BackendEngineAPI } from '@/lib/backend-engine';
import { SupabaseEngineAPI } from '@/lib/supabase-engine';

const useBackend = typeof window !== 'undefined' && backendConfigured();

// Backwards-compatible: existing UI imports expect `mockEngine`.
// Prefer the local backend when configured (enables real pipeline + volume IO).
export const mockEngine = useBackend ? new BackendEngineAPI() : new SupabaseEngineAPI();

// Backwards-compatible exports (kept to avoid breaking any stale imports).
export const engineKind = (useBackend ? 'backend' : 'supabase') as const;
export const isBackendEngine = useBackend;
