import { BackendEngineAPI, backendConfigured } from '@/lib/backend-engine';
import { SupabaseEngineAPI } from '@/lib/supabase-engine';

// Prefer the real backend worker API when explicitly configured (via env or
// `?backend=https://...`), otherwise fall back to Supabase.
export const mockEngine = backendConfigured() ? new BackendEngineAPI() : new SupabaseEngineAPI();

// Backwards-compatible exports (not currently used elsewhere).
export const engineKind = backendConfigured() ? ('backend' as const) : ('supabase' as const);
export const isBackendEngine = backendConfigured();
