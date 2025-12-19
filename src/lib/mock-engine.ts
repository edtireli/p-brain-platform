import { SupabaseEngineAPI } from '@/lib/supabase-engine';

// Backwards-compatible: existing UI imports expect `mockEngine`.
// The app now uses Supabase as its backend/data layer.
export const mockEngine = new SupabaseEngineAPI();

// Backwards-compatible exports (kept to avoid breaking any stale imports).
export const engineKind = 'supabase' as const;
export const isBackendEngine = false;
