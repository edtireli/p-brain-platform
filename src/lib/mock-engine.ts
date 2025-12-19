import { SupabaseEngineAPI } from '@/lib/supabase-engine';

// GitHub Pages + Supabase-only mode: never call a local backend.
export const mockEngine = new SupabaseEngineAPI();

// Backwards-compatible exports.
export const engineKind = 'supabase' as const;
export const isBackendEngine = false;
