import { SupabaseEngineAPI } from '@/lib/supabase-engine';

// Supabase acts as the control plane; jobs are queued in Supabase and picked up by a local runner.
export const engine = new SupabaseEngineAPI();

export function assertBackendConfigured(): void {
  /* Supabase auth already gates the UI; nothing to assert here. */
}

// Kept for compatibility with callers that gate on backendConfigured(). Always true when Supabase is configured.
export function backendConfigured(): boolean {
  return true;
}

// No-op placeholder to satisfy callers that set overrides in backend mode.
export function setBackendOverride(_url: string): string | null {
  return null;
}
