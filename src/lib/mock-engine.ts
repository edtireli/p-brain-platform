// Legacy compatibility shim.
// Keep minimal exports so older components can still compile.

export { engine as mockEngine, isBackendEngine } from '@/lib/engine';
export const engineKind = 'supabase' as const;
