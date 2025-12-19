// Legacy compatibility shim.
// Intentionally NO Supabase/Storage fallback: this app runs the real p-brain pipeline
// via an external backend/worker.

export { engine as mockEngine, backendConfigured as isBackendEngine } from '@/lib/engine';
export const engineKind = 'backend' as const;
