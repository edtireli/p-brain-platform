// Local-only build: Supabase removed.

export const SUPABASE_REMEMBER_ME_KEY = 'pbrain.rememberMe';
export const SUPABASE_REINIT_EVENT = 'pbrain:supabase-reinit';

function readRememberMe(): boolean {
  try {
    const raw = window.localStorage.getItem(SUPABASE_REMEMBER_ME_KEY);
    if (raw === null) return true;
    return raw === 'true';
  } catch {
    return true;
  }
}

export function getRememberMe(): boolean {
  return readRememberMe();
}

export function setRememberMe(remember: boolean): void {
  try {
    window.localStorage.setItem(SUPABASE_REMEMBER_ME_KEY, remember ? 'true' : 'false');
  } catch {
    /* ignore */
  }
  try {
    window.dispatchEvent(new CustomEvent(SUPABASE_REINIT_EVENT));
  } catch {
    /* ignore */
  }
}

export const supabase = null as any;

export function supabaseConfigured(): boolean {
  return false;
}
