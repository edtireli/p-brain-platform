import { createClient } from '@supabase/supabase-js';

const supabaseUrl = (import.meta as any).env?.VITE_SUPABASE_URL as string | undefined;
const supabaseAnonKey = (import.meta as any).env?.VITE_SUPABASE_ANON_KEY as string | undefined;

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
  initSupabase();
  try {
    window.dispatchEvent(new CustomEvent(SUPABASE_REINIT_EVENT));
  } catch {
    /* ignore */
  }
}

export let supabase = null as ReturnType<typeof createClient> | null;

export function initSupabase(): void {
  if (!supabaseUrl || !supabaseAnonKey) {
    supabase = null;
    return;
  }

  const remember = readRememberMe();
  const storage = (() => {
    try {
      return remember ? window.localStorage : window.sessionStorage;
    } catch {
      return undefined;
    }
  })();

  supabase = createClient(supabaseUrl, supabaseAnonKey, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
      storage,
    },
  });
}

initSupabase();

export function supabaseConfigured(): boolean {
  return !!supabase;
}
