import { useEffect, useState } from 'react';
import type { Session, User } from '@supabase/supabase-js';
import { supabase, SUPABASE_REINIT_EVENT, SUPABASE_REMEMBER_ME_KEY } from '@/lib/supabase';

export type AuthState = {
  loading: boolean;
  configured: boolean;
  session: Session | null;
  user: User | null;
};

export function useSupabaseAuth(): AuthState {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    let unsubscribe: (() => void) | null = null;

    const setup = () => {
      if (unsubscribe) {
        unsubscribe();
        unsubscribe = null;
      }

      if (!supabase) {
        setSession(null);
        setLoading(false);
        return;
      }

      setLoading(true);
      supabase.auth.getSession().then(({ data }) => {
        if (!mounted) return;
        setSession(data.session ?? null);
        setLoading(false);
      });

      const { data: sub } = supabase.auth.onAuthStateChange((_event, nextSession) => {
        setSession(nextSession);
      });

      unsubscribe = () => sub.subscription.unsubscribe();
    };

    const onReinit = () => setup();
    const onStorage = (e: StorageEvent) => {
      if (e.key === SUPABASE_REMEMBER_ME_KEY) setup();
    };

    setup();
    try {
      window.addEventListener(SUPABASE_REINIT_EVENT, onReinit as any);
      window.addEventListener('storage', onStorage);
    } catch {
      /* ignore */
    }

    return () => {
      mounted = false;
      try {
        window.removeEventListener(SUPABASE_REINIT_EVENT, onReinit as any);
        window.removeEventListener('storage', onStorage);
      } catch {
        /* ignore */
      }
      if (unsubscribe) unsubscribe();
    };
  }, []);

  return {
    loading,
    configured: !!supabase,
    session,
    user: session?.user ?? null,
  };
}
