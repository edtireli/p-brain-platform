// Local-only build: Supabase removed.

export type AuthState = {
  loading: boolean;
  configured: boolean;
  session: any | null;
  user: any | null;
};

export function useSupabaseAuth(): AuthState {
  return {
    loading: false,
    configured: false,
    session: null,
    user: null,
  };
}
