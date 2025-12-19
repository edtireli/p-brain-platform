import { useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';
import { supabase, supabaseConfigured } from '@/lib/supabase';

export function LoginPage() {
  const configured = useMemo(() => supabaseConfigured(), []);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isBusy, setIsBusy] = useState(false);

  const signInPassword = async () => {
    if (!supabase) return;
    setIsBusy(true);
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });
      if (error) throw error;
    } catch (e: any) {
      toast.error(e?.message || 'Sign in failed');
    } finally {
      setIsBusy(false);
    }
  };

  const sendMagicLink = async () => {
    if (!supabase) return;
    setIsBusy(true);
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: {
          emailRedirectTo: window.location.origin,
        },
      });
      if (error) throw error;
      toast.success('Check your email for the sign-in link');
    } catch (e: any) {
      toast.error(e?.message || 'Could not send sign-in link');
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-md pt-12">
        <Card>
          <CardHeader>
            <CardTitle>
              <span className="italic">p</span>-Brain web
            </CardTitle>
            <CardDescription>Sign in to continue.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {!configured ? (
              <div className="text-sm text-muted-foreground">
                Supabase is not configured. Set <span className="mono">VITE_SUPABASE_URL</span> and{' '}
                <span className="mono">VITE_SUPABASE_ANON_KEY</span>.
              </div>
            ) : null}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@lab.org"
                disabled={!configured || isBusy}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password (optional)</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={!configured || isBusy}
              />
            </div>

            <div className="flex gap-3">
              <Button
                className="flex-1"
                type="button"
                onClick={signInPassword}
                disabled={!configured || isBusy || !email.trim() || !password}
              >
                Sign in
              </Button>
              <Button
                className="flex-1"
                type="button"
                variant="secondary"
                onClick={sendMagicLink}
                disabled={!configured || isBusy || !email.trim()}
              >
                Email link
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
