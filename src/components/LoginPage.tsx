import { useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { toast } from 'sonner';
import { getRememberMe, setRememberMe, supabase, supabaseConfigured } from '@/lib/supabase';

export function LoginPage() {
  const configured = useMemo(() => supabaseConfigured(), []);
  const [mode, setMode] = useState<'signin' | 'register' | 'reset' | 'updatePassword'>('signin');
  const [error, setError] = useState('');

  const [rememberMe, setRememberMeState] = useState(() => {
    try {
      return getRememberMe();
    } catch {
      return true;
    }
  });

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [institution, setInstitution] = useState('');

  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [isBusy, setIsBusy] = useState(false);

  useEffect(() => {
    // If the user opened a Supabase recovery link, show a "set new password" form.
    try {
      const hash = window.location.hash || '';
      const params = new URLSearchParams(hash.startsWith('#') ? hash.slice(1) : hash);
      if (params.get('type') === 'recovery') {
        setMode('updatePassword');
      }
    } catch {
      /* ignore */
    }
  }, []);

  const signInPassword = async () => {
    if (!supabase) return;
    setIsBusy(true);
    setError('');
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });
      if (error) throw error;
    } catch (e: any) {
      setError(e?.message || 'Sign in failed');
    } finally {
      setIsBusy(false);
    }
  };

  const signUpPassword = async () => {
    if (!supabase) return;
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    setIsBusy(true);
    setError('');
    try {
      const { data, error } = await supabase.auth.signUp({
        email: email.trim(),
        password,
        options: {
          emailRedirectTo: window.location.origin,
          data: {
            full_name: fullName.trim() || undefined,
            institution: institution.trim() || undefined,
          },
        },
      });
      if (error) throw error;
      if (data.session) {
        toast.success('Account created');
      } else {
        toast.success('Check your email to confirm your account');
      }
      setMode('signin');
    } catch (e: any) {
      setError(e?.message || 'Sign up failed');
    } finally {
      setIsBusy(false);
    }
  };

  const sendPasswordReset = async () => {
    if (!supabase) return;
    setIsBusy(true);
    setError('');
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
        redirectTo: window.location.origin,
      });
      if (error) throw error;
      toast.success('Check your email for the reset link');
      setMode('signin');
    } catch (e: any) {
      setError(e?.message || 'Could not send reset email');
    } finally {
      setIsBusy(false);
    }
  };

  const updatePassword = async () => {
    if (!supabase) return;
    if (newPassword !== confirmNewPassword) {
      setError('Passwords do not match');
      return;
    }
    setIsBusy(true);
    setError('');
    try {
      const { error } = await supabase.auth.updateUser({ password: newPassword });
      if (error) throw error;
      toast.success('Password updated');
      setMode('signin');
      setNewPassword('');
      setConfirmNewPassword('');
    } catch (e: any) {
      setError(e?.message || 'Could not update password');
    } finally {
      setIsBusy(false);
    }
  };

  const sendMagicLink = async () => {
    if (!supabase) return;
    setIsBusy(true);
    setError('');
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
      setError(e?.message || 'Could not send sign-in link');
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
            <CardDescription>
              {mode === 'signin' && 'Sign in to continue.'}
              {mode === 'register' && 'Create an account to continue.'}
              {mode === 'reset' && 'Request a password reset link.'}
              {mode === 'updatePassword' && 'Set a new password.'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {!configured ? (
              <div className="text-sm text-muted-foreground">
                Supabase is not configured. Set <span className="mono">VITE_SUPABASE_URL</span> and{' '}
                <span className="mono">VITE_SUPABASE_ANON_KEY</span>.
              </div>
            ) : null}

            {error ? (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
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

            {mode === 'updatePassword' ? (
              <>
                <div className="space-y-2">
                  <Label htmlFor="newPassword">New password</Label>
                  <Input
                    id="newPassword"
                    type="password"
                    autoComplete="new-password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    disabled={!configured || isBusy}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmNewPassword">Confirm new password</Label>
                  <Input
                    id="confirmNewPassword"
                    type="password"
                    autoComplete="new-password"
                    value={confirmNewPassword}
                    onChange={(e) => setConfirmNewPassword(e.target.value)}
                    disabled={!configured || isBusy}
                  />
                </div>

                <div className="flex gap-3">
                  <Button
                    className="flex-1"
                    type="button"
                    onClick={updatePassword}
                    disabled={!configured || isBusy || !newPassword || !confirmNewPassword}
                  >
                    Update password
                  </Button>
                  <Button
                    className="flex-1"
                    type="button"
                    variant="secondary"
                    onClick={() => {
                      setError('');
                      setMode('signin');
                    }}
                    disabled={isBusy}
                  >
                    Back
                  </Button>
                </div>
              </>
            ) : (
              <>
                {(mode === 'signin' || mode === 'register') ? (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="password">Password</Label>
                      <Input
                        id="password"
                        type="password"
                        autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={!configured || isBusy}
                      />
                    </div>

                    {mode === 'register' ? (
                      <>
                        <div className="space-y-2">
                          <Label htmlFor="confirmPassword">Confirm password</Label>
                          <Input
                            id="confirmPassword"
                            type="password"
                            autoComplete="new-password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            disabled={!configured || isBusy}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="fullName">Full name</Label>
                          <Input
                            id="fullName"
                            type="text"
                            autoComplete="name"
                            value={fullName}
                            onChange={(e) => setFullName(e.target.value)}
                            placeholder="Dr. Jane Smith"
                            disabled={!configured || isBusy}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="institution">Institution (optional)</Label>
                          <Input
                            id="institution"
                            type="text"
                            autoComplete="organization"
                            value={institution}
                            onChange={(e) => setInstitution(e.target.value)}
                            placeholder="University / Hospital"
                            disabled={!configured || isBusy}
                          />
                        </div>
                      </>
                    ) : null}

                    {mode === 'signin' ? (
                      <div className="flex items-center justify-between gap-3">
                        <label className="flex items-center gap-2 text-sm text-muted-foreground" htmlFor="rememberMe">
                          <Checkbox
                            id="rememberMe"
                            checked={rememberMe}
                            onCheckedChange={(v) => {
                              const next = v === true;
                              setRememberMeState(next);
                              setRememberMe(next);
                            }}
                            disabled={!configured || isBusy}
                          />
                          Remember me
                        </label>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setError('');
                            setMode('reset');
                          }}
                          disabled={isBusy}
                        >
                          Forgot password?
                        </Button>
                      </div>
                    ) : null}

                    <div className="flex gap-3">
                      {mode === 'signin' ? (
                        <Button
                          className="flex-1"
                          type="button"
                          onClick={signInPassword}
                          disabled={!configured || isBusy || !email.trim() || !password}
                        >
                          Sign in
                        </Button>
                      ) : (
                        <Button
                          className="flex-1"
                          type="button"
                          onClick={signUpPassword}
                          disabled={!configured || isBusy || !email.trim() || !password || !confirmPassword}
                        >
                          Create account
                        </Button>
                      )}

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

                    <div className="text-center text-sm">
                      {mode === 'signin' ? (
                        <>
                          <span className="text-muted-foreground">Don't have an account? </span>
                          <Button
                            type="button"
                            variant="link"
                            className="px-0"
                            onClick={() => {
                              setError('');
                              setMode('register');
                            }}
                            disabled={isBusy}
                          >
                            Create one
                          </Button>
                        </>
                      ) : (
                        <>
                          <span className="text-muted-foreground">Already have an account? </span>
                          <Button
                            type="button"
                            variant="link"
                            className="px-0"
                            onClick={() => {
                              setError('');
                              setMode('signin');
                            }}
                            disabled={isBusy}
                          >
                            Sign in
                          </Button>
                        </>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-sm text-muted-foreground">
                      Enter your email and we’ll send a password reset link.
                    </div>

                    <div className="flex gap-3">
                      <Button
                        className="flex-1"
                        type="button"
                        onClick={sendPasswordReset}
                        disabled={!configured || isBusy || !email.trim()}
                      >
                        Send reset email
                      </Button>
                      <Button
                        className="flex-1"
                        type="button"
                        variant="secondary"
                        onClick={() => {
                          setError('');
                          setMode('signin');
                        }}
                        disabled={isBusy}
                      >
                        Back
                      </Button>
                    </div>
                  </>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
