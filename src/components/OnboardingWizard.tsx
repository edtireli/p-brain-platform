import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { engine } from '@/lib/engine';
import { checkLocalBackendHealth, pickFolderWithNativeDialog } from '@/lib/local-backend';
import type { AppSettings, SystemDeps } from '@/types';

type StepId =
  | 'hello'
  | 'name'
  | 'about'
  | 'pbrain'
  | 'fastsurfer'
  | 'freesurfer'
  | 'howto';

const STEPS: StepId[] = ['hello', 'name', 'about', 'pbrain', 'fastsurfer', 'freesurfer', 'howto'];

function basenameOfPath(p: string): string {
  const s = (p || '').replace(/[\\/]+$/g, '');
  const parts = s.split(/[\\/]/g).filter(Boolean);
  return parts[parts.length - 1] || s;
}

function looksLikeAbsolutePath(p: string): boolean {
  const s = (p || '').trim();
  if (!s) return false;
  if (s.startsWith('/')) return true;
  // Windows-ish path handling for completeness
  return /^[a-zA-Z]:\\/.test(s);
}

export function OnboardingWizard({ onDone }: { onDone: () => void }) {
  const [stepIndex, setStepIndex] = useState(0);
  const step = STEPS[Math.min(stepIndex, STEPS.length - 1)];

  const [deps, setDeps] = useState<SystemDeps | null>(null);
  const [busy, setBusy] = useState(false);
  const [scanning, setScanning] = useState(false);

  const [firstName, setFirstName] = useState('');
  const [pbrainMainPy, setPbrainMainPy] = useState('');
  const [fastsurferDir, setFastSurferDir] = useState('');
  const [freesurferHome, setFreeSurferHome] = useState('');

  const installPBrainDependencies = async () => {
    setBusy(true);
    try {
      const picked = await pickFolderWithNativeDialog();
      if (!picked) return;

      const res = await engine.installPBrain(picked);
      toast.success(`p-Brain cloned to ${basenameOfPath(res.pbrainDir)}`);

      const req = await engine.installPBrainRequirements(res.pbrainDir);
      if (req?.outputTail) console.log(req.outputTail);
      toast.success('Dependencies installed');

      const s = await engine.getSettings();
      setPbrainMainPy(s.pbrainMainPy || res.pbrainMainPy || '');
      await refreshDeps();
    } catch (e: any) {
      toast.error(e?.message || 'Install dependencies failed');
      console.error(e);
    } finally {
      setBusy(false);
    }
  };

  const refreshStageRunners = async () => {
    const ok = await checkLocalBackendHealth();
    if (!ok) {
      toast.error('Local backend is not running');
      return;
    }
    setBusy(true);
    try {
      const res = await engine.refreshStageRunners();
      const n = (res?.refreshed || []).length;
      toast.success(n > 0 ? `Updated stage runners (${n})` : 'No stage runners updated');
    } catch (e: any) {
      toast.error(e?.message || 'Failed to refresh stage runners');
      console.error(e);
    } finally {
      setBusy(false);
    }
  };

  const restartBackend = async () => {
    const ok = await checkLocalBackendHealth();
    if (!ok) {
      toast.error('Local backend is not running');
      return;
    }
    setBusy(true);
    try {
      await engine.restartBackend();
      toast.message('Restarting backend…');
      // In dev, this will stop uvicorn and you must restart it manually.
      // In the packaged app, the launcher should bring it back up.
      window.setTimeout(() => {
        try {
          window.location.reload();
        } catch {
          // ignore
        }
      }, 800);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to restart backend');
      console.error(e);
      setBusy(false);
    }
  };

  const refreshDeps = async () => {
    try {
      const d = await engine.getSystemDeps();
      setDeps(d);
    } catch {
      // ignore
    }
  };

  const scanForDeps = async () => {
    setScanning(true);
    try {
      await engine.scanSystemDeps(true);
      const s = await engine.getSettings();
      setFirstName(s.firstName || '');
      setPbrainMainPy(s.pbrainMainPy || '');
      setFastSurferDir(s.fastsurferDir || '');
      setFreeSurferHome(s.freesurferHome || '');
      await refreshDeps();
    } catch (e: any) {
      toast.error(e?.message || 'Scan failed');
      console.error(e);
    } finally {
      setScanning(false);
    }
  };

  const load = async () => {
    const ok = await checkLocalBackendHealth();
    if (!ok) {
      toast.error('Local backend is not running');
      return;
    }

    try {
      const s = await engine.getSettings();
      setFirstName(s.firstName || '');
      setPbrainMainPy(s.pbrainMainPy || '');
      setFastSurferDir(s.fastsurferDir || '');
      setFreeSurferHome(s.freesurferHome || '');
    } catch (e) {
      toast.error('Failed to load app settings');
      console.error(e);
    }

    await refreshDeps();
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    // keep hook order stable; no-op
  }, [step]);

  const savePatch = async (patch: Partial<AppSettings>) => {
    const ok = await checkLocalBackendHealth();
    if (!ok) {
      toast.error('Local backend is not running');
      return;
    }

    try {
      const s = await engine.updateSettings(patch);
      setFirstName(s.firstName || '');
      setPbrainMainPy(s.pbrainMainPy || '');
      setFastSurferDir(s.fastsurferDir || '');
      setFreeSurferHome(s.freesurferHome || '');
      await refreshDeps();
    } catch (e) {
      toast.error('Failed to save settings');
      console.error(e);
    }
  };

  const skip = async () => {
    await savePatch({ onboardingCompleted: true });
    onDone();
  };

  const next = () => {
    setStepIndex((i) => Math.min(i + 1, STEPS.length - 1));
  };

  const finish = async () => {
    await savePatch({ onboardingCompleted: true });
    toast.success('All set');
    onDone();
  };

  const content = (() => {
    if (step === 'hello') {
      return (
        <div className="flex h-full w-full flex-col items-center justify-center gap-10">
          <div className="text-center text-5xl font-medium tracking-tight text-foreground">Welcome</div>
          <div className="text-sm text-muted-foreground">p-Brain platform setup</div>
          <Button size="lg" onClick={next}>
            Continue
          </Button>
        </div>
      );
    }

    if (step === 'name') {
      return (
        <Card className="w-full max-w-xl">
          <CardHeader>
            <CardTitle>Let’s get started</CardTitle>
            <CardDescription>What’s your first name?</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="firstName">First name</Label>
              <Input
                id="firstName"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                placeholder="e.g. Alex"
                autoFocus
              />
              <div className="text-xs text-muted-foreground">This is so that I know what to call you.</div>
            </div>

            <div className="flex items-center justify-end">
              <Button
                onClick={async () => {
                  setBusy(true);
                  try {
                    await savePatch({ firstName: firstName.trim() });
                    next();
                  } finally {
                    setBusy(false);
                  }
                }}
                disabled={!firstName.trim() || busy}
              >
                Continue
              </Button>
            </div>
          </CardContent>
        </Card>
      );
    }

    if (step === 'about') {
      return (
        <Card className="w-full max-w-xl">
          <CardHeader>
            <CardTitle>p-Brain App</CardTitle>
            <CardDescription>Fully offline, local neuroimaging analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-sm text-foreground">
              This app is a fully offline, local, user platform for p-Brain neuroimaging analysis. Your patient data and
              results stay on this machine.
            </div>
            <div className="flex items-center justify-end">
              <Button onClick={next}>Continue</Button>
            </div>
          </CardContent>
        </Card>
      );
    }

    if (step === 'pbrain') {
      const pbrainFound = deps?.pbrainMainPy?.exists ?? false;
      const fastsurferFound = deps?.fastsurfer?.ok ?? false;
      const freesurferFound = deps?.freesurfer?.ok ?? false;

      return (
        <Card className="w-full max-w-xl">
          <CardHeader>
            <CardTitle>p-Brain install location</CardTitle>
            <CardDescription>Select where p-Brain is installed (the folder containing main.py).</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-lg border border-border p-3 text-xs text-muted-foreground">
              Install method: <span className="text-foreground">git clone (raw)</span>
              <div className="mt-1 mono break-all">git clone https://github.com/edtireli/p-brain.git</div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="pbrainMainPy">Path to p-Brain main.py</Label>
              <div className="flex gap-2">
                <Input
                  id="pbrainMainPy"
                  value={pbrainMainPy}
                  onChange={(e) => setPbrainMainPy(e.target.value)}
                  placeholder="/Users/you/p-brain/main.py"
                />
                <Button
                  type="button"
                  variant="secondary"
                  disabled={busy}
                  onClick={async () => {
                    setBusy(true);
                    try {
                      const picked = await pickFolderWithNativeDialog();
                      if (!picked) return;
                      const candidate = `${picked.replace(/[\\/]+$/g, '')}/main.py`;
                      setPbrainMainPy(candidate);
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  Browse
                </Button>
              </div>
              <div className="text-xs text-muted-foreground">
                p-Brain platform uses this to run the pipeline locally.
              </div>
            </div>

            <Separator />

            <div className="space-y-2 rounded-lg border border-border p-3 text-xs text-muted-foreground">
              <div className="flex items-center justify-between gap-3">
                <div>
                  Status:{' '}
                  <span className="text-foreground">
                    {scanning ? 'Finding…' : pbrainFound ? 'Found' : 'Not found'}
                  </span>
                </div>
                {scanning ? (
                  <div className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
                ) : null}
              </div>
              <div>
                FastSurfer:{' '}
                <span className="text-foreground">
                  {scanning ? 'Finding…' : fastsurferFound ? 'Found' : 'Not found'}
                </span>
              </div>
              <div>
                FreeSurfer:{' '}
                <span className="text-foreground">
                  {scanning ? 'Finding…' : freesurferFound ? 'Found' : 'Not found'}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="secondary" disabled={busy || scanning} onClick={scanForDeps}>
                Scan for me
              </Button>

              {!pbrainFound ? (
                <Button type="button" disabled={busy || scanning} onClick={installPBrainDependencies}>
                  Install dependencies
                </Button>
              ) : null}
            </div>

            <div className="flex items-center justify-end">
              <Button
                disabled={busy || scanning || !looksLikeAbsolutePath(pbrainMainPy)}
                onClick={async () => {
                  setBusy(true);
                  try {
                    await savePatch({ pbrainMainPy: pbrainMainPy.trim() });
                    await refreshDeps();
                    if (!(await engine.getSystemDeps()).pbrainMainPy.exists) {
                      toast.error('main.py not found at that path');
                      return;
                    }
                    next();
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                Continue
              </Button>
            </div>
          </CardContent>
        </Card>
      );
    }

    if (step === 'fastsurfer') {
      const ok = deps?.fastsurfer?.ok ?? false;
      return (
        <Card className="w-full max-w-xl">
          <CardHeader>
            <CardTitle>FastSurfer</CardTitle>
            <CardDescription>Segmentation dependency</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3 text-xs text-muted-foreground">
              <div>
                Status:{' '}
                <span className="text-foreground">
                  {scanning ? 'Finding…' : ok ? 'Detected' : 'Not detected'}
                </span>
              </div>
              {!ok ? (
                <div className="flex items-center gap-2">
                  {scanning ? (
                    <div className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
                  ) : null}
                  <Button type="button" variant="secondary" disabled={busy || scanning} onClick={scanForDeps}>
                    Scan
                  </Button>
                </div>
              ) : null}
            </div>

            {deps?.fastsurfer?.runScript ? (
              <div className="rounded-lg border border-border p-3 text-xs text-muted-foreground">
                <div className="mono break-all">{deps.fastsurfer.runScript}</div>
              </div>
            ) : null}

            {!ok ? (
              <>
                <div className="rounded-lg border border-border p-3 text-xs text-muted-foreground">
                  Install method: <span className="text-foreground">git clone (raw)</span>
                  <div className="mt-1">No conda/venv/Docker required for this step.</div>
                  <div className="mt-1 mono break-all">git clone https://github.com/Deep-MI/FastSurfer.git</div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="fastsurferDir">FastSurfer folder</Label>
                  <div className="flex gap-2">
                    <Input
                      id="fastsurferDir"
                      value={fastsurferDir}
                      onChange={(e) => setFastSurferDir(e.target.value)}
                      placeholder="/Users/you/FastSurfer"
                    />
                    <Button
                      type="button"
                      variant="secondary"
                      disabled={busy}
                      onClick={async () => {
                        setBusy(true);
                        try {
                          const picked = await pickFolderWithNativeDialog();
                          if (!picked) return;
                          setFastSurferDir(picked);
                        } finally {
                          setBusy(false);
                        }
                      }}
                    >
                      Browse
                    </Button>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    disabled={busy || scanning || !looksLikeAbsolutePath(fastsurferDir)}
                    onClick={async () => {
                      setBusy(true);
                      try {
                        await savePatch({ fastsurferDir: fastsurferDir.trim() });
                        await refreshDeps();
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    Save
                  </Button>

                  <Button
                    type="button"
                    disabled={busy || scanning}
                    onClick={async () => {
                      setBusy(true);
                      try {
                        const picked = await pickFolderWithNativeDialog();
                        if (!picked) return;
                        const res = await engine.installFastSurfer(picked);
                        toast.success(`FastSurfer installed to ${basenameOfPath(res.fastsurferDir)}`);
                        await refreshDeps();
                        setFastSurferDir(res.fastsurferDir);
                      } catch (e: any) {
                        toast.error(e?.message || 'FastSurfer install failed');
                        console.error(e);
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    Install for me
                  </Button>
                </div>
              </>
            ) : null}

            <div className="flex items-center justify-end">
              <Button onClick={next} disabled={busy}>Continue</Button>
            </div>
          </CardContent>
        </Card>
      );
    }

    if (step === 'freesurfer') {
      const ok = deps?.freesurfer?.ok ?? false;
      return (
        <Card className="w-full max-w-xl">
          <CardHeader>
            <CardTitle>FreeSurfer</CardTitle>
            <CardDescription>Confirm FreeSurfer is available</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3 text-xs text-muted-foreground">
              <div>
                Status:{' '}
                <span className="text-foreground">
                  {scanning ? 'Finding…' : ok ? 'Detected' : 'Not detected'}
                </span>
                {deps?.freesurfer?.reconAll ? (
                  <div className="mt-1 mono break-all">recon-all: {deps.freesurfer.reconAll}</div>
                ) : null}
              </div>
              {!ok ? (
                <div className="flex items-center gap-2">
                  {scanning ? (
                    <div className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
                  ) : null}
                  <Button type="button" variant="secondary" disabled={busy || scanning} onClick={scanForDeps}>
                    Scan
                  </Button>
                </div>
              ) : null}
            </div>

            {!ok ? (
              <>
                <div className="space-y-2">
                  <Label htmlFor="freesurferHome">FREESURFER_HOME (optional)</Label>
                  <div className="flex gap-2">
                    <Input
                      id="freesurferHome"
                      value={freesurferHome}
                      onChange={(e) => setFreeSurferHome(e.target.value)}
                      placeholder="/Applications/freesurfer"
                    />
                    <Button
                      type="button"
                      variant="secondary"
                      disabled={busy}
                      onClick={async () => {
                        setBusy(true);
                        try {
                          const picked = await pickFolderWithNativeDialog();
                          if (!picked) return;
                          setFreeSurferHome(picked);
                        } finally {
                          setBusy(false);
                        }
                      }}
                    >
                      Browse
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    If FreeSurfer is already on your PATH, this can be left blank.
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    disabled={busy || scanning || (!!freesurferHome && !looksLikeAbsolutePath(freesurferHome))}
                    onClick={async () => {
                      setBusy(true);
                      try {
                        await savePatch({ freesurferHome: freesurferHome.trim() });
                        await refreshDeps();
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>

                <div className="text-xs text-muted-foreground">
                  FreeSurfer isn’t detected. Install it and ensure <span className="mono">recon-all</span> is on your PATH,
                  or set <span className="mono">FREESURFER_HOME</span>.
                </div>
              </>
            ) : null}

            <div className="flex items-center justify-end">
              <Button onClick={next} disabled={busy}>Continue</Button>
            </div>
          </CardContent>
        </Card>
      );
    }

    // howto
    return (
      <Card className="w-full max-w-xl">
        <CardHeader>
          <CardTitle>How to use p-Brain</CardTitle>
          <CardDescription>Quick workflow</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2 text-sm text-foreground">
            <div>1) Create a project from your patient data folder.</div>
            <div>2) Import subject folders (e.g. 20230912x1).</div>
            <div>3) Run the full pipeline and monitor progress in Jobs.</div>
            <div>4) Open a subject to view volumes, maps, curves, and the statistics table.</div>
          </div>

          <Separator />

          <div className="space-y-2">
            <div className="text-xs text-muted-foreground">Advanced</div>
            <div className="flex flex-wrap gap-2">
              <Button type="button" variant="secondary" disabled={busy} onClick={refreshStageRunners}>
                Update stage runners
              </Button>
              <Button type="button" variant="secondary" disabled={busy} onClick={restartBackend}>
                Restart backend
              </Button>
            </div>
            <div className="text-xs text-muted-foreground">
              If you’re running the backend manually, restart it in Terminal after clicking restart.
            </div>
          </div>

          <div className="flex items-center justify-end">
            <Button onClick={finish} disabled={busy}>Finish</Button>
          </div>
        </CardContent>
      </Card>
    );
  })();

  return (
    <div className="fixed inset-0 z-50 bg-background">
      <div className="fixed bottom-6 right-6 z-[60]">
        <Button variant="ghost" size="sm" onClick={skip}>
          Skip
        </Button>
      </div>
      <div className="mx-auto flex h-full max-w-6xl flex-col items-center justify-center p-6">
        {step !== 'hello' ? (
          <div className="mb-6 w-full max-w-xl text-xs text-muted-foreground">
            Step {stepIndex} of {STEPS.length - 1}
          </div>
        ) : null}
        {content}
        {step !== 'hello' ? (
          <div className="mt-4 text-xs text-muted-foreground">You can change these later.</div>
        ) : null}
      </div>
    </div>
  );
}
