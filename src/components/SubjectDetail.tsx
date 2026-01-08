import { useMemo, useState, useEffect } from 'react';
import { ArrowLeft, Eye, ChartLine, MapTrifold, Table as TableIcon, XCircle, Play } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { engine } from '@/lib/engine';
import type { Job, RoiMaskVolume, RoiOverlay, StageId, Subject } from '@/types';
import { STAGE_DEPENDENCIES, STAGE_NAMES } from '@/types';
import { VolumeViewer } from './VolumeViewer';
import { CurvesView } from './CurvesView';
import { MapsView } from './MapsView';
import { TablesView } from './TablesView';
import { TractographyView } from './TractographyView';
import { LogStream } from './LogStream';
import { motion } from 'framer-motion';

interface SubjectDetailProps {
  subjectId: string;
  onBack: () => void;
}

export function SubjectDetail({ subjectId, onBack }: SubjectDetailProps) {
  const [subject, setSubject] = useState<Subject | null>(null);
  const [viewerKind] = useState<'dce' | 't1' | 't2' | 'flair' | 'diffusion'>('dce');

  const [showRoiOverlays, setShowRoiOverlays] = useState(false);
  const [roiOverlays, setRoiOverlays] = useState<RoiOverlay[]>([]);
  const [roiMasks, setRoiMasks] = useState<RoiMaskVolume[]>([]);
  const [roiEnsureAttempted, setRoiEnsureAttempted] = useState(false);
  const [roiLoading, setRoiLoading] = useState(false);

  const [logOpen, setLogOpen] = useState(false);
  const [selectedStage, setSelectedStage] = useState<StageId | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [jobLoading, setJobLoading] = useState(false);
  const [runningStageId, setRunningStageId] = useState<StageId | null>(null);

  const [jobsById, setJobsById] = useState<Record<string, Job>>({});

  const jobs = useMemo(() => Object.values(jobsById), [jobsById]);
  const hasActiveRun = useMemo(
    () => jobs.some(j => j.status === 'queued' || j.status === 'running'),
    [jobs]
  );

  useEffect(() => {
    loadSubject();

    (async () => {
      try {
        const js = await engine.getJobs({ subjectId });
        const next: Record<string, Job> = {};
        for (const j of js) next[j.id] = j;
        setJobsById(next);
      } catch {
        setJobsById({});
      }
    })();

    const unsubscribe = engine.onStatusUpdate(update => {
      if (update.subjectId === subjectId) {
        setSubject(prev =>
          prev
            ? { ...prev, stageStatuses: { ...prev.stageStatuses, [update.stageId]: update.status } }
            : null
        );
      }
    });

    const unsubscribeJobs = engine.onJobUpdate(job => {
      if (job.subjectId !== subjectId) return;
      setJobsById(prev => ({ ...prev, [job.id]: job }));
    });

    return () => {
      unsubscribe();
      unsubscribeJobs();
    };
  }, [subjectId]);

  const loadSubject = async () => {
    const data = await engine.getSubject(subjectId);
    if (data) setSubject(data);
  };

  const ensureRoiOverlaysLoaded = async () => {
    if (roiLoading) return;
    setRoiLoading(true);
    try {
      const overlays = await engine.getSubjectRoiOverlays(subjectId);
      const list = Array.isArray(overlays) ? overlays : [];
      setRoiOverlays(list);

      if (isBackendEngine && list.length === 0 && !roiEnsureAttempted) {
        setRoiEnsureAttempted(true);
        try {
          await engine.ensureSubjectArtifacts(subjectId, 'roi');
        } catch {
          // ignore; viewer can still work without ROI artifacts
        }

        // Poll briefly for ROI overlays to appear.
        let attempts = 0;
        const t = window.setInterval(async () => {
          attempts += 1;
          try {
            const next = await engine.getSubjectRoiOverlays(subjectId);
            if (Array.isArray(next) && next.length > 0) {
              setRoiOverlays(next);
              window.clearInterval(t);
            }
          } catch {
            // ignore
          }
          if (attempts >= 24) {
            window.clearInterval(t);
          }
        }, 2500);
      }
    } catch {
      setRoiOverlays([]);
    } finally {
      setRoiLoading(false);
    }
  };

  const ensureRoiMasksLoaded = async () => {
    try {
      const masks = await engine.getSubjectRoiMasks(subjectId);
      setRoiMasks(Array.isArray(masks) ? masks : []);
    } catch {
      setRoiMasks([]);
    }
  };

  const STAGE_ORDER: StageId[] = [
    'import',
    't1_fit',
    'input_functions',
    'time_shift',
    'segmentation',
    'tissue_ctc',
    'modelling',
    'diffusion',
    'tractography',
  ];

  const normalizeStageStatuses = (stageStatuses: any): Record<StageId, any> => {
    const out: Record<StageId, any> = {} as any;
    for (const s of STAGE_ORDER) out[s] = 'not_run';
    if (stageStatuses && typeof stageStatuses === 'object') {
      for (const [k, v] of Object.entries(stageStatuses)) {
        if (!STAGE_ORDER.includes(k as StageId)) continue;
        if (v === ('pending' as any)) {
          out[k as StageId] = 'not_run';
          continue;
        }
        out[k as StageId] = v as any;
      }
    }
    return out;
  };

  const normalizedStatuses = normalizeStageStatuses(subject?.stageStatuses);

  const openStageLogs = async (stageId: StageId) => {
    setSelectedStage(stageId);
    setSelectedJob(null);
    setLogOpen(true);
    setJobLoading(true);
    try {
      const jobs: Job[] = await engine.getJobs({ subjectId });
      const forStage = jobs
        .filter(j => j.subjectId === subjectId && j.stageId === stageId)
        .sort((a, b) => String(b.startTime || '').localeCompare(String(a.startTime || '')));
      setSelectedJob(forStage[0] ?? null);
    } catch {
      setSelectedJob(null);
    } finally {
      setJobLoading(false);
    }
  };

  const canRunStage = (stageId: StageId): boolean => {
    if (!subject) return false;
    if (runningStageId) return false;
    if (hasActiveRun) return false;
    const statuses = normalizedStatuses;
    if (statuses[stageId] === 'running') return false;

    // If a stage is already completed, allow rerun even if some dependency
    // statuses are missing/unknown (common for older subjects).
    if (statuses[stageId] === 'done') return true;

    const deps = STAGE_DEPENDENCIES[stageId] || [];
    return deps.every(d => statuses[d] === 'done');
  };

  const stageUiStatus = (stageId: StageId): 'not_run' | 'queued' | 'running' | 'done' | 'failed' => {
    if (!subject) return 'not_run';

    const perStage = jobs.filter(j => j.subjectId === subjectId && j.stageId === stageId);
    if (perStage.length > 0) {
      const newest = [...perStage].sort((a, b) => {
        const ak = String(a.startTime || a.endTime || a.id || '');
        const bk = String(b.startTime || b.endTime || b.id || '');
        return bk.localeCompare(ak);
      })[0];

      if (newest.status === 'running') return 'running';
      if (newest.status === 'queued') return 'queued';
      if (newest.status === 'completed') return 'done';
      if (newest.status === 'failed' || newest.status === 'cancelled') return 'failed';
    }

    const persisted = normalizedStatuses?.[stageId] ?? 'not_run';

    // Back-compat: older runs could leak multiple stages as `running`.
    // If no job is actually running for this stage during an active subject run, treat it as queued/not started.
    if (persisted === 'running' && hasActiveRun) return 'not_run';

    return persisted as any;
  };

  const runStage = async (stageId: StageId) => {
    if (!subject) return;
    if (!canRunStage(stageId)) return;
    setRunningStageId(stageId);
    try {
      await engine.runSubjectStage(subject.id, stageId);
    } finally {
      setRunningStageId(null);
    }
  };

  if (!subject) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>;
  }

  const completedStages = STAGE_ORDER.filter(s => normalizedStatuses?.[s] === 'done').length;
  const totalStages = STAGE_ORDER.length;
  const progressPercent = (completedStages / totalStages) * 100;

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b border-border bg-card">
        <div className="mx-auto max-w-full px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={onBack} className="gap-2">
                <ArrowLeft size={18} />
              </Button>
              <div>
                <h1 className="text-2xl font-medium tracking-tight">{subject.name}</h1>
                <p className="mono text-xs text-muted-foreground mt-0.5">{subject.sourcePath}</p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                {subject.hasT1 && (
                  <Badge variant="secondary" className="text-xs font-normal px-2 py-1">
                    T1
                  </Badge>
                )}
                {subject.hasDCE && (
                  <Badge variant="secondary" className="text-xs font-normal px-2 py-1">
                    DCE
                  </Badge>
                )}
                {subject.hasDiffusion && (
                  <Badge variant="secondary" className="text-xs font-normal px-2 py-1">
                    DTI
                  </Badge>
                )}
              </div>
              
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-xs text-muted-foreground">Progress</div>
                  <div className="text-sm font-medium tabular-nums">
                    {completedStages} / {totalStages}
                  </div>
                </div>
                <div className="relative h-12 w-12">
                  <svg viewBox="0 0 100 100" className="rotate-[-90deg]">
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="6"
                      className="text-muted opacity-30"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="6"
                      strokeDasharray={`${progressPercent * 2.638} ${263.8 - progressPercent * 2.638}`}
                      className="text-accent transition-all duration-500"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium tabular-nums">
                    {Math.round(progressPercent)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-full p-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="inline-flex h-9 rounded-lg bg-muted p-1 text-muted-foreground">
            <TabsTrigger value="overview" className="gap-2 text-sm font-normal px-4">
              Overview
            </TabsTrigger>
            <TabsTrigger value="viewer" className="gap-2 text-sm font-normal px-4">
              <Eye size={16} />
              Viewer
            </TabsTrigger>
            <TabsTrigger value="curves" className="gap-2 text-sm font-normal px-4">
              <ChartLine size={16} />
              Curves
            </TabsTrigger>
            <TabsTrigger value="maps" className="gap-2 text-sm font-normal px-4">
              <MapTrifold size={16} />
              Maps
            </TabsTrigger>
            <TabsTrigger value="tractography" className="gap-2 text-sm font-normal px-4">
              Tractography
            </TabsTrigger>
            <TabsTrigger value="tables" className="gap-2 text-sm font-normal px-4">
              <TableIcon size={16} />
              Tables
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-5">
            <Card className="border-0 shadow-sm">
              <div className="p-5">
                <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground mb-4">
                  Pipeline Status
                </h2>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {STAGE_ORDER.map(stageId => {
                    const status = stageUiStatus(stageId);
                    return (
                    <div
                      key={stageId}
                      onClick={() => openStageLogs(stageId)}
                      role="button"
                      tabIndex={0}
                      className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3 text-left transition-colors hover:bg-muted/30"
                    >
                      <span className="text-sm font-normal">
                        {STAGE_NAMES[stageId]}
                      </span>
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="secondary"
                          className="h-8 gap-2"
                          disabled={!canRunStage(stageId)}
                          onClick={(e) => {
                            e.stopPropagation();
                            runStage(stageId);
                          }}
                        >
                          <Play size={14} weight="fill" />
                          {status === 'done' ? 'Rerun' : 'Run'}
                        </Button>
                        {status === 'done' ? (
                          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-success/10">
                            <div className="h-2 w-2 rounded-full bg-success" />
                          </div>
                        ) : status === 'failed' ? (
                          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-destructive/10">
                            <XCircle size={12} weight="bold" className="text-destructive" />
                          </div>
                        ) : status === 'running' ? (
                          <div className="relative flex h-6 w-6 items-center justify-center">
                            <motion.div
                              className="absolute h-6 w-6 rounded-full border-2 border-accent/30"
                              animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0, 0.6] }}
                              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                            />
                            <motion.div
                              className="absolute h-4 w-4 rounded-full border-2 border-transparent border-t-accent"
                              animate={{ rotate: 360 }}
                              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            />
                            <div className="h-2 w-2 rounded-full bg-accent" />
                          </div>
                        ) : status === 'queued' ? (
                          <div className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-accent/30">
                            <div className="h-2 w-2 rounded-full bg-accent/60" />
                          </div>
                        ) : (
                          <div className="flex h-6 w-6 items-center justify-center">
                            <div className="h-1.5 w-1.5 rounded-full bg-border" />
                          </div>
                        )}
                      </div>
                    </div>
                    );
                  })}
                </div>
              </div>
            </Card>

            <Dialog open={logOpen} onOpenChange={setLogOpen}>
              <DialogContent className="max-w-3xl">
                <DialogHeader>
                  <DialogTitle>
                    {selectedStage ? STAGE_NAMES[selectedStage] : 'Stage logs'}
                  </DialogTitle>
                  <DialogDescription>
                    {jobLoading
                      ? 'Loading most recent job…'
                      : selectedJob
                        ? `${selectedJob.status}${selectedJob.startTime ? ` • started ${new Date(selectedJob.startTime).toLocaleString()}` : ''}`
                        : 'No job has run for this stage yet.'}
                  </DialogDescription>
                </DialogHeader>

                {selectedJob?.error ? (
                  <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
                    {selectedJob.error}
                  </div>
                ) : null}

                {selectedJob?.id ? (
                  <LogStream jobId={selectedJob.id} />
                ) : (
                  <div className="text-sm text-muted-foreground">No logs available.</div>
                )}
              </DialogContent>
            </Dialog>

            <Card className="border-0 shadow-sm">
              <div className="p-5">
                <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground mb-4">
                  Data Availability
                </h2>
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3">
                    <div>
                      <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">T1 / IR</div>
                      <div className="text-sm font-normal">
                        {subject.hasT1 ? 'Available' : 'Missing'}
                      </div>
                    </div>
                    {subject.hasT1 ? (
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-success/10">
                        <div className="h-2 w-2 rounded-full bg-success" />
                      </div>
                    ) : (
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-muted">
                        <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground" />
                      </div>
                    )}
                  </div>
                  <div className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3">
                    <div>
                      <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">DCE Series</div>
                      <div className="text-sm font-normal">
                        {subject.hasDCE ? 'Available' : 'Missing'}
                      </div>
                    </div>
                    {subject.hasDCE ? (
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-success/10">
                        <div className="h-2 w-2 rounded-full bg-success" />
                      </div>
                    ) : (
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-muted">
                        <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground" />
                      </div>
                    )}
                  </div>
                  <div className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3">
                    <div>
                      <div className="mb-0.5 text-xs uppercase tracking-wide text-muted-foreground">Diffusion</div>
                      <div className="text-sm font-normal">
                        {subject.hasDiffusion ? 'Available' : 'Missing'}
                      </div>
                    </div>
                    {subject.hasDiffusion ? (
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-success/10">
                        <div className="h-2 w-2 rounded-full bg-success" />
                      </div>
                    ) : (
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-muted">
                        <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground" />
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="viewer">
            <div className="space-y-4">
              {viewerKind === 'dce' ? (
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={showRoiOverlays}
                    onCheckedChange={(v) => {
                      const next = v === true;
                      setShowRoiOverlays(next);
                      if (next) {
                        void ensureRoiOverlaysLoaded();
                        void ensureRoiMasksLoaded();
                      }
                    }}
                  />
                  <Label className="text-sm text-muted-foreground">Show AIF/VIF ROI</Label>
                </div>
              ) : null}

              <VolumeViewer
                subjectId={subjectId}
                kind={viewerKind}
                showRoiOverlays={showRoiOverlays}
                roiOverlays={roiOverlays}
                roiMasks={roiMasks}
              />
            </div>
          </TabsContent>

          <TabsContent value="curves">
            <CurvesView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="maps">
            <MapsView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="tractography">
            <TractographyView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="tables">
            <TablesView subjectId={subjectId} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
