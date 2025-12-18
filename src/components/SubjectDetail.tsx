import { useState, useEffect } from 'react';
import { ArrowLeft, Eye, ChartLine, MapTrifold, Table as TableIcon, XCircle } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { mockEngine, engineKind, isBackendEngine } from '@/lib/mock-engine';
import type { Job, StageId, Subject } from '@/types';
import { STAGE_NAMES } from '@/types';
import { VolumeViewer } from './VolumeViewer';
import { CurvesView } from './CurvesView';
import { MapsView } from './MapsView';
import { TablesView } from './TablesView';
import { LogStream } from './LogStream';
import { motion } from 'framer-motion';

function buildEngineSwitchUrl(nextEngine: 'backend' | 'demo'): string {
  const url = new URL(window.location.href);
  url.searchParams.set('engine', nextEngine);
  if (nextEngine === 'backend') {
    const hasBackend = (url.searchParams.get('backend') || '').trim().length > 0;
    if (!hasBackend) {
      const defaultBackend = window.location.protocol === 'https:' ? 'https://127.0.0.1:8787' : 'http://127.0.0.1:8787';
      url.searchParams.set('backend', defaultBackend);
    }
  }
  return url.toString();
}

interface SubjectDetailProps {
  subjectId: string;
  onBack: () => void;
}

export function SubjectDetail({ subjectId, onBack }: SubjectDetailProps) {
  const [subject, setSubject] = useState<Subject | null>(null);

  const [logOpen, setLogOpen] = useState(false);
  const [selectedStage, setSelectedStage] = useState<StageId | null>(null);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [jobLoading, setJobLoading] = useState(false);

  useEffect(() => {
    loadSubject();

    const unsubscribe = mockEngine.onStatusUpdate(update => {
      if (update.subjectId === subjectId) {
        setSubject(prev =>
          prev
            ? { ...prev, stageStatuses: { ...prev.stageStatuses, [update.stageId]: update.status } }
            : null
        );
      }
    });

    return () => {
      unsubscribe();
    };
  }, [subjectId]);

  const loadSubject = async () => {
    const data = await mockEngine.getSubject(subjectId);
    if (data) setSubject(data);
  };

  const openStageLogs = async (stageId: StageId) => {
    setSelectedStage(stageId);
    setSelectedJob(null);
    setLogOpen(true);
    setJobLoading(true);
    try {
      const jobs: Job[] = await mockEngine.getJobs({ subjectId });
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

  if (!subject) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>;
  }

  const completedStages = Object.values(subject.stageStatuses).filter(s => s === 'done').length;
  const totalStages = Object.keys(subject.stageStatuses).length;
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
                <Badge variant={isBackendEngine ? 'default' : 'secondary'} className="text-xs font-normal px-2 py-1">
                  Engine: {engineKind}
                </Badge>

                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => window.location.assign(buildEngineSwitchUrl(isBackendEngine ? 'demo' : 'backend'))}
                  className="h-7 px-2 text-xs"
                >
                  {isBackendEngine ? 'Use demo' : 'Use backend'}
                </Button>
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
                  {Object.entries(subject.stageStatuses).map(([stageId, status]) => (
                    <button
                      key={stageId}
                      type="button"
                      onClick={() => openStageLogs(stageId as StageId)}
                      className="flex items-center justify-between rounded-md border border-border bg-card px-4 py-3 text-left transition-colors hover:bg-muted/30"
                    >
                      <span className="text-sm font-normal">
                        {STAGE_NAMES[stageId as keyof typeof STAGE_NAMES]}
                      </span>
                      <div className="flex items-center gap-2">
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
                        ) : (
                          <div className="flex h-6 w-6 items-center justify-center">
                            <div className="h-1.5 w-1.5 rounded-full bg-border" />
                          </div>
                        )}
                      </div>
                    </button>
                  ))}
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
            <VolumeViewer subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="curves">
            <CurvesView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="maps">
            <MapsView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="tables">
            <TablesView subjectId={subjectId} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
