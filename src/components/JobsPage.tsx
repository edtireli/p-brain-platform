import { useState, useEffect } from 'react';
import { ArrowLeft, FunnelSimple, MagnifyingGlass, Check, X, Clock, Play, ArrowsClockwise } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { engine } from '@/lib/engine';
import type { Job } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'framer-motion';

interface JobsPageProps {
  onBack: () => void;
}

export function JobsPage({ onBack }: JobsPageProps) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [filteredJobs, setFilteredJobs] = useState<Job[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);
  const [jobLogs, setJobLogs] = useState<Record<string, Job['logs']>>({});
  const [runnerStatus, setRunnerStatus] = useState<{ lastSeen?: string; workerId?: string; hostname?: string } | null>(null);

  useEffect(() => {
    let disposed = false;
    let t: number | null = null;

    const tick = async () => {
      if (disposed) return;
      try {
        await loadJobs();
      } catch {
        // ignore
      }
      if (disposed) return;
      t = window.setTimeout(tick, 2000);
    };

    void tick();
    return () => {
      disposed = true;
      if (t != null) window.clearTimeout(t);
    };
  }, []);

  useEffect(() => {
    if (jobs.length === 0) return;
    const hasActive = jobs.some(j => j.status === 'running' || j.status === 'queued');
    const anyRunning = jobs.some(j => j.status === 'running');
    if (!hasActive || anyRunning) {
      setRunnerStatus(null);
      return;
    }
    (async () => {
      try {
        const beats = await (engine as any).getRunnerHeartbeats?.();
        const first = Array.isArray(beats) ? beats[0] : null;
        if (first?.lastSeen) {
          setRunnerStatus({ lastSeen: first.lastSeen, workerId: first.workerId, hostname: first.hostname });
        } else {
          setRunnerStatus({});
        }
      } catch {
        setRunnerStatus({});
      }
    })();
  }, [jobs]);

  useEffect(() => {
    filterJobs();
  }, [jobs, searchQuery, statusFilter]);

  const loadJobs = async () => {
    const allJobs = await engine.getJobs({});
    setJobs(allJobs);
  };

  const filterJobs = () => {
    let filtered = jobs;

    if (statusFilter !== 'all') {
      filtered = filtered.filter(job => job.status === statusFilter);
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        job =>
          job.subjectId.toLowerCase().includes(query) ||
          job.stageId.toLowerCase().includes(query) ||
          job.projectId.toLowerCase().includes(query)
      );
    }

    setFilteredJobs(filtered);
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await engine.cancelJob(jobId);
      toast.success('Job cancelled');
      loadJobs();
    } catch (error) {
      toast.error('Failed to cancel job');
    }
  };

  const handleCancelAllJobs = async () => {
    try {
      const res = await (engine as any).cancelAllJobs?.({});
      const cancelled = typeof res?.cancelled === 'number' ? res.cancelled : undefined;
      toast.success(cancelled != null ? `Cancelled ${cancelled} jobs` : 'Cancelled jobs');
      loadJobs();
    } catch {
      toast.error('Failed to cancel jobs');
    }
  };

  const handleRetryJob = async (jobId: string) => {
    try {
      await engine.retryJob(jobId);
      toast.success('Job restarted');
      loadJobs();
    } catch (error) {
      toast.error('Failed to retry job');
    }
  };

  const RunningIndicator = () => (
    <div className="relative flex h-4 w-4 items-center justify-center">
      <motion.div
        className="absolute h-4 w-4 rounded-full border-2 border-accent/30"
        animate={{ scale: [1, 1.5, 1], opacity: [0.6, 0, 0.6] }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute h-3 w-3 rounded-full border-2 border-transparent border-t-accent"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      />
      <div className="h-1.5 w-1.5 rounded-full bg-accent" />
    </div>
  );

  const loadJobLogs = async (jobId: string) => {
    try {
      const events = await (engine as any).getJobEvents?.(jobId);
      if (Array.isArray(events)) {
        setJobLogs(prev => ({ ...prev, [jobId]: events }));
      }
    } catch {
      /* ignore log fetch errors */
    }
  };

  const getStatusIcon = (status: Job['status']) => {
    switch (status) {
      case 'completed':
        return <Check size={16} weight="bold" className="text-success" />;
      case 'failed':
        return <X size={16} weight="bold" className="text-destructive" />;
      case 'running':
        return <RunningIndicator />;
      case 'queued':
        return <Clock size={16} className="text-muted-foreground" />;
      default:
        return null;
    }
  };

  const getStatusBadgeVariant = (status: Job['status']) => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'failed':
        return 'destructive';
      case 'running':
        return 'secondary';
      case 'queued':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const formatDuration = (start: string, end?: string) => {
    const startTime = new Date(start);
    const endTime = end ? new Date(end) : new Date();
    const durationMs = endTime.getTime() - startTime.getTime();
    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const activeJobs = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
  const completedJobs = jobs.filter(j => j.status === 'completed').length;
  const failedJobs = jobs.filter(j => j.status === 'failed').length;

  const oldestQueuedAgeSec = (() => {
    const queued = jobs.filter(j => j.status === 'queued');
    if (queued.length === 0) return 0;
    const oldest = queued
      .map(j => Date.parse((j as any).created_at ?? j.startTime ?? '') || Date.now())
      .reduce((a, b) => Math.min(a, b), Date.now());
    return Math.max(0, Math.floor((Date.now() - oldest) / 1000));
  })();
  const showRunnerHint = activeJobs > 0 && jobs.every(j => j.status !== 'running') && oldestQueuedAgeSec >= 15;

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b border-border bg-card">
        <div className="mx-auto max-w-7xl px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={onBack} className="gap-2">
                <ArrowLeft size={18} />
              </Button>
              <div>
                <h1 className="text-2xl font-medium tracking-tight">Jobs</h1>
                <p className="text-sm text-muted-foreground mt-0.5">
                  Monitor and manage computational tasks
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-accent"></div>
                  <span className="text-muted-foreground">Active:</span>
                  <span className="font-medium">{activeJobs}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-success"></div>
                  <span className="text-muted-foreground">Completed:</span>
                  <span className="font-medium">{completedJobs}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-destructive"></div>
                  <span className="text-muted-foreground">Failed:</span>
                  <span className="font-medium">{failedJobs}</span>
                </div>
              </div>

              <Button
                variant="destructive"
                size="sm"
                className="gap-2"
                disabled={activeJobs === 0}
                onClick={handleCancelAllJobs}
              >
                <X size={18} />
                Kill all
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-6 py-6">
        {showRunnerHint ? (
          <Card className="mb-6 border border-border bg-card">
            <div className="px-5 py-4 text-sm">
              <div className="font-medium">Jobs are queued but nothing is claiming them.</div>
              <div className="mt-1 text-muted-foreground">
                In Supabase control-plane mode you must run the local worker on the machine that has the data.
                Start it with <span className="mono">backend/scripts/run_supabase_worker.sh</span> (configure <span className="mono">backend/.env</span>).
              </div>
              {runnerStatus ? (
                <div className="mt-2 text-muted-foreground">
                  {runnerStatus.lastSeen
                    ? <>Runner last seen: <span className="mono">{runnerStatus.lastSeen}</span>{runnerStatus.hostname ? <> (<span className="mono">{runnerStatus.hostname}</span>)</> : null}</>
                    : <>Runner status: <span className="mono">unknown</span> (no heartbeat found)</>}
                </div>
              ) : null}
            </div>
          </Card>
        ) : null}

        <div className="mb-6 flex items-center gap-3">
          <div className="relative flex-1">
            <MagnifyingGlass size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search by subject, stage, or project..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-48">
              <FunnelSimple size={18} className="mr-2" />
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="queued">Queued</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {filteredJobs.length === 0 ? (
          <Card className="border-dashed">
            <div className="flex flex-col items-center justify-center py-16">
              <Play size={56} className="mb-4 text-muted-foreground" />
              <h3 className="mb-2 text-base font-medium">No jobs found</h3>
              <p className="text-center text-sm text-muted-foreground">
                {searchQuery || statusFilter !== 'all'
                  ? 'Try adjusting your filters'
                  : 'Jobs will appear here when pipeline stages are executed'}
              </p>
            </div>
          </Card>
        ) : (
          <div className="space-y-2">
            <AnimatePresence mode="popLayout">
              {filteredJobs.map(job => (
                <motion.div
                  key={job.id}
                  layout
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                >
                  <Card className={`overflow-hidden transition-all hover:shadow-sm ${job.status === 'running' ? 'border-accent/40 shadow-accent/5 shadow-lg' : ''}`}>
                    <div
                      className="flex cursor-pointer items-center justify-between p-4"
                      onClick={() => {
                        const next = expandedJobId === job.id ? null : job.id;
                        setExpandedJobId(next);
                        if (next) loadJobLogs(job.id);
                      }}
                    >
                      <div className="flex flex-1 items-center gap-4">
                        <motion.div 
                          className={`flex h-10 w-10 items-center justify-center rounded-lg ${job.status === 'running' ? 'bg-accent/10' : 'bg-muted'}`}
                          animate={job.status === 'running' ? { 
                            boxShadow: ['0 0 0 0 rgba(var(--accent), 0)', '0 0 0 8px rgba(var(--accent), 0.1)', '0 0 0 0 rgba(var(--accent), 0)']
                          } : {}}
                          transition={{ duration: 2, repeat: Infinity }}
                        >
                          {getStatusIcon(job.status)}
                        </motion.div>

                        <div className="flex-1 space-y-1">
                          <div className="flex items-center gap-3">
                            <span className="text-sm font-medium">
                              {STAGE_NAMES[job.stageId as keyof typeof STAGE_NAMES]}
                            </span>
                            <Badge variant={getStatusBadgeVariant(job.status)} className={`text-xs ${job.status === 'running' ? 'animate-pulse' : ''}`}>
                              {job.status}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-3 text-xs text-muted-foreground">
                            <span className="mono">{job.subjectId}</span>
                            <span>•</span>
                            <span>{job.startTime ? new Date(job.startTime).toLocaleString() : 'N/A'}</span>
                            {job.status === 'running' && job.startTime && (() => {
                              const start = job.startTime;
                              return (
                                <>
                                  <span>•</span>
                                  <motion.span
                                    key={formatDuration(start)}
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                  >
                                    {formatDuration(start)}
                                  </motion.span>
                                </>
                              );
                            })()}
                            {job.endTime && job.startTime && (() => {
                              const start = job.startTime;
                              const end = job.endTime;
                              return (
                                <>
                                  <span>•</span>
                                  <span>Duration: {formatDuration(start, end)}</span>
                                </>
                              );
                            })()}
                          </div>
                        </div>

                        {job.status === 'running' && job.progress !== undefined && (
                          <div className="flex items-center gap-3">
                            <div className="relative h-1.5 w-32 overflow-hidden rounded-full bg-muted">
                              <motion.div
                                className="absolute inset-y-0 left-0 bg-accent"
                                initial={{ width: 0 }}
                                animate={{ width: `${job.progress}%` }}
                                transition={{ duration: 0.5, ease: "easeOut" }}
                              />
                              <motion.div
                                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                                animate={{ x: ['-100%', '200%'] }}
                                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                              />
                            </div>
                            <motion.span 
                              className="text-sm font-medium tabular-nums"
                              key={job.progress}
                              initial={{ scale: 1.2, color: 'var(--accent)' }}
                              animate={{ scale: 1, color: 'var(--foreground)' }}
                              transition={{ duration: 0.3 }}
                            >
                              {job.progress}%
                            </motion.span>
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        {job.status === 'running' && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={e => {
                              e.stopPropagation();
                              handleCancelJob(job.id);
                            }}
                          >
                            Cancel
                          </Button>
                        )}
                        {job.status === 'failed' && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={e => {
                              e.stopPropagation();
                              handleRetryJob(job.id);
                            }}
                            className="gap-2"
                          >
                            <ArrowsClockwise size={16} />
                            Retry
                          </Button>
                        )}
                      </div>
                    </div>

                    <AnimatePresence>
                      {expandedJobId === job.id && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="border-t border-border bg-muted/20 p-4">
                            <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                              Logs
                            </div>
                            <ScrollArea className="h-64 min-w-0 rounded-md border border-border bg-card">
                              <pre className="mono min-w-0 whitespace-pre-wrap break-all p-4 text-xs leading-relaxed">
                                {(jobLogs[job.id] ?? []).length === 0 ? (
                                  <div className="text-muted-foreground">Waiting for logs…</div>
                                ) : (
                                  jobLogs[job.id]!.map((log, i) => (
                                    <motion.div
                                      key={i}
                                      className="py-0.5"
                                      initial={{ opacity: 0, x: -10 }}
                                      animate={{ opacity: 1, x: 0 }}
                                      transition={{ delay: i * 0.02 }}
                                    >
                                      <span className="text-muted-foreground">[{log.timestamp.toLocaleTimeString()}]</span>{' '}
                                      <span
                                        className={
                                          log.level === 'error'
                                            ? 'text-destructive'
                                            : log.level === 'warning'
                                            ? 'text-warning'
                                            : ''
                                        }
                                      >
                                        {log.message}
                                      </span>
                                    </motion.div>
                                  ))
                                )}
                              </pre>
                            </ScrollArea>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
