import { useState, useEffect } from 'react';
import { ArrowLeft, FunnelSimple, MagnifyingGlass, Check, X, Spinner, Clock, Play, ArrowsClockwise } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { mockEngine } from '@/lib/mock-engine';
import type { Job } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';

interface JobsPageProps {
  onBack: () => void;
}

export function JobsPage({ onBack }: JobsPageProps) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [filteredJobs, setFilteredJobs] = useState<Job[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    filterJobs();
  }, [jobs, searchQuery, statusFilter]);

  const loadJobs = async () => {
    const allJobs = await mockEngine.getJobs({});
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
      await mockEngine.cancelJob(jobId);
      toast.success('Job cancelled');
      loadJobs();
    } catch (error) {
      toast.error('Failed to cancel job');
    }
  };

  const handleRetryJob = async (jobId: string) => {
    try {
      await mockEngine.retryJob(jobId);
      toast.success('Job restarted');
      loadJobs();
    } catch (error) {
      toast.error('Failed to retry job');
    }
  };

  const getStatusIcon = (status: Job['status']) => {
    switch (status) {
      case 'completed':
        return <Check size={16} weight="bold" className="text-success" />;
      case 'failed':
        return <X size={16} weight="bold" className="text-destructive" />;
      case 'running':
        return <Spinner size={16} className="animate-spin text-accent" />;
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
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-6 py-6">
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
            {filteredJobs.map(job => (
              <Card key={job.id} className="overflow-hidden transition-all hover:shadow-sm">
                <div
                  className="flex cursor-pointer items-center justify-between p-4"
                  onClick={() => setExpandedJobId(expandedJobId === job.id ? null : job.id)}
                >
                  <div className="flex flex-1 items-center gap-4">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      {getStatusIcon(job.status)}
                    </div>

                    <div className="flex-1 space-y-1">
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-medium">
                          {STAGE_NAMES[job.stageId as keyof typeof STAGE_NAMES]}
                        </span>
                        <Badge variant={getStatusBadgeVariant(job.status)} className="text-xs">
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
                              <span>{formatDuration(start)}</span>
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
                        <div className="h-1.5 w-32 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full bg-accent transition-all duration-300"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium tabular-nums">{job.progress}%</span>
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

                {expandedJobId === job.id && job.logs && (
                  <div className="border-t border-border bg-muted/20 p-4">
                    <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Logs
                    </div>
                    <ScrollArea className="h-64 rounded-md border border-border bg-card">
                      <pre className="mono p-4 text-xs leading-relaxed">
                        {job.logs.map((log, i) => (
                          <div key={i} className="py-0.5">
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
                          </div>
                        ))}
                      </pre>
                    </ScrollArea>
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
