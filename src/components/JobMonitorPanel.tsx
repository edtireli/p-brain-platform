import { useState, useEffect } from 'react';
import { X, Play, ArrowsClockwise, Spinner, Check, XCircle, Clock, List } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { mockEngine } from '@/lib/mock-engine';
import type { Job, JobStatus } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';
import { LogStream } from './LogStream';
import { ConnectionStatus } from './ConnectionStatus';

interface JobMonitorPanelProps {
  projectId: string;
  isOpen: boolean;
  onClose: () => void;
}

export function JobMonitorPanel({ projectId, isOpen, onClose }: JobMonitorPanelProps) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [filter, setFilter] = useState<JobStatus | 'all'>('all');
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(true);

  useEffect(() => {
    if (isOpen) {
      loadJobs();

      const unsubscribe = mockEngine.onJobUpdate(updatedJob => {
        setJobs(prevJobs => {
          const index = prevJobs.findIndex(j => j.id === updatedJob.id);
          if (index >= 0) {
            const newJobs = [...prevJobs];
            newJobs[index] = updatedJob;
            return newJobs;
          }
          return [updatedJob, ...prevJobs];
        });
      });

      const interval = setInterval(loadJobs, 5000);

      return () => {
        unsubscribe();
        clearInterval(interval);
      };
    }
  }, [isOpen, projectId]);

  const loadJobs = async () => {
    const data = await mockEngine.getJobs({ projectId });
    setJobs(data);
  };

  const handleCancel = async (jobId: string) => {
    try {
      await mockEngine.cancelJob(jobId);
      toast.success('Job cancelled');
      loadJobs();
    } catch (error) {
      toast.error('Failed to cancel job');
    }
  };

  const handleRetry = async (jobId: string) => {
    try {
      await mockEngine.retryJob(jobId);
      toast.success('Job restarted');
      loadJobs();
    } catch (error) {
      toast.error('Failed to retry job');
    }
  };

  const getStatusIcon = (status: JobStatus) => {
    switch (status) {
      case 'completed':
        return <Check size={20} weight="bold" className="text-success" />;
      case 'failed':
        return <XCircle size={20} weight="fill" className="text-destructive" />;
      case 'running':
        return <Spinner size={20} className="animate-spin text-accent" />;
      case 'cancelled':
        return <X size={20} weight="bold" className="text-muted-foreground" />;
      case 'queued':
        return <Clock size={20} className="text-muted-foreground" />;
    }
  };

  const getStatusColor = (status: JobStatus) => {
    switch (status) {
      case 'completed':
        return 'bg-success text-success-foreground';
      case 'failed':
        return 'bg-destructive text-destructive-foreground';
      case 'running':
        return 'bg-accent text-accent-foreground';
      case 'cancelled':
        return 'bg-muted text-muted-foreground';
      case 'queued':
        return 'bg-secondary text-secondary-foreground';
    }
  };

  const formatDuration = (startTime?: string, endTime?: string) => {
    if (!startTime) return '-';
    
    const start = new Date(startTime).getTime();
    const end = endTime ? new Date(endTime).getTime() : Date.now();
    const duration = Math.floor((end - start) / 1000);
    
    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const filteredJobs = filter === 'all' ? jobs : jobs.filter(j => j.status === filter);

  const runningCount = jobs.filter(j => j.status === 'running').length;
  const queuedCount = jobs.filter(j => j.status === 'queued').length;
  const completedCount = jobs.filter(j => j.status === 'completed').length;
  const failedCount = jobs.filter(j => j.status === 'failed').length;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full max-w-2xl border-l border-border bg-background shadow-2xl">
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-between border-b border-border bg-card px-6 py-4">
          <div className="flex items-center gap-3">
            <List size={24} weight="bold" className="text-primary" />
            <div>
              <h2 className="text-xl font-semibold">Job Monitor</h2>
              <ConnectionStatus isConnected={isConnected} className="mt-1" />
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X size={20} />
          </Button>
        </div>

        <div className="border-b border-border bg-muted/30 px-6 py-4">
          <div className="grid grid-cols-4 gap-4">
            <Card className="border-accent/20">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Spinner size={20} className="animate-spin text-accent" />
                  <div>
                    <p className="text-2xl font-semibold">{runningCount}</p>
                    <p className="text-xs text-muted-foreground">Running</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Clock size={20} className="text-muted-foreground" />
                  <div>
                    <p className="text-2xl font-semibold">{queuedCount}</p>
                    <p className="text-xs text-muted-foreground">Queued</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-success/20">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Check size={20} weight="bold" className="text-success" />
                  <div>
                    <p className="text-2xl font-semibold">{completedCount}</p>
                    <p className="text-xs text-muted-foreground">Completed</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-destructive/20">
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <XCircle size={20} weight="fill" className="text-destructive" />
                  <div>
                    <p className="text-2xl font-semibold">{failedCount}</p>
                    <p className="text-xs text-muted-foreground">Failed</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        <Tabs value={filter} onValueChange={(v) => setFilter(v as JobStatus | 'all')} className="flex-1 flex flex-col">
          <div className="border-b border-border px-6">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="all">All ({jobs.length})</TabsTrigger>
              <TabsTrigger value="running">Running ({runningCount})</TabsTrigger>
              <TabsTrigger value="queued">Queued ({queuedCount})</TabsTrigger>
              <TabsTrigger value="completed">Completed ({completedCount})</TabsTrigger>
              <TabsTrigger value="failed">Failed ({failedCount})</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value={filter} className="flex-1 mt-0">
            <ScrollArea className="h-full">
              <div className="space-y-3 p-6">
                {filteredJobs.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <List size={64} className="mb-4 text-muted-foreground" />
                    <h3 className="mb-2 text-lg font-medium">No jobs</h3>
                    <p className="text-sm text-muted-foreground">
                      {filter === 'all' ? 'No jobs have been started yet' : `No ${filter} jobs`}
                    </p>
                  </div>
                ) : (
                  filteredJobs.map(job => {
                    const subject = jobs.find(j => j.subjectId === job.subjectId);
                    const isExpanded = expandedJobId === job.id;

                    return (
                      <Card key={job.id} className="overflow-hidden transition-all">
                        <div
                          className="cursor-pointer p-4 hover:bg-muted/50"
                          onClick={() => setExpandedJobId(isExpanded ? null : job.id)}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex items-start gap-3 flex-1 min-w-0">
                              <div className="mt-1">{getStatusIcon(job.status)}</div>
                              
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <h3 className="font-semibold truncate">{STAGE_NAMES[job.stageId]}</h3>
                                  <Badge variant="outline" className="mono text-xs">
                                    {job.id.split('_').slice(-1)[0]}
                                  </Badge>
                                </div>
                                
                                <p className="text-sm text-muted-foreground mb-2 truncate">
                                  Subject: {job.subjectId}
                                </p>
                                
                                {job.status === 'running' && (
                                  <div className="space-y-2">
                                    <div className="flex items-center justify-between text-sm">
                                      <span className="text-muted-foreground">{job.currentStep}</span>
                                      <span className="font-medium mono">{Math.round(job.progress)}%</span>
                                    </div>
                                    <Progress value={job.progress} className="h-2" />
                                  </div>
                                )}
                                
                                {job.status === 'failed' && job.error && (
                                  <p className="text-sm text-destructive mt-2">{job.error}</p>
                                )}
                                
                                <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                  <span>Duration: {formatDuration(job.startTime, job.endTime)}</span>
                                  {job.startTime && (
                                    <span>Started: {new Date(job.startTime).toLocaleTimeString()}</span>
                                  )}
                                </div>
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-2">
                              <Badge className={getStatusColor(job.status)}>
                                {job.status}
                              </Badge>
                            </div>
                          </div>
                        </div>
                        
                        {isExpanded && (
                          <div className="border-t border-border bg-muted/30 p-4">
                            <div className="space-y-4">
                              <div>
                                <h4 className="mb-2 text-sm font-semibold">Job Details</h4>
                                <div className="mono space-y-1 text-xs text-muted-foreground">
                                  <div className="flex justify-between">
                                    <span>Job ID:</span>
                                    <span>{job.id}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Subject ID:</span>
                                    <span>{job.subjectId}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span>Stage:</span>
                                    <span>{job.stageId}</span>
                                  </div>
                                  {job.startTime && (
                                    <div className="flex justify-between">
                                      <span>Started:</span>
                                      <span>{new Date(job.startTime).toLocaleString()}</span>
                                    </div>
                                  )}
                                  {job.endTime && (
                                    <div className="flex justify-between">
                                      <span>Ended:</span>
                                      <span>{new Date(job.endTime).toLocaleString()}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                              
                              {job.logPath && (
                                <div>
                                  <h4 className="mb-2 text-sm font-semibold">Log Path</h4>
                                  <p className="mono text-xs text-muted-foreground">{job.logPath}</p>
                                </div>
                              )}
                              
                              {(job.status === 'running' || job.status === 'completed' || job.status === 'failed') && (
                                <div>
                                  <h4 className="mb-2 text-sm font-semibold">Live Logs</h4>
                                  <LogStream jobId={job.id} />
                                </div>
                              )}
                              
                              <div className="flex gap-2">
                                {(job.status === 'running' || job.status === 'queued') && (
                                  <Button
                                    size="sm"
                                    variant="destructive"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleCancel(job.id);
                                    }}
                                    className="gap-2"
                                  >
                                    <X size={16} weight="bold" />
                                    Cancel
                                  </Button>
                                )}
                                
                                {(job.status === 'failed' || job.status === 'cancelled') && (
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleRetry(job.id);
                                    }}
                                    className="gap-2"
                                  >
                                    <ArrowsClockwise size={16} weight="bold" />
                                    Retry
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        )}
                      </Card>
                    );
                  })
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
