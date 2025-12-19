import { useState, useEffect, useRef } from 'react';
import { X, ArrowClockwise, Check, XCircle, Clock, List, Timer } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { mockEngine } from '@/lib/mock-engine';
import { playSuccessSound, playErrorSound, resumeAudioContext } from '@/lib/sounds';
import type { Job, JobStatus } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';
import { LogStream } from './LogStream';
import { ConnectionStatus } from './ConnectionStatus';
import { motion, AnimatePresence } from 'framer-motion';

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
  const previousJobStatusesRef = useRef<Map<string, Job['status']>>(new Map());

  useEffect(() => {
    if (isOpen) {
      loadJobs();
      resumeAudioContext();

      const unsubscribe = mockEngine.onJobUpdate(updatedJob => {
        const previousStatus = previousJobStatusesRef.current.get(updatedJob.id);
        
        if (previousStatus && previousStatus !== updatedJob.status) {
          if (updatedJob.status === 'completed' && previousStatus === 'running') {
            playSuccessSound();
          } else if (updatedJob.status === 'failed' && previousStatus === 'running') {
            playErrorSound();
          }
        }
        
        previousJobStatusesRef.current.set(updatedJob.id, updatedJob.status);

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

      const interval = setInterval(() => {
        if (document.visibilityState !== 'visible') return;
        loadJobs();
      }, 30000);

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

  const RunningIndicator = ({ size = 12 }: { size?: number }) => (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <motion.div
        className="absolute rounded-full border border-accent/40"
        style={{ width: size, height: size }}
        animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
        transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute rounded-full border border-transparent border-t-accent"
        style={{ width: size * 0.8, height: size * 0.8 }}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
      />
      <div 
        className="rounded-full bg-accent" 
        style={{ width: size * 0.25, height: size * 0.25 }}
      />
    </div>
  );

  const getStatusIcon = (status: JobStatus) => {
    switch (status) {
      case 'completed':
        return <Check size={14} className="text-success" />;
      case 'failed':
        return <XCircle size={14} className="text-destructive" />;
      case 'running':
        return <RunningIndicator size={14} />;
      case 'cancelled':
        return <X size={14} className="text-muted-foreground" />;
      case 'queued':
        return <Clock size={14} className="text-muted-foreground" />;
    }
  };

  const getStatusBadgeStyle = (status: JobStatus) => {
    switch (status) {
      case 'completed':
        return 'bg-success/10 text-success border-success/20';
      case 'failed':
        return 'bg-destructive/10 text-destructive border-destructive/20';
      case 'running':
        return 'bg-accent/10 text-accent border-accent/20';
      case 'cancelled':
        return 'bg-muted text-muted-foreground border-border';
      case 'queued':
        return 'bg-secondary/50 text-secondary-foreground border-border';
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

  const formatEstimatedTime = (seconds?: number) => {
    if (seconds === undefined || seconds === null) return null;
    if (seconds <= 0) return 'Almost done';
    if (seconds < 60) return `~${seconds}s remaining`;
    if (seconds < 3600) return `~${Math.floor(seconds / 60)}m ${seconds % 60}s remaining`;
    return `~${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m remaining`;
  };

  const filteredJobs = filter === 'all' ? jobs : jobs.filter(j => j.status === filter);

  const runningCount = jobs.filter(j => j.status === 'running').length;
  const queuedCount = jobs.filter(j => j.status === 'queued').length;
  const completedCount = jobs.filter(j => j.status === 'completed').length;
  const failedCount = jobs.filter(j => j.status === 'failed').length;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 w-full max-w-xl border-l border-border bg-background shadow-2xl">
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-between border-b border-border bg-card px-5 py-3">
          <div className="flex items-center gap-2.5">
            <List size={16} className="text-primary" />
            <div>
              <h2 className="text-sm font-medium tracking-tight">Job Monitor</h2>
              <ConnectionStatus isConnected={isConnected} className="mt-0.5" />
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-7 w-7 p-0">
            <X size={14} />
          </Button>
        </div>

        <div className="border-b border-border bg-muted/20 px-5 py-3">
          <div className="grid grid-cols-4 gap-2">
            <div className="flex items-center gap-2 rounded-md border border-accent/20 bg-card px-3 py-2">
              <RunningIndicator size={10} />
              <div>
                <p className="text-base font-medium leading-none">{runningCount}</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Running</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2">
              <Clock size={10} className="text-muted-foreground" />
              <div>
                <p className="text-base font-medium leading-none">{queuedCount}</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Queued</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 rounded-md border border-success/20 bg-card px-3 py-2">
              <Check size={10} className="text-success" />
              <div>
                <p className="text-base font-medium leading-none">{completedCount}</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Done</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 rounded-md border border-destructive/20 bg-card px-3 py-2">
              <XCircle size={10} className="text-destructive" />
              <div>
                <p className="text-base font-medium leading-none">{failedCount}</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">Failed</p>
              </div>
            </div>
          </div>
        </div>

        <Tabs value={filter} onValueChange={(v) => setFilter(v as JobStatus | 'all')} className="flex-1 flex flex-col overflow-hidden">
          <div className="border-b border-border px-5">
            <TabsList className="w-full justify-start h-9 bg-transparent p-0 gap-0">
              <TabsTrigger value="all" className="text-xs px-3 h-9 rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent">All ({jobs.length})</TabsTrigger>
              <TabsTrigger value="running" className="text-xs px-3 h-9 rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent">Running ({runningCount})</TabsTrigger>
              <TabsTrigger value="queued" className="text-xs px-3 h-9 rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent">Queued ({queuedCount})</TabsTrigger>
              <TabsTrigger value="completed" className="text-xs px-3 h-9 rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent">Done ({completedCount})</TabsTrigger>
              <TabsTrigger value="failed" className="text-xs px-3 h-9 rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent">Failed ({failedCount})</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value={filter} className="flex-1 mt-0 overflow-hidden">
            <ScrollArea className="h-full">
              <div className="space-y-2 p-4">
                {filteredJobs.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <List size={32} className="mb-3 text-muted-foreground/50" />
                    <h3 className="text-sm font-medium text-muted-foreground">No jobs</h3>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      {filter === 'all' ? 'No jobs have been started yet' : `No ${filter} jobs`}
                    </p>
                  </div>
                ) : (
                  <AnimatePresence mode="popLayout">
                    {filteredJobs.map(job => {
                      const isExpanded = expandedJobId === job.id;

                      return (
                        <motion.div
                          key={job.id}
                          layout
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, scale: 0.98 }}
                          transition={{ duration: 0.15 }}
                        >
                          <Card className={`overflow-hidden transition-all ${job.status === 'running' ? 'border-accent/30 shadow-sm' : 'border-border'}`}>
                            <div
                              className="cursor-pointer px-3 py-2.5 hover:bg-muted/30 transition-colors"
                              onClick={() => setExpandedJobId(isExpanded ? null : job.id)}
                            >
                              <div className="flex items-start justify-between gap-3">
                                <div className="flex items-start gap-2.5 flex-1 min-w-0">
                                  <div className="mt-0.5 flex-shrink-0">
                                    {getStatusIcon(job.status)}
                                  </div>
                                  
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                      <h3 className="text-xs font-medium truncate">{STAGE_NAMES[job.stageId]}</h3>
                                      <span className="mono text-[10px] text-muted-foreground/60">
                                        #{job.id.split('_').slice(-1)[0]}
                                      </span>
                                    </div>
                                    
                                    <p className="text-[11px] text-muted-foreground truncate mt-0.5">
                                      {job.subjectId}
                                    </p>
                                    
                                    {job.status === 'running' && (
                                      <div className="mt-2 space-y-1.5">
                                        <div className="flex items-center justify-between">
                                          <motion.span 
                                            className="text-[10px] text-muted-foreground"
                                            key={job.currentStep}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                          >
                                            {job.currentStep}
                                          </motion.span>
                                          <span className="text-[10px] font-medium mono text-accent">
                                            {Math.round(job.progress)}%
                                          </span>
                                        </div>
                                        <div className="relative h-1 w-full overflow-hidden rounded-full bg-muted">
                                          <motion.div
                                            className="absolute inset-y-0 left-0 rounded-full bg-accent"
                                            initial={{ width: 0 }}
                                            animate={{ width: `${job.progress}%` }}
                                            transition={{ duration: 0.3, ease: "easeOut" }}
                                          />
                                        </div>
                                        {job.estimatedTimeRemaining !== undefined && (
                                          <div className="flex items-center gap-1 text-[10px] text-accent/80">
                                            <Timer size={10} />
                                            <span className="mono">{formatEstimatedTime(job.estimatedTimeRemaining)}</span>
                                          </div>
                                        )}
                                      </div>
                                    )}
                                    
                                    {job.status === 'failed' && job.error && (
                                      <p className="text-[10px] text-destructive mt-1.5 line-clamp-1">{job.error}</p>
                                    )}
                                    
                                    <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted-foreground/60">
                                      <span>{formatDuration(job.startTime, job.endTime)}</span>
                                      {job.startTime && (
                                        <span>{new Date(job.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                      )}
                                    </div>
                                  </div>
                                </div>
                                
                                <Badge variant="outline" className={`text-[10px] px-1.5 py-0 h-5 font-normal ${getStatusBadgeStyle(job.status)}`}>
                                  {job.status}
                                </Badge>
                              </div>
                            </div>
                            
                            <AnimatePresence>
                              {isExpanded && (
                                <motion.div
                                  initial={{ height: 0, opacity: 0 }}
                                  animate={{ height: 'auto', opacity: 1 }}
                                  exit={{ height: 0, opacity: 0 }}
                                  transition={{ duration: 0.15 }}
                                  className="overflow-hidden"
                                >
                                  <div className="border-t border-border bg-muted/20 px-3 py-3">
                                    <div className="space-y-3">
                                      <div>
                                        <h4 className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Details</h4>
                                        <div className="mono space-y-0.5 text-[10px]">
                                          <div className="flex justify-between">
                                            <span className="text-muted-foreground">Job ID</span>
                                            <span className="text-foreground">{job.id}</span>
                                          </div>
                                          <div className="flex justify-between">
                                            <span className="text-muted-foreground">Stage</span>
                                            <span className="text-foreground">{job.stageId}</span>
                                          </div>
                                          {job.startTime && (
                                            <div className="flex justify-between">
                                              <span className="text-muted-foreground">Started</span>
                                              <span className="text-foreground">{new Date(job.startTime).toLocaleString()}</span>
                                            </div>
                                          )}
                                          {job.endTime && (
                                            <div className="flex justify-between">
                                              <span className="text-muted-foreground">Ended</span>
                                              <span className="text-foreground">{new Date(job.endTime).toLocaleString()}</span>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                      
                                      {job.logPath && (
                                        <div>
                                          <h4 className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1">Log Path</h4>
                                          <p className="mono text-[10px] text-foreground/80 break-all">{job.logPath}</p>
                                        </div>
                                      )}
                                      
                                      {(job.status === 'running' || job.status === 'completed' || job.status === 'failed') && (
                                        <div>
                                          <h4 className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">Live Logs</h4>
                                          <LogStream jobId={job.id} />
                                        </div>
                                      )}
                                      
                                      <div className="flex gap-2 pt-1">
                                        {(job.status === 'running' || job.status === 'queued') && (
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              handleCancel(job.id);
                                            }}
                                            className="h-7 text-[11px] gap-1.5 text-destructive border-destructive/30 hover:bg-destructive/10"
                                          >
                                            <X size={12} />
                                            Cancel
                                          </Button>
                                        )}
                                        
                                        {(job.status === 'failed' || job.status === 'cancelled') && (
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              handleRetry(job.id);
                                            }}
                                            className="h-7 text-[11px] gap-1.5"
                                          >
                                            <ArrowClockwise size={12} />
                                            Retry
                                          </Button>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </Card>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
