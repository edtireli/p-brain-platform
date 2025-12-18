import { useState, useEffect, useRef } from 'react';
import { Play, UserPlus, ArrowLeft, X, List, CheckSquare, Square, MinusSquare } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { mockEngine } from '@/lib/mock-engine';
import { playSuccessSound, playErrorSound, resumeAudioContext } from '@/lib/sounds';
import type { Project, Subject, StageId, StageStatus, Job } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';
import { JobMonitorPanel } from './JobMonitorPanel';
import { motion, AnimatePresence } from 'framer-motion';

interface ProjectDashboardProps {
  projectId: string;
  onBack: () => void;
  onSelectSubject: (subjectId: string) => void;
}

export function ProjectDashboard({ projectId, onBack, onSelectSubject }: ProjectDashboardProps) {
  const [project, setProject] = useState<Project | null>(null);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isJobMonitorOpen, setIsJobMonitorOpen] = useState(false);
  const [activeJobsCount, setActiveJobsCount] = useState(0);
  const [runningSubjectIds, setRunningSubjectIds] = useState<Set<string>>(new Set());
  const [selectedSubjectIds, setSelectedSubjectIds] = useState<Set<string>>(new Set());
  const previousJobStatusesRef = useRef<Map<string, Job['status']>>(new Map());
  const lastSelectedIndexRef = useRef<number | null>(null);

  useEffect(() => {
    loadProject();
    loadSubjects();

    const unsubscribeStatus = mockEngine.onStatusUpdate(update => {
      setSubjects(prev =>
        prev.map(s =>
          s.id === update.subjectId
            ? { ...s, stageStatuses: { ...s.stageStatuses, [update.stageId]: update.status } }
            : s
        )
      );
    });

    const unsubscribeJob = mockEngine.onJobUpdate((job: Job) => {
      const previousStatus = previousJobStatusesRef.current.get(job.id);
      
      if (previousStatus && previousStatus !== job.status) {
        if (job.status === 'completed' && previousStatus === 'running') {
          playSuccessSound();
        } else if (job.status === 'failed' && previousStatus === 'running') {
          playErrorSound();
        }
      }
      
      previousJobStatusesRef.current.set(job.id, job.status);

      if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
        setRunningSubjectIds(prev => {
          const next = new Set(prev);
          next.delete(job.subjectId);
          return next;
        });
      }
    });

    const interval = setInterval(async () => {
      const jobs = await mockEngine.getJobs({ projectId });
      const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
      setActiveJobsCount(active);
      
      jobs.forEach(job => {
        previousJobStatusesRef.current.set(job.id, job.status);
      });
    }, 2000);

    return () => {
      unsubscribeStatus();
      unsubscribeJob();
      clearInterval(interval);
    };
  }, [projectId]);

  const loadProject = async () => {
    const data = await mockEngine.getProject(projectId);
    if (data) setProject(data);
  };

  const loadSubjects = async () => {
    const data = await mockEngine.getSubjects(projectId);
    setSubjects(data);
  };

  const handleAddSubjects = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const subjectNames = (formData.get('subjectNames') as string)
      .split('\n')
      .filter(name => name.trim());

    if (subjectNames.length === 0) {
      toast.error('Please enter at least one subject name');
      return;
    }

    const subjectsToImport = subjectNames.map(name => ({
      name: name.trim(),
      sourcePath: `/data/subjects/${name.trim()}`,
    }));

    try {
      await mockEngine.importSubjects(projectId, subjectsToImport);
      toast.success(`Added ${subjectsToImport.length} subject(s)`);
      setIsAddDialogOpen(false);
      loadSubjects();
    } catch (error) {
      toast.error('Failed to add subjects');
      console.error(error);
    }
  };

  const handleRunFullPipeline = async () => {
    if (subjects.length === 0) {
      toast.error('No subjects to process');
      return;
    }

    resumeAudioContext();
    setIsRunning(true);
    try {
      const subjectIds = subjects.map(s => s.id);
      setRunningSubjectIds(new Set(subjectIds));
      await mockEngine.runFullPipeline(projectId, subjectIds);
      toast.success('Pipeline started for all subjects');
    } catch (error) {
      toast.error('Failed to start pipeline');
      console.error(error);
    } finally {
      setIsRunning(false);
    }
  };

  const handleRunSelectedPipeline = async () => {
    if (selectedSubjectIds.size === 0) {
      toast.error('No subjects selected');
      return;
    }

    resumeAudioContext();
    setIsRunning(true);
    try {
      const subjectIds = Array.from(selectedSubjectIds);
      setRunningSubjectIds(prev => new Set([...prev, ...subjectIds]));
      await mockEngine.runFullPipeline(projectId, subjectIds);
      toast.success(`Pipeline started for ${subjectIds.length} subject${subjectIds.length > 1 ? 's' : ''}`);
      setSelectedSubjectIds(new Set());
    } catch (error) {
      toast.error('Failed to start pipeline');
      console.error(error);
    } finally {
      setIsRunning(false);
    }
  };

  const handleSelectSubjectToggle = (subjectId: string, index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    
    const isCtrlOrCmd = e.ctrlKey || e.metaKey;
    
    if (e.shiftKey && lastSelectedIndexRef.current !== null) {
      const start = Math.min(lastSelectedIndexRef.current, index);
      const end = Math.max(lastSelectedIndexRef.current, index);
      const rangeIds = subjects.slice(start, end + 1).map(s => s.id);
      
      setSelectedSubjectIds(prev => {
        const next = new Set(prev);
        rangeIds.forEach(id => next.add(id));
        return next;
      });
    } else if (isCtrlOrCmd) {
      setSelectedSubjectIds(prev => {
        const next = new Set(prev);
        if (next.has(subjectId)) {
          next.delete(subjectId);
        } else {
          next.add(subjectId);
        }
        return next;
      });
    } else {
      setSelectedSubjectIds(prev => {
        const next = new Set(prev);
        if (next.has(subjectId)) {
          next.delete(subjectId);
        } else {
          next.add(subjectId);
        }
        return next;
      });
      lastSelectedIndexRef.current = index;
    }
  };

  const handleSelectAll = () => {
    if (selectedSubjectIds.size === subjects.length) {
      setSelectedSubjectIds(new Set());
    } else {
      setSelectedSubjectIds(new Set(subjects.map(s => s.id)));
    }
  };

  const isAllSelected = subjects.length > 0 && selectedSubjectIds.size === subjects.length;
  const isSomeSelected = selectedSubjectIds.size > 0 && selectedSubjectIds.size < subjects.length;

  const handleRunSubjectPipeline = async (e: React.MouseEvent, subjectId: string, subjectName: string) => {
    e.stopPropagation();
    resumeAudioContext();
    
    setRunningSubjectIds(prev => new Set(prev).add(subjectId));
    try {
      await mockEngine.runSubjectPipeline(projectId, subjectId);
      toast.success(`Pipeline started for ${subjectName}`);
    } catch (error) {
      toast.error(`Failed to start pipeline for ${subjectName}`);
      setRunningSubjectIds(prev => {
        const next = new Set(prev);
        next.delete(subjectId);
        return next;
      });
      console.error(error);
    }
  };

  const getStatusIndicator = (status: StageStatus) => {
    switch (status) {
      case 'done':
        return (
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-success/10">
            <div className="h-2 w-2 rounded-full bg-success" />
          </div>
        );
      case 'failed':
        return (
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-destructive/10">
            <X size={12} weight="bold" className="text-destructive" />
          </div>
        );
      case 'running':
        return (
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
        );
      default:
        return (
          <div className="flex h-6 w-6 items-center justify-center">
            <div className="h-1.5 w-1.5 rounded-full bg-border" />
          </div>
        );
    }
  };

  if (!project) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>;
  }

  const stages: StageId[] = [
    'import',
    't1_fit',
    'input_functions',
    'time_shift',
    'segmentation',
    'tissue_ctc',
    'modelling',
    'diffusion',
    'montage_qc',
  ];

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
                <h1 className="text-2xl font-medium tracking-tight">{project.name}</h1>
                <p className="mono text-xs text-muted-foreground mt-0.5">{project.storagePath}</p>
              </div>
            </div>

            <div className="flex gap-3 items-center">
              <AnimatePresence>
                {selectedSubjectIds.size > 0 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.2 }}
                    className="flex items-center gap-3"
                  >
                    <span className="text-sm text-muted-foreground">
                      {selectedSubjectIds.size} selected
                    </span>
                    <Button 
                      onClick={handleRunSelectedPipeline} 
                      disabled={isRunning} 
                      className="gap-2 bg-accent hover:bg-accent/90"
                    >
                      <Play size={18} weight="fill" />
                      Run Selected ({selectedSubjectIds.size})
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedSubjectIds(new Set())}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Clear
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>

              <Button 
                variant="outline" 
                onClick={() => setIsJobMonitorOpen(true)}
                className="gap-2 relative"
              >
                <List size={20} weight="bold" />
                Jobs
                {activeJobsCount > 0 && (
                  <motion.span 
                    className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-xs font-medium text-accent-foreground"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, damping: 25 }}
                  >
                    <motion.span
                      animate={{ scale: [1, 1.1, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      {activeJobsCount}
                    </motion.span>
                  </motion.span>
                )}
              </Button>

              <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="secondary" className="gap-2">
                    <UserPlus size={20} weight="bold" />
                    Add Subjects
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add Subjects</DialogTitle>
                    <DialogDescription>
                      Enter subject names (one per line) to import into the project.
                    </DialogDescription>
                  </DialogHeader>

                  <form onSubmit={handleAddSubjects} className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="subjectNames">Subject Names</Label>
                      <textarea
                        id="subjectNames"
                        name="subjectNames"
                        rows={8}
                        className="mono w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                        placeholder="subject_001&#10;subject_002&#10;subject_003"
                        required
                      />
                      <p className="text-xs text-muted-foreground">
                        Each subject should have NIfTI data in the expected structure
                      </p>
                    </div>

                    <div className="flex justify-end gap-3">
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={() => setIsAddDialogOpen(false)}
                      >
                        Cancel
                      </Button>
                      <Button type="submit">Import Subjects</Button>
                    </div>
                  </form>
                </DialogContent>
              </Dialog>

              <Button onClick={handleRunFullPipeline} disabled={isRunning || subjects.length === 0} className="gap-2">
                <Play size={20} weight="fill" />
                Run All
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-full p-6">
        {subjects.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-16">
              <UserPlus size={64} className="mb-4 text-muted-foreground" />
              <h3 className="mb-2 text-base font-medium">No subjects yet</h3>
              <p className="mb-6 text-center text-sm text-muted-foreground">
                Add subjects to begin neuroimaging analysis
              </p>
              <Button onClick={() => setIsAddDialogOpen(true)} className="gap-2">
                <UserPlus size={20} weight="bold" />
                Add Subjects
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-0 shadow-sm">
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <TooltipProvider>
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border bg-muted/50">
                        <th className="sticky left-0 z-10 bg-muted/50 px-3 py-3 text-center w-10">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                onClick={handleSelectAll}
                                className="flex h-6 w-6 items-center justify-center rounded transition-colors hover:bg-muted"
                              >
                                {isAllSelected ? (
                                  <CheckSquare size={18} weight="fill" className="text-primary" />
                                ) : isSomeSelected ? (
                                  <MinusSquare size={18} weight="fill" className="text-primary" />
                                ) : (
                                  <Square size={18} className="text-muted-foreground" />
                                )}
                              </button>
                            </TooltipTrigger>
                            <TooltipContent>
                              {isAllSelected ? 'Deselect all' : 'Select all'}
                            </TooltipContent>
                          </Tooltip>
                        </th>
                        <th className="sticky left-10 z-10 bg-muted/50 px-2 py-3 text-center text-xs font-medium uppercase tracking-wider w-12">
                          Run
                        </th>
                        <th className="sticky left-[5.5rem] z-10 bg-muted/50 px-4 py-3 text-left text-xs font-medium uppercase tracking-wider">
                          Subject
                        </th>
                        {stages.map(stageId => (
                          <th
                            key={stageId}
                            className="px-3 py-3 text-center text-xs font-medium uppercase tracking-wider"
                          >
                            <div className="whitespace-nowrap">{STAGE_NAMES[stageId]}</div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {subjects.map((subject, index) => {
                        const isSubjectRunning = runningSubjectIds.has(subject.id) || 
                          Object.values(subject.stageStatuses).some(s => s === 'running');
                        const isSelected = selectedSubjectIds.has(subject.id);
                        
                        return (
                          <tr
                            key={subject.id}
                            className={`cursor-pointer border-b border-border transition-colors hover:bg-muted/50 ${isSelected ? 'bg-primary/5' : ''}`}
                            onClick={() => onSelectSubject(subject.id)}
                          >
                            <td className={`sticky left-0 z-10 px-3 py-3 ${isSelected ? 'bg-primary/5' : 'bg-card'} hover:bg-muted/50`}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={(e) => handleSelectSubjectToggle(subject.id, index, e)}
                                    className="flex h-6 w-6 items-center justify-center rounded transition-colors hover:bg-muted"
                                  >
                                    {isSelected ? (
                                      <CheckSquare size={18} weight="fill" className="text-primary" />
                                    ) : (
                                      <Square size={18} className="text-muted-foreground hover:text-foreground" />
                                    )}
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <span>Click to select</span>
                                  <span className="block text-xs text-muted-foreground">Shift+click for range · Ctrl+click to toggle</span>
                                </TooltipContent>
                              </Tooltip>
                            </td>
                            <td className={`sticky left-10 z-10 px-2 py-3 ${isSelected ? 'bg-primary/5' : 'bg-card'} hover:bg-muted/50`}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={(e) => handleRunSubjectPipeline(e, subject.id, subject.name)}
                                    disabled={isSubjectRunning}
                                    className={`flex h-8 w-8 items-center justify-center rounded-md transition-all ${
                                      isSubjectRunning 
                                        ? 'cursor-not-allowed opacity-50' 
                                        : 'hover:bg-primary hover:text-primary-foreground text-muted-foreground hover:scale-110'
                                    }`}
                                  >
                                    {isSubjectRunning ? (
                                      <motion.div
                                        className="h-4 w-4 rounded-full border-2 border-transparent border-t-accent"
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                      />
                                    ) : (
                                      <Play size={16} weight="fill" />
                                    )}
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent side="right">
                                  {isSubjectRunning ? 'Pipeline running...' : 'Run full pipeline for this subject'}
                                </TooltipContent>
                              </Tooltip>
                            </td>
                            <td className={`sticky left-[5.5rem] z-10 px-4 py-3 font-normal ${isSelected ? 'bg-primary/5' : 'bg-card'} hover:bg-muted/50`}>
                              <div className="flex flex-col gap-1">
                                <span className="text-sm">{subject.name}</span>
                                <div className="flex gap-2">
                                  {subject.hasT1 && (
                                    <Badge variant="secondary" className="text-xs font-normal px-1.5 py-0">
                                      T1
                                    </Badge>
                                  )}
                                  {subject.hasDCE && (
                                    <Badge variant="secondary" className="text-xs font-normal px-1.5 py-0">
                                      DCE
                                    </Badge>
                                  )}
                                  {subject.hasDiffusion && (
                                    <Badge variant="secondary" className="text-xs font-normal px-1.5 py-0">
                                      DTI
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </td>
                            {stages.map(stageId => {
                              const status = subject.stageStatuses[stageId];
                              return (
                                <td key={stageId} className="px-3 py-3">
                                  <div className="flex justify-center">
                                    {getStatusIndicator(status)}
                                  </div>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </TooltipProvider>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <JobMonitorPanel
        projectId={projectId}
        isOpen={isJobMonitorOpen}
        onClose={() => setIsJobMonitorOpen(false)}
      />
    </div>
  );
}
