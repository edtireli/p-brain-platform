import { useState, useEffect } from 'react';
import { Play, UserPlus, ArrowLeft, X, List } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { mockEngine } from '@/lib/mock-engine';
import type { Project, Subject, StageId, StageStatus } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';
import { JobMonitorPanel } from './JobMonitorPanel';
import { motion } from 'framer-motion';

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

  useEffect(() => {
    loadProject();
    loadSubjects();

    const unsubscribe = mockEngine.onStatusUpdate(update => {
      setSubjects(prev =>
        prev.map(s =>
          s.id === update.subjectId
            ? { ...s, stageStatuses: { ...s.stageStatuses, [update.stageId]: update.status } }
            : s
        )
      );
    });

    const interval = setInterval(async () => {
      const jobs = await mockEngine.getJobs({ projectId });
      const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
      setActiveJobsCount(active);
    }, 2000);

    return () => {
      unsubscribe();
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

    setIsRunning(true);
    try {
      const subjectIds = subjects.map(s => s.id);
      await mockEngine.runFullPipeline(projectId, subjectIds);
      toast.success('Pipeline started for all subjects');
    } catch (error) {
      toast.error('Failed to start pipeline');
      console.error(error);
    } finally {
      setIsRunning(false);
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

            <div className="flex gap-3">
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
                Run Full Auto Pipeline
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
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border bg-muted/50">
                      <th className="sticky left-0 z-10 bg-muted/50 px-4 py-3 text-left text-xs font-medium uppercase tracking-wider">
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
                    {subjects.map(subject => (
                      <tr
                        key={subject.id}
                        className="cursor-pointer border-b border-border transition-colors hover:bg-muted/50"
                        onClick={() => onSelectSubject(subject.id)}
                      >
                        <td className="sticky left-0 z-10 bg-card px-4 py-3 font-normal hover:bg-muted/50">
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
                    ))}
                  </tbody>
                </table>
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
