import { useState, useEffect, useCallback } from 'react';
import { FolderPlus, Folder, Calendar, HardDrive, Trash, UploadSimple } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { mockEngine, engineKind, isBackendEngine } from '@/lib/mock-engine';
import type { Project } from '@/types';
import { toast } from 'sonner';

function slugify(input: string): string {
  return (input || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)+/g, '')
    .slice(0, 60) || 'project';
}

function defaultStoragePath(projectName: string): string {
  return `~/pbrain-projects/${slugify(projectName)}`;
}

interface ProjectsPageProps {
  onSelectProject: (projectId: string) => void;
}

export function ProjectsPage({ onSelectProject }: ProjectsPageProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const [draftName, setDraftName] = useState('');
  const [draftStoragePath, setDraftStoragePath] = useState('');
  const [draftCopyData, setDraftCopyData] = useState(true);
  const [isDraggingCreate, setIsDraggingCreate] = useState(false);
  const dropZoneId = 'project-drop-zone';

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    setIsLoading(true);
    try {
      const data = await mockEngine.getProjects();
      setProjects(data);
    } catch (error) {
      toast.error('Failed to load projects');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateProject = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const name = draftName.trim();
    if (!name) {
      toast.error('Please enter a project name');
      return;
    }

    const storagePath = (draftStoragePath || defaultStoragePath(name)).trim();
    if (!storagePath) {
      toast.error('Please enter a project storage path');
      return;
    }

    try {
      const project = await mockEngine.createProject({
        name,
        storagePath,
        copyDataIntoProject: !!draftCopyData,
      });
      
      toast.success('Project created successfully');
      setIsCreateDialogOpen(false);
      setDraftName('');
      setDraftStoragePath('');
      setDraftCopyData(true);
      loadProjects();
      onSelectProject(project.id);
    } catch (error) {
      toast.error('Failed to create project');
      console.error(error);
    }
  };

  const handleDeleteDemoProject = async (projectId: string) => {
    try {
      await mockEngine.deleteProject(projectId);
      toast.success('Demo project deleted');
      loadProjects();
    } catch (error) {
      toast.error('Failed to delete project');
      console.error(error);
    }
  };

  const openCreateDialog = (prefillName?: string) => {
    setDraftName(prefillName ?? '');
    const nextName = prefillName ?? draftName;
    setDraftStoragePath(defaultStoragePath(nextName || 'project'));
    setDraftCopyData(true);
    setIsCreateDialogOpen(true);
  };

  const processDroppedItems = useCallback((items: DataTransferItemList) => {
    const entries: FileSystemDirectoryEntry[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file') {
        const entry = item.webkitGetAsEntry();
        if (entry?.isDirectory) {
          entries.push(entry as FileSystemDirectoryEntry);
        }
      }
    }

    if (entries.length === 0) {
      toast.error('Please drop a folder');
      return;
    }

    const rootEntry = entries[0];
    openCreateDialog(rootEntry.name);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingCreate(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const related = e.relatedTarget as HTMLElement | null;
    if (!related) {
      setIsDraggingCreate(false);
      return;
    }
    if (!related.closest(`#${dropZoneId}`)) {
      setIsDraggingCreate(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingCreate(false);
    if (e.dataTransfer.items) {
      processDroppedItems(e.dataTransfer.items);
    }
  }, [processDroppedItems]);

  const computedStoragePath = defaultStoragePath(draftName || 'my-study');

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-medium tracking-tight text-foreground">
              <span className="italic">p</span>-Brain web
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Advanced neuroimaging analysis platform
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant={isBackendEngine ? 'default' : 'secondary'} className="text-xs font-normal">
              Engine: {engineKind}
            </Badge>
          </div>
          
          <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
            <DialogTrigger asChild>
              <Button size="lg" className="gap-2">
                <FolderPlus size={20} weight="bold" />
                New Project
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Project</DialogTitle>
                <DialogDescription>
                  Create a new neuroimaging analysis project.
                </DialogDescription>
              </DialogHeader>
              
              <form onSubmit={handleCreateProject} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Project Name</Label>
                  <Input
                    id="name"
                    name="name"
                    placeholder="My Study 2024"
                    value={draftName}
                    onChange={(e) => setDraftName(e.target.value)}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="storagePath">Project storage path</Label>
                  <Input
                    id="storagePath"
                    name="storagePath"
                    placeholder={computedStoragePath}
                    value={draftStoragePath}
                    onChange={(e) => setDraftStoragePath(e.target.value)}
                    required
                  />
                  <p className="text-xs text-muted-foreground">
                    For your dataset at <span className="mono">/Volumes/T5_EVO_EDT/data</span>, set this to the parent folder that contains subject folders.
                  </p>
                </div>

                <div className="flex items-start gap-3 rounded-lg border border-border bg-card p-4">
                  <Checkbox
                    id="copyData"
                    checked={draftCopyData}
                    onCheckedChange={(v) => setDraftCopyData(v === true)}
                  />
                  <div className="space-y-1">
                    <Label htmlFor="copyData">Copy data into project</Label>
                    <p className="text-xs text-muted-foreground">
                      When enabled, the backend copies subjects into the project folder before processing.
                    </p>
                  </div>
                </div>

                <div className="rounded-lg border border-border bg-card p-4">
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    This web app writes and manipulates data on your machine. You are responsible for maintaining
                    backups. We are not liable for loss or corruption of data.
                  </p>
                  <div className="mt-3">
                    <p className="text-xs text-muted-foreground">Project storage location</p>
                    <p className="mono text-xs text-foreground mt-1">{draftStoragePath || computedStoragePath}</p>
                  </div>
                </div>

                <div className="flex justify-end gap-3">
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => setIsCreateDialogOpen(false)}
                  >
                    Cancel
                  </Button>
                  <Button type="submit">Create Project</Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>
        </div>

        <Card
          id={dropZoneId}
          className={`mb-8 border-dashed ${
            isDraggingCreate ? 'border-accent bg-accent/5' : 'hover:border-muted-foreground/50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => openCreateDialog()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') openCreateDialog();
          }}
        >
          <CardContent className="flex items-center justify-between gap-4 py-6">
            <div className="flex items-center gap-3">
              <UploadSimple size={24} className={isDraggingCreate ? 'text-accent' : 'text-muted-foreground'} />
              <div>
                <p className="text-sm font-medium text-foreground">Drop a study folder to create a project</p>
                <p className="text-xs text-muted-foreground">
                  Drag and drop a folder here (or click) to prefill project details.
                </p>
              </div>
            </div>
            <Button variant="secondary" className="gap-2" onClick={(e) => { e.stopPropagation(); openCreateDialog(); }}>
              <FolderPlus size={18} weight="bold" />
              Create
            </Button>
          </CardContent>
        </Card>

        {isLoading ? (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map(i => (
              <Card key={i} className="animate-pulse">
                <CardHeader>
                  <div className="h-6 w-3/4 rounded bg-muted" />
                  <div className="h-4 w-1/2 rounded bg-muted" />
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="h-4 w-full rounded bg-muted" />
                    <div className="h-4 w-2/3 rounded bg-muted" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : projects.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-16">
              <Folder size={64} className="mb-4 text-muted-foreground" />
              <h3 className="mb-2 text-base font-medium">No projects yet</h3>
              <p className="mb-6 text-center text-sm text-muted-foreground">
                Create your first neuroimaging analysis project to get started
              </p>
              <Button onClick={() => setIsCreateDialogOpen(true)} className="gap-2">
                <FolderPlus size={20} weight="bold" />
                Create Project
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map(project => (
              <Card
                key={project.id}
                className="cursor-pointer border-0 shadow-sm transition-all hover:shadow-md"
                onClick={() => onSelectProject(project.id)}
              >
                <CardHeader>
                  <CardTitle className="flex items-start justify-between gap-3 text-base font-medium">
                    <span className="flex items-center gap-2">
                      <Folder size={20} weight="fill" className="text-primary" />
                      {project.name}
                    </span>

                    {project.id === 'demo_proj_001' && (
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        aria-label="Delete demo project"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteDemoProject(project.id);
                        }}
                      >
                        <Trash size={18} />
                      </button>
                    )}
                  </CardTitle>
                  <CardDescription className="mono text-xs">
                    {project.storagePath}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Calendar size={16} />
                    <span>
                      Created {new Date(project.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <HardDrive size={16} />
                    <span>
                      {project.copyDataIntoProject ? 'Data copied' : 'References source'}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
