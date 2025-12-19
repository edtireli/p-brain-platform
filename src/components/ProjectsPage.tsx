import { useState, useEffect, useCallback } from 'react';
import { FolderPlus, Folder, Calendar, HardDrive, UploadSimple, DotsThreeVertical, PencilSimple, Trash } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Switch } from '@/components/ui/switch';
import { mockEngine } from '@/lib/mock-engine';
import type { Project } from '@/types';
import { toast } from 'sonner';
import { useSupabaseAuth } from '@/hooks/use-supabase-auth';

function greetingForLocalTime(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return 'Good morning';
  if (h >= 12 && h < 18) return 'Good afternoon';
  return 'Good evening';
}

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
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const auth = useSupabaseAuth();

  const [draftName, setDraftName] = useState('');
  const [draftStoragePath, setDraftStoragePath] = useState('');
  const [storeWithPatientData, setStoreWithPatientData] = useState(true);
  const [autoFillFromFolder, setAutoFillFromFolder] = useState(true);
  const [isDraggingCreate, setIsDraggingCreate] = useState(false);
  const dropZoneId = 'project-drop-zone';

  const [editProject, setEditProject] = useState<Project | null>(null);
  const [editName, setEditName] = useState('');
  const [editStoragePath, setEditStoragePath] = useState('');

  const subjectIdRe = /^\d{8}x\d+$/i;

  const autoImportSubjects = async (project: Project, storagePath: string) => {
    try {
      const scan = await mockEngine.scanProjectSubjects(project.id);
      let toImport = (scan?.subjects || []).slice();

      if (toImport.length === 0) {
        const base = (storagePath || '')
          .trim()
          .replace(/\/+$/, '')
          .split('/')
          .filter(Boolean)
          .pop();
        if (base && subjectIdRe.test(base)) {
          toImport = [{ name: base, sourcePath: storagePath }];
        }
      }

      if (toImport.length === 0) {
        toast.info('Project created, but no subject folders were found in the storage path.');
        return;
      }

      await mockEngine.importSubjects(project.id, toImport);
      toast.success(`Imported ${toImport.length} subject${toImport.length > 1 ? 's' : ''}`);
    } catch (error) {
      toast.error('Project created, but failed to auto-import subjects');
      console.error(error);
    }
  };

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

    const storagePath = (storeWithPatientData ? defaultStoragePath(name) : draftStoragePath).trim();
    if (!storeWithPatientData && !storagePath) {
      toast.error('Please enter a project storage path');
      return;
    }

    try {
      const project = await mockEngine.createProject({
        name,
        storagePath,
        // When not storing alongside patient data, copy data into the project folder.
        copyDataIntoProject: !storeWithPatientData,
      });
      
      toast.success('Project created successfully');
      setIsCreateDialogOpen(false);
      setDraftName('');
      setDraftStoragePath('');
      setStoreWithPatientData(true);
      setAutoFillFromFolder(true);
      loadProjects();

      if (autoFillFromFolder) {
        await autoImportSubjects(project, storagePath);
      }
      onSelectProject(project.id);
    } catch (error) {
      toast.error('Failed to create project');
      console.error(error);
    }
  };

  const handleUpdateProject = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!editProject) return;

    const name = editName.trim();
    if (!name) {
      toast.error('Please enter a project name');
      return;
    }

    const storagePath = editStoragePath.trim();
    if (!storagePath) {
      toast.error('Please enter a project storage path');
      return;
    }

    try {
      await mockEngine.updateProject(editProject.id, { name, storagePath });
      toast.success('Project updated');
      setIsEditDialogOpen(false);
      setEditProject(null);
      loadProjects();
    } catch (error) {
      toast.error('Failed to update project');
      console.error(error);
    }
  };

  const openCreateDialog = (prefill?: { name?: string; storagePath?: string }) => {
    setDraftName(prefill?.name ?? '');
    setDraftStoragePath(prefill?.storagePath ?? '');
    setStoreWithPatientData(true);
    setAutoFillFromFolder(true);
    setIsCreateDialogOpen(true);
  };

  const openEditDialog = (project: Project) => {
    setEditProject(project);
    setEditName(project.name);
    setEditStoragePath(project.storagePath);
    setIsEditDialogOpen(true);
  };

  const handleDeleteProject = async (project: Project) => {
    const ok = window.confirm(`Delete project "${project.name}"? This removes it from the app, not from disk.`);
    if (!ok) return;
    try {
      await mockEngine.deleteProject(project.id);
      toast.success('Project deleted');
      loadProjects();
    } catch (error) {
      toast.error('Failed to delete project');
      console.error(error);
    }
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
    // Browsers don't provide a trustworthy absolute path for dropped folders.
    // Prefill name only; user must paste the storage path.
    openCreateDialog({ name: rootEntry.name });
    toast.info('Paste the folder path into “Project storage path” to enable backend processing.');
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

  const computedStoragePath = defaultStoragePath(draftName || 'project');

  const displayName = (() => {
    const u: any = auth.user;
    const full = (u?.user_metadata?.full_name || u?.user_metadata?.name || '').toString().trim();
    if (full) {
      const comma = full.indexOf(',');
      const normalized = comma >= 0 ? full.slice(comma + 1).trim() : full;
      return normalized.split(/\s+/)[0] || '';
    }

    const email = (u?.email || '').toString().trim();
    if (!email) return '';
    const localPart = email.split('@')[0] || '';
    return localPart.split(/[._-]+/)[0] || localPart;
  })();

  const greeting = `${greetingForLocalTime()}${displayName ? `, ${displayName}` : ''}`;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-medium tracking-tight text-foreground">
              <span className="italic">p</span>-Brain web
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">Advanced neuroimaging analysis platform</p>
          </div>

          <div />

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

                <div className="space-y-3 rounded-lg border border-border p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <Label>Store project with patient data</Label>
                      <p className="text-xs text-muted-foreground">
                        Keep project state alongside the patient data folder (recommended).
                      </p>
                    </div>
                    <Switch
                      checked={storeWithPatientData}
                      onCheckedChange={(checked) => setStoreWithPatientData(checked)}
                    />
                  </div>

                  {!storeWithPatientData && (
                    <div className="space-y-2">
                      <Label htmlFor="storagePath">Project storage path</Label>
                      <Input
                        id="storagePath"
                        name="storagePath"
                        placeholder="Enter or paste a folder path"
                        value={draftStoragePath}
                        onChange={(e) => setDraftStoragePath(e.target.value)}
                        required
                      />
                    </div>
                  )}

                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <Label>Auto-import patient data from folder</Label>
                      <p className="text-xs text-muted-foreground">Scan the storage path for subject folders after creating the project.</p>
                    </div>
                    <Switch
                      checked={autoFillFromFolder}
                      onCheckedChange={(checked) => setAutoFillFromFolder(checked)}
                    />
                  </div>
                </div>

                <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground space-y-2">
                  <p className="font-medium text-foreground">Data handling</p>
                  <p>The backend processes subjects directly from the storage path and stores logs/indexes in a <span className="mono">.pbrain-web</span> folder under it.</p>
                  <p>{storeWithPatientData ? 'Project state will live beside the patient data (in-place).' : 'Project state will live in the custom path you provide (data copied into project).'}
                  </p>
                  {draftName.trim() ? (
                    <p className="mono text-xs">Default project location: {computedStoragePath}</p>
                  ) : null}
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

        <div className="mb-8 text-center">
          <div className="text-2xl font-medium tracking-tight text-foreground">{greeting}</div>
        </div>

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
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <Card
              id={dropZoneId}
              className={`cursor-pointer border-dashed ${
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
              <CardHeader>
                <CardTitle className="text-base font-medium">
                  <span className="flex items-center gap-2">
                    <UploadSimple size={20} className={isDraggingCreate ? 'text-accent' : 'text-muted-foreground'} />
                    Create from folder
                  </span>
                </CardTitle>
                <CardDescription>
                  Drop a study folder here (or click) to prefill project details.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  variant="secondary"
                  className="gap-2"
                  onClick={(e) => {
                    e.stopPropagation();
                    openCreateDialog();
                  }}
                >
                  <FolderPlus size={18} weight="bold" />
                  Create
                </Button>
              </CardContent>
            </Card>

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
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                          onClick={(e) => e.stopPropagation()}
                          aria-label="Project actions"
                        >
                          <DotsThreeVertical size={18} />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                        <DropdownMenuItem onSelect={() => openEditDialog(project)}>
                          <PencilSimple size={16} />
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem variant="destructive" onSelect={() => handleDeleteProject(project)}>
                          <Trash size={16} />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </CardTitle>
                  <CardDescription className="mono text-xs">
                    {project.storagePath}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Calendar size={16} />
                    <span>Created {new Date(project.createdAt).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <HardDrive size={16} />
                    <span>{project.copyDataIntoProject ? 'Data copied' : 'In-place processing'}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      <Dialog
        open={isEditDialogOpen}
        onOpenChange={(open) => {
          setIsEditDialogOpen(open);
          if (!open) setEditProject(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Project</DialogTitle>
            <DialogDescription>Update the project name and storage path.</DialogDescription>
          </DialogHeader>

          <form onSubmit={handleUpdateProject} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="editName">Project Name</Label>
              <Input
                id="editName"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="editStoragePath">Project storage path</Label>
              <Input
                id="editStoragePath"
                value={editStoragePath}
                onChange={(e) => setEditStoragePath(e.target.value)}
                required
              />
            </div>

            <div className="flex justify-end gap-3">
              <Button type="button" variant="secondary" onClick={() => setIsEditDialogOpen(false)}>
                Cancel
              </Button>
              <Button type="submit">Save</Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
