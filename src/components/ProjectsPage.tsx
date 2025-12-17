import { useState, useEffect } from 'react';
import { FolderPlus, Folder, Calendar, HardDrive } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { mockEngine } from '@/lib/mock-engine';
import type { Project } from '@/types';
import { toast } from 'sonner';

interface ProjectsPageProps {
  onSelectProject: (projectId: string) => void;
}

export function ProjectsPage({ onSelectProject }: ProjectsPageProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

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
    const formData = new FormData(e.currentTarget);
    
    const name = formData.get('name') as string;
    const storagePath = formData.get('storagePath') as string;
    const copyDataIntoProject = formData.get('copyData') === 'on';

    if (!name || !storagePath) {
      toast.error('Please fill in all required fields');
      return;
    }

    try {
      const project = await mockEngine.createProject({
        name,
        storagePath,
        copyDataIntoProject,
      });
      
      toast.success('Project created successfully');
      setIsCreateDialogOpen(false);
      loadProjects();
      onSelectProject(project.id);
    } catch (error) {
      toast.error('Failed to create project');
      console.error(error);
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground">
              p-brain Local Studio
            </h1>
            <p className="mt-2 text-muted-foreground">
              Local-first neuroimaging analysis platform
            </p>
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
                  Set up a new neuroimaging analysis project with local storage.
                </DialogDescription>
              </DialogHeader>
              
              <form onSubmit={handleCreateProject} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Project Name</Label>
                  <Input
                    id="name"
                    name="name"
                    placeholder="My Study 2024"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="storagePath">Storage Path</Label>
                  <Input
                    id="storagePath"
                    name="storagePath"
                    placeholder="/Users/researcher/pbrain-projects/my-study"
                    required
                    className="mono text-sm"
                  />
                  <p className="text-xs text-muted-foreground">
                    Local directory where project data and artifacts will be stored
                  </p>
                </div>

                <div className="flex items-center justify-between rounded-lg border border-border bg-card p-4">
                  <div className="space-y-0.5">
                    <Label htmlFor="copyData" className="text-base">
                      Copy data into project
                    </Label>
                    <p className="text-sm text-muted-foreground">
                      Import files to project storage instead of referencing source paths
                    </p>
                  </div>
                  <Switch id="copyData" name="copyData" />
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
              <h3 className="mb-2 text-lg font-medium">No projects yet</h3>
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
                className="cursor-pointer transition-all hover:shadow-lg hover:border-accent"
                onClick={() => onSelectProject(project.id)}
              >
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Folder size={24} weight="fill" className="text-primary" />
                    {project.name}
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
