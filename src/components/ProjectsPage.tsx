import { useState, useEffect, useCallback } from 'react';
import { FolderPlus, Folder, Calendar, HardDrive, UploadSimple, DotsThreeVertical, PencilSimple, Trash } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Switch } from '@/components/ui/switch';
import { engine } from '@/lib/engine';
import { checkLocalBackendHealth, pickFolderWithNativeDialog, resolveStoragePathFromDrop } from '@/lib/local-backend';
import type { FolderStructureConfig, Project } from '@/types';
import { DEFAULT_FOLDER_STRUCTURE } from '@/types';
import { toast } from 'sonner';

async function probeHealth(base: string, timeoutMs: number = 350): Promise<boolean> {
  const controller = new AbortController();
  const t = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base.replace(/\/+$/, '')}/health`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

async function discoverLocalBackendBaseUrl(): Promise<string | null> {
  // Only attempt discovery for non-Tauri local dev flows.
  // In the packaged Tauri app, the launcher must provide the backend URL.
  try {
    const hasTauri = typeof (window as any)?.__TAURI__ !== 'undefined';
    if (hasTauri) return null;

    const proto = window.location.protocol;
    const isLocal = proto === 'file:';
    if (!isLocal) return null;
  } catch {
    return null;
  }

  const ports: number[] = [];
  for (let p = 8787; p <= 8887; p++) ports.push(p);

  // Probe in small batches to keep startup snappy.
  const batchSize = 16;
  for (let i = 0; i < ports.length; i += batchSize) {
    const batch = ports.slice(i, i + batchSize);
    const bases = batch.map(p => `http://127.0.0.1:${p}`);

    const results = await Promise.all(bases.map(async b => ((await probeHealth(b)) ? b : null)));
    const found = results.find(Boolean) as string | null;
    if (found) return found;
  }

  return null;
}

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

function looksLikeAbsolutePath(p: string): boolean {
  const s = (p || '').trim();
  return s.startsWith('/') || s.startsWith('~/') || s.startsWith('file://');
}

interface ProjectsPageProps {
  onSelectProject: (projectId: string) => void;
}

export function ProjectsPage({ onSelectProject }: ProjectsPageProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [backendStatus, setBackendStatus] = useState<'waiting' | 'ready' | 'error'>('waiting');
  const [warmStatus, setWarmStatus] = useState<'idle' | 'running' | 'done' | 'failed'>('idle');
  const [startupNote, setStartupNote] = useState<string>('Starting backend…');

  const [draftName, setDraftName] = useState('');
  const [draftStoragePath, setDraftStoragePath] = useState('');
  const [storeWithPatientData, setStoreWithPatientData] = useState(true);
	const [pendingDroppedSubjects, setPendingDroppedSubjects] = useState<Array<{ name: string; sourcePath: string }>>([]);
  const [draftFolderStructure, setDraftFolderStructure] = useState<FolderStructureConfig>(DEFAULT_FOLDER_STRUCTURE);
  const [isDraggingCreate, setIsDraggingCreate] = useState(false);
  const dropZoneId = 'project-drop-zone';

  const [editProject, setEditProject] = useState<Project | null>(null);
  const [editName, setEditName] = useState('');
  const [editStoragePath, setEditStoragePath] = useState('');
  const [isPickingFolder, setIsPickingFolder] = useState(false);
  const [firstName, setFirstName] = useState<string>('');

  const subjectIdRe = /^\d{8}x\d+$/i;

  const basenameOfPath = (p: string): string => {
    const s = (p || '').replace(/[\\/]+$/g, '');
    const parts = s.split(/[\\/]/g).filter(Boolean);
    return parts[parts.length - 1] || s;
  };

  const applyDroppedStorageFolder = useCallback(
    (absPath: string) => {
      const p = absPath.trim();
      if (!p) return;
      setStoreWithPatientData(true);
      setDraftStoragePath(p);
      setDraftName((prev) => (prev.trim() ? prev : basenameOfPath(p)));
    },
    [setDraftName, setDraftStoragePath, setStoreWithPatientData]
  );

  const handleDroppedCreateFolder = useCallback(
    (absPath: string) => {
      const p = (absPath || '').trim();
      if (!p) return;
      setDraftName(basenameOfPath(p));
      setDraftStoragePath(p);
      setPendingDroppedSubjects([]);
      setStoreWithPatientData(true);
      setDraftFolderStructure(DEFAULT_FOLDER_STRUCTURE);
      setIsCreateDialogOpen(true);
      toast.info('Folder dropped. Create a project from it.');
    },
    [setDraftName, setDraftStoragePath, setPendingDroppedSubjects, setStoreWithPatientData, setDraftFolderStructure, setIsCreateDialogOpen]
  );

  useEffect(() => {
    const tauri = (window as any)?.__TAURI__;
    const listen = tauri?.event?.listen;
    if (typeof listen !== 'function') return;

    let unlisten: undefined | (() => void);
    void (async () => {
      try {
        unlisten = await listen('tauri://file-drop', (e: any) => {
          const payload = e?.payload;
          const paths: unknown = Array.isArray(payload) ? payload : payload?.paths;
          const first = Array.isArray(paths) ? paths[0] : undefined;
          if (typeof first !== 'string') return;

          // If the create dialog is open, treat it as a storage path drop.
          if (isCreateDialogOpen) {
            applyDroppedStorageFolder(first);
            return;
          }

          // Otherwise, treat it as a quick-create drop.
          handleDroppedCreateFolder(first);
        });
      } catch {
        // ignore
      }
    })();

    return () => {
      try {
        unlisten?.();
      } catch {
        // ignore
      }
    };
  }, [isCreateDialogOpen, applyDroppedStorageFolder, handleDroppedCreateFolder]);

  const importDroppedSubjectsOnly = async (project: Project) => {
    try {
      if (pendingDroppedSubjects.length === 0) return;
      await engine.importSubjects(project.id, pendingDroppedSubjects);
      toast.success(`Imported ${pendingDroppedSubjects.length} subject${pendingDroppedSubjects.length > 1 ? 's' : ''}`);
      setPendingDroppedSubjects([]);
    } catch (error) {
      toast.error('Project created, but failed to import dropped subjects');
      console.error(error);
    }
  };

  const kickoffWarm = useCallback(async () => {
    if (warmStatus === 'running' || warmStatus === 'done') return;
    setWarmStatus('running');
    setStartupNote('Warming backend (preloading imports)…');
    try {
      const res = await engine.warmBackend();
      if (res?.error) {
        setWarmStatus('failed');
        toast.warning('Backend warmup encountered an optional dependency issue. You can keep using the app.');
        console.warn(res.error);
      } else {
        setWarmStatus('done');
        setStartupNote('');
      }
    } catch (err) {
      setWarmStatus('failed');
      console.error(err);
    }
  }, [warmStatus]);

  useEffect(() => {
    let disposed = false;

    let pollTimer: number | null = null;
    let discoverTimer: number | null = null;
    let startupTimeout: number | null = null;

    const onBackendReady = () => {
      if (disposed) return;
      setBackendStatus('ready');
      setStartupNote('Backend ready. Loading projects…');
      loadProjects();
      void kickoffWarm();
    };

    const onBackendError = () => {
      if (disposed) return;
      try {
        const err = (globalThis as any)?.window?.__PBRAIN_BACKEND_ERROR;
        const msg = typeof err === 'string' && err.trim().length > 0 ? err : 'Backend startup failed';
        toast.error(msg);
        setBackendStatus('error');
        setStartupNote(msg);
      } catch {
        toast.error('Backend startup failed');
        setBackendStatus('error');
      }
    };

    window.addEventListener('pbrain-backend-ready', onBackendReady);
    window.addEventListener('pbrain-backend-error', onBackendError);

    // If the launcher injected the backend URL before React mounted, recover.
    pollTimer = window.setInterval(() => {
      if (disposed) return;
      try {
        const injected = (globalThis as any)?.window?.__PBRAIN_BACKEND_URL;
        if (typeof injected === 'string' && injected.trim().length > 0) {
          setBackendStatus('ready');
          setStartupNote('Backend ready. Loading projects…');
          loadProjects();
          void kickoffWarm();
          if (pollTimer != null) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
        }
      } catch {
        // ignore
      }
    }, 250);

    // If injection never arrives in non-Tauri local contexts, try to discover the backend.
    discoverTimer = window.setTimeout(() => {
      if (disposed) return;
      void (async () => {
        try {
          const injected = (globalThis as any)?.window?.__PBRAIN_BACKEND_URL;
          if (typeof injected === 'string' && injected.trim().length > 0) return;

          const found = await discoverLocalBackendBaseUrl();
          if (!found || disposed) return;

          (globalThis as any).window.__PBRAIN_BACKEND_URL = found;
          setBackendStatus('ready');
          setStartupNote('Backend ready. Loading projects…');
          loadProjects();
          void kickoffWarm();
        } catch {
          // ignore
        }
      })();
    }, 1200);

    // Never allow an infinite skeleton; surface a clear error instead.
    startupTimeout = window.setTimeout(() => {
      if (disposed) return;
      setIsLoading(false);
      try {
        const err = (globalThis as any)?.window?.__PBRAIN_BACKEND_ERROR;
        const msg = typeof err === 'string' && err.trim().length > 0 ? err : 'Backend not connected. Quit and reopen the app.';
        toast.error(msg);
        setBackendStatus('error');
        setStartupNote(msg);
      } catch {
        toast.error('Backend not connected. Quit and reopen the app.');
        setBackendStatus('error');
      }
    }, 120000);

    loadProjects();
    (async () => {
      try {
        const s = await engine.getSettings();
        setFirstName((s.firstName || '').trim());
      } catch {
        // ignore
      }
    })();

    return () => {
      disposed = true;
      window.removeEventListener('pbrain-backend-ready', onBackendReady);
      window.removeEventListener('pbrain-backend-error', onBackendError);

      if (pollTimer != null) clearInterval(pollTimer);
      if (discoverTimer != null) clearTimeout(discoverTimer);
      if (startupTimeout != null) clearTimeout(startupTimeout);
    };
  }, [kickoffWarm]);

  const loadProjects = async () => {
    setIsLoading(true);
    setStartupNote('Loading projects…');
    try {
      const data = await engine.getProjects();
      setProjects(data);
      setIsLoading(false);
      setStartupNote('');
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.includes('BACKEND_NOT_CONFIGURED')) {
        // Launchers may inject the backend URL asynchronously.
        // Keep the loading state until we receive pbrain-backend-ready.
        setStartupNote('Waiting for backend to finish starting…');
        return;
      }
      toast.error('Failed to load projects');
      console.error(error);
      setIsLoading(false);
      setStartupNote('Failed to load projects');
    }
  };

  const handleCreateProject = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const name = draftName.trim();
    if (!name) {
      toast.error('Please enter a project name');
      return;
    }

    const rawDraft = draftStoragePath.trim();
    let storagePath = (storeWithPatientData ? rawDraft : (rawDraft || defaultStoragePath(name))).trim();
    if (!storagePath) {
      toast.error(storeWithPatientData ? 'Please paste the patient data folder path' : 'Please enter a project storage path');
      return;
    }
    if (storeWithPatientData && !looksLikeAbsolutePath(storagePath)) {
      // Preferred: open a native folder picker on the backend machine.
      const ok = await checkLocalBackendHealth();
      if (ok) {
        const picked = await pickFolderWithNativeDialog();
        if (picked) {
          storagePath = picked;
          setDraftStoragePath(picked);
        }
      }

      // Fallback: try to infer from allowed roots when picker is unavailable.
      if (!looksLikeAbsolutePath(storagePath)) {
        const sampleSubject = pendingDroppedSubjects?.[0]?.name || undefined;
        const inferred = await resolveStoragePathFromDrop(storagePath, sampleSubject);
        if (inferred) {
          storagePath = inferred;
          setDraftStoragePath(inferred);
        }
      }
    }
    if (storeWithPatientData && !looksLikeAbsolutePath(storagePath)) {
      toast.error('Could not infer the absolute data folder path. Please paste it (e.g. /Volumes/.../data).');
      return;
    }

    try {
      const project = await engine.createProject({
        name,
        storagePath,
        // When not storing alongside patient data, copy data into the project folder.
        copyDataIntoProject: !storeWithPatientData,
      });

		// If the user configured folder matching rules, persist them before import.
		try {
			await engine.updateProjectConfig(project.id, { folderStructure: draftFolderStructure });
		} catch (err) {
			toast.error('Project created, but failed to save folder structure config');
			console.error(err);
		}
      
      toast.success('Project created successfully');
      setIsCreateDialogOpen(false);
      setDraftName('');
      setDraftStoragePath('');
      setStoreWithPatientData(true);
		setDraftFolderStructure(DEFAULT_FOLDER_STRUCTURE);
  		setPendingDroppedSubjects([]);
      loadProjects();

      await importDroppedSubjectsOnly(project);
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
      await engine.updateProject(editProject.id, { name, storagePath });
      toast.success('Project updated');
      setIsEditDialogOpen(false);
      setEditProject(null);
      loadProjects();
    } catch (error) {
      toast.error('Failed to update project');
      console.error(error);
    }
  };

  const openCreateDialog = (prefill?: { name?: string; storagePath?: string; subjects?: Array<{ name: string; sourcePath: string }> }) => {
    setDraftName(prefill?.name ?? '');
    setDraftStoragePath(prefill?.storagePath ?? '');
    setPendingDroppedSubjects(prefill?.subjects ?? []);
    setStoreWithPatientData(true);
		setDraftFolderStructure(DEFAULT_FOLDER_STRUCTURE);
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
      await engine.deleteProject(project.id);
      toast.success('Project deleted');
      loadProjects();
    } catch (error) {
      toast.error('Failed to delete project');
      console.error(error);
    }
  };

  const processDroppedItems = useCallback(async (items: DataTransferItemList) => {
    let inferredAbsPath: string | null = null;
    const entries: FileSystemDirectoryEntry[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file') {
        try {
          const f = item.getAsFile?.();
          const p = (f as any)?.path as string | undefined;
          if (!inferredAbsPath && typeof p === 'string' && p.trim()) inferredAbsPath = p.trim();
        } catch {
          // ignore
        }
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

    const listChildDirs = async (dir: FileSystemDirectoryEntry): Promise<string[]> => {
      const out: string[] = [];
      try {
        const reader = dir.createReader();
        while (true) {
          // readEntries is callback-based
          const batch: FileSystemEntry[] = await new Promise((resolve) => reader.readEntries(resolve));
          if (!batch || batch.length === 0) break;
          for (const e of batch) {
            if (e.isDirectory) out.push(e.name);
          }
        }
      } catch {
        // ignore
      }
      return out;
    };

    const childDirs = await listChildDirs(rootEntry);
    const subjectDirs = childDirs.filter(n => subjectIdRe.test(n));
    let subjects: Array<{ name: string; sourcePath: string }> = [];
    if (subjectDirs.length > 0) {
      // Store relative paths under the project storage path.
      subjects = subjectDirs.map(n => ({ name: n, sourcePath: n }));
    } else if (subjectIdRe.test(rootEntry.name)) {
      subjects = [{ name: rootEntry.name, sourcePath: rootEntry.name }];
    }

    // Prefill the folder name for convenience. If an absolute path is available (desktop shells/electron), use it.
    // Otherwise, the user must paste the patient data folder path.
    let storagePrefill = inferredAbsPath || rootEntry.name;
    if (!looksLikeAbsolutePath(storagePrefill)) {
      const ok = await checkLocalBackendHealth();
      if (ok) {
        // Preferred: native folder picker.
        const picked = await pickFolderWithNativeDialog();
        if (picked) {
          storagePrefill = picked;
        } else {
          const inferred = await resolveStoragePathFromDrop(storagePrefill, subjects?.[0]?.name);
          if (inferred) storagePrefill = inferred;
        }
      } else {
        toast.error('Local backend is not running, so the app cannot infer the absolute data folder path. Start the local app backend.');
      }
    }

    openCreateDialog({ name: rootEntry.name, storagePath: storagePrefill, subjects });
    if (subjects.length > 0) {
      toast.info('Folder dropped. Subject folders will be imported after project creation.');
    } else {
      toast.info('Folder dropped. No subject folders detected.');
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      e.dataTransfer.dropEffect = 'copy';
    } catch {
      // ignore
    }
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

  const handlePageDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDraggingCreate(false);

      // In packaged shells (Tauri/Electron), a dropped folder is commonly surfaced as a File with an absolute path.
      try {
        const f = e.dataTransfer.files?.[0] as any;
        const absPath = typeof f?.path === 'string' ? (f.path as string) : undefined;
        if (absPath && looksLikeAbsolutePath(absPath)) {
          handleDroppedCreateFolder(absPath);
          return;
        }
      } catch {
        // ignore
      }

      // Browser fallbacks (may not work for folders depending on engine/security policy).
      if (e.dataTransfer.items) {
        void processDroppedItems(e.dataTransfer.items);
      }
    },
    [handleDroppedCreateFolder, processDroppedItems]
  );

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingCreate(false);
    // In packaged shells (Tauri/Electron), DataTransferItem APIs may be empty but the File can carry an absolute path.
    try {
      const f = e.dataTransfer.files?.[0] as any;
      const absPath = typeof f?.path === 'string' ? (f.path as string) : undefined;
      if (absPath && looksLikeAbsolutePath(absPath)) {
        handleDroppedCreateFolder(absPath);
        return;
      }
    } catch {
      // ignore
    }

    if (e.dataTransfer.items) {
      void processDroppedItems(e.dataTransfer.items);
    }
  }, [handleDroppedCreateFolder, processDroppedItems]);

  const handleStoragePathDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();

      let inferredAbsPath: string | undefined;
      try {
        const f = e.dataTransfer.files?.[0];
        inferredAbsPath = (f as any)?.path as string | undefined;
      } catch {
        // ignore
      }
      if (!inferredAbsPath) {
        try {
          const item = e.dataTransfer.items?.[0];
          const f = item?.getAsFile?.();
          inferredAbsPath = (f as any)?.path as string | undefined;
        } catch {
          // ignore
        }
      }

      if (inferredAbsPath && looksLikeAbsolutePath(inferredAbsPath)) {
        applyDroppedStorageFolder(inferredAbsPath);
      } else {
        toast.error('Drop a folder from Finder to set the path');
      }
    },
    [applyDroppedStorageFolder]
  );

  const computedStoragePath = defaultStoragePath(draftName || 'project');

  const greeting = firstName ? `${greetingForLocalTime()}, ${firstName}` : `${greetingForLocalTime()}`;

  const showStartupBanner = backendStatus !== 'ready' || warmStatus === 'running' || (isLoading && projects.length === 0);
  let startupTitle = '';
  if (backendStatus !== 'ready') {
    startupTitle = 'Starting backend…';
  } else if (isLoading && projects.length === 0) {
    startupTitle = 'Connecting to backend…';
  } else if (warmStatus === 'running') {
    startupTitle = 'Warming backend…';
  }

  return (
    <div
      className="min-h-screen bg-background p-6"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handlePageDrop}
    >
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-medium tracking-tight text-foreground">
              <span className="italic">p</span>-Brain platform
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

                  <div className="space-y-2">
                    <Label htmlFor="storagePath">{storeWithPatientData ? 'Patient data folder' : 'Project storage path'}</Label>
                    <div className="flex gap-2">
                      <Input
                        id="storagePath"
                        name="storagePath"
                        placeholder={storeWithPatientData ? 'Choose or paste the patient data folder path' : 'Enter or paste a folder path'}
                        value={draftStoragePath}
                        onChange={(e) => setDraftStoragePath(e.target.value)}
                        onDragOver={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                        }}
                        onDrop={handleStoragePathDrop}
                      />
                      {storeWithPatientData ? (
                        <Button
                          type="button"
                          variant="secondary"
                          disabled={isPickingFolder}
                          onClick={async () => {
                            setIsPickingFolder(true);
                            try {
                              const ok = await checkLocalBackendHealth();
                              if (!ok) {
                                toast.error('Local backend is not running. Start the local app backend to choose a folder.');
                                return;
                              }
                              const picked = await pickFolderWithNativeDialog();
                              if (picked) setDraftStoragePath(picked);
                            } finally {
                              setIsPickingFolder(false);
                            }
                          }}
                        >
                          Choose…
                        </Button>
                      ) : null}
                    </div>
                  </div>
                </div>

        {pendingDroppedSubjects.length > 0 ? (
          <div className="space-y-3 rounded-lg border border-border p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <Label>Detected subjects</Label>
                <p className="text-xs text-muted-foreground">
                  {pendingDroppedSubjects.length} folder{pendingDroppedSubjects.length > 1 ? 's' : ''} will be imported after project creation.
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <Label className="text-xs">Subject folder pattern</Label>
                <Input
                  value={draftFolderStructure.subjectFolderPattern}
                  onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, subjectFolderPattern: e.target.value }))}
                  placeholder="{subject_id}"
                  className="mono"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">NIfTI subfolder</Label>
                <Input
                  value={draftFolderStructure.niftiSubfolder}
                  onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, niftiSubfolder: e.target.value }))}
                  placeholder="NIfTI"
                  className="mono"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <Label>Treat data as nested structure</Label>
                  <p className="text-xs text-muted-foreground">If enabled, the worker looks under the NIfTI subfolder for volumes.</p>
                </div>
                <Switch
                  checked={draftFolderStructure.useNestedStructure}
                  onCheckedChange={(checked) => setDraftFolderStructure(prev => ({ ...prev, useNestedStructure: checked }))}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label className="text-xs">T1 filename patterns (comma-separated)</Label>
              <Input
                value={draftFolderStructure.t1Pattern}
                onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, t1Pattern: e.target.value }))}
                className="mono"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">DCE filename patterns (comma-separated)</Label>
              <Input
                value={draftFolderStructure.dcePattern}
                onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, dcePattern: e.target.value }))}
                className="mono"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Diffusion filename patterns (comma-separated)</Label>
              <Input
                value={draftFolderStructure.diffusionPattern}
                onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, diffusionPattern: e.target.value }))}
                className="mono"
              />
            </div>
          </div>
        ) : null}

                <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground space-y-2">
                  <p className="font-medium text-foreground">Data handling</p>
                  <p>The backend processes subjects directly from the storage path and stores logs/indexes in a <span className="mono">.pbrain-web</span> folder under it.</p>
                  <p>{storeWithPatientData ? 'Project state will live beside the patient data (in-place).' : 'Project state will live in the custom path you provide (data copied into project).'}
                  </p>
                  {!storeWithPatientData && draftName.trim() ? (
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

        {showStartupBanner ? (
          <div className="mb-6 rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
            <div className="font-medium text-foreground">{startupTitle || 'Starting…'}</div>
            <div className="mt-1 text-muted-foreground">
              {startupNote || 'Waiting for the local backend to finish starting.'}
            </div>
            {warmStatus === 'running' ? (
              <div className="mt-1 text-xs text-muted-foreground">Preloading heavy imports so your first run is faster…</div>
            ) : null}
          </div>
        ) : null}

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
