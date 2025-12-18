import type {
  Curve,
  DeconvolutionData,
  Job,
  MapVolume,
  MetricsTable,
  PatlakData,
  Project,
  Subject,
  ToftsData,
  VolumeFile,
  VolumeInfo,
} from '@/types';

type Unsubscribe = () => void;

function backendUrl(): string {
  return (import.meta as any).env?.VITE_BACKEND_URL || 'http://127.0.0.1:8787';
}

export function getBackendBaseUrl(): string {
  return backendUrl();
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${backendUrl()}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export class BackendEngineAPI {
  private jobListeners = new Set<(job: Job) => void>();
  private statusListeners = new Set<(update: { subjectId: string; stageId: any; status: any }) => void>();
  private logListeners = new Map<string, Set<(log: string) => void>>();

  private pollTimer: number | null = null;
  private lastJobs = new Map<string, Job>();
  private lastSubjects = new Map<string, Subject>();

  constructor() {
    this.startPolling();
  }

  private startPolling() {
    if (this.pollTimer != null) return;

    this.pollTimer = window.setInterval(async () => {
      try {
        const jobs = await this.getJobs({});
        for (const job of jobs) {
          const prev = this.lastJobs.get(job.id);
          if (!prev || JSON.stringify(prev) !== JSON.stringify(job)) {
            this.jobListeners.forEach(l => l(job));
            this.lastJobs.set(job.id, job);
          }
        }

        // Emit status updates by diffing subjects (cheap, coarse).
        const projects = await this.getProjects();
        for (const p of projects) {
          const subjects = await this.getSubjects(p.id);
          for (const s of subjects) {
            const prev = this.lastSubjects.get(s.id);
            if (prev) {
              for (const [stageId, status] of Object.entries(s.stageStatuses)) {
                const old = (prev.stageStatuses as any)[stageId];
                if (old !== status) {
                  this.statusListeners.forEach(l => l({ subjectId: s.id, stageId: stageId as any, status }));
                }
              }
            }
            this.lastSubjects.set(s.id, s);
          }
        }

        // Poll logs for any jobs that have listeners.
        for (const [jobId, listeners] of this.logListeners.entries()) {
          if (listeners.size === 0) continue;
          const payload = await api<{ lines: string[] }>(`/jobs/${encodeURIComponent(jobId)}/logs?tail=80`);
          // naive: emit all lines each time; UI de-dupes by appending, so we only emit the last line.
          const last = payload.lines[payload.lines.length - 1];
          if (last) listeners.forEach(l => l(last));
        }
      } catch {
        // ignore transient backend errors
      }
    }, 1200);
  }

  onJobUpdate(listener: (job: Job) => void): Unsubscribe {
    this.jobListeners.add(listener);
    return () => this.jobListeners.delete(listener);
  }

  onStatusUpdate(listener: (update: { subjectId: string; stageId: any; status: any }) => void): Unsubscribe {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  onJobLogs(jobId: string, listener: (log: string) => void): Unsubscribe {
    if (!this.logListeners.has(jobId)) this.logListeners.set(jobId, new Set());
    this.logListeners.get(jobId)!.add(listener);
    return () => {
      const set = this.logListeners.get(jobId);
      if (!set) return;
      set.delete(listener);
      if (set.size === 0) this.logListeners.delete(jobId);
    };
  }

  async getProjects(): Promise<Project[]> {
    return api<Project[]>('/projects');
  }

  async getProject(id: string): Promise<Project | undefined> {
    return api<Project>(`/projects/${encodeURIComponent(id)}`);
  }

  async createProject(data: { name: string; storagePath: string; copyDataIntoProject: boolean }): Promise<Project> {
    return api<Project>('/projects', { method: 'POST', body: JSON.stringify(data) });
  }

  async deleteProject(projectId: string): Promise<void> {
    await api(`/projects/${encodeURIComponent(projectId)}`, { method: 'DELETE' });
  }

  async updateProjectConfig(projectId: string, configUpdate: any): Promise<Project | undefined> {
    return api<Project>(`/projects/${encodeURIComponent(projectId)}/config`, {
      method: 'PATCH',
      body: JSON.stringify({ configUpdate }),
    });
  }

  async getSubjects(projectId: string): Promise<Subject[]> {
    return api<Subject[]>(`/projects/${encodeURIComponent(projectId)}/subjects`);
  }

  async getSubject(id: string): Promise<Subject | undefined> {
    return api<Subject>(`/subjects/${encodeURIComponent(id)}`);
  }

  async importSubjects(projectId: string, subjects: Array<{ name: string; sourcePath: string }>): Promise<Subject[]> {
    return api<Subject[]>(`/projects/${encodeURIComponent(projectId)}/subjects/import`, {
      method: 'POST',
      body: JSON.stringify({ subjects }),
    });
  }

  async runFullPipeline(projectId: string, subjectIds: string[]): Promise<Job[]> {
    return api<Job[]>(`/projects/${encodeURIComponent(projectId)}/run-full`, {
      method: 'POST',
      body: JSON.stringify({ subjectIds }),
    });
  }

  async getJobs(filters: { projectId?: string; subjectId?: string; status?: string }): Promise<Job[]> {
    const params = new URLSearchParams();
    if (filters.projectId) params.set('projectId', filters.projectId);
    if (filters.subjectId) params.set('subjectId', filters.subjectId);
    if (filters.status) params.set('status', filters.status);
    const qs = params.toString();
    return api<Job[]>(`/jobs${qs ? `?${qs}` : ''}`);
  }

  async cancelJob(jobId: string): Promise<void> {
    await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
  }

  async retryJob(jobId: string): Promise<Job> {
    return api<Job>(`/jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST' });
  }

  async resolveDefaultVolume(subjectId: string, kind: 'dce' | 't1' | 'diffusion' = 'dce'): Promise<string> {
    const res = await api<{ path: string }>(
      `/subjects/${encodeURIComponent(subjectId)}/default-volume?kind=${encodeURIComponent(kind)}`
    );
    return res.path;
  }

  // ----------------------------
  // Volumes (real backend)
  // ----------------------------

  async getVolumeInfo(path: string, subjectId?: string): Promise<VolumeInfo> {
    if (!subjectId) throw new Error('BackendEngineAPI.getVolumeInfo requires subjectId');

    return api<VolumeInfo>(`/volumes/info?subjectId=${encodeURIComponent(subjectId)}`, {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
  }

  async getSliceData(
    path: string,
    z: number,
    t: number = 0,
    subjectId?: string
  ): Promise<{ data: number[][]; min: number; max: number }> {
    if (!subjectId) throw new Error('BackendEngineAPI.getSliceData requires subjectId');

    return api<{ data: number[][]; min: number; max: number }>(
      `/volumes/slice?subjectId=${encodeURIComponent(subjectId)}`,
      { method: 'POST', body: JSON.stringify({ path, z, t }) }
    );
  }

  async getCurves(subjectId: string): Promise<Curve[]> {
    const res = await api<{ curves: Curve[] }>(`/subjects/${encodeURIComponent(subjectId)}/curves`);
    return res.curves;
  }

  async getMapVolumes(subjectId: string): Promise<MapVolume[]> {
    const res = await api<{ maps: MapVolume[] }>(`/subjects/${encodeURIComponent(subjectId)}/maps`);
    return res.maps;
  }

  async getVolumes(subjectId: string): Promise<VolumeFile[]> {
    const res = await api<{ volumes: VolumeFile[] }>(`/subjects/${encodeURIComponent(subjectId)}/volumes`);
    return res.volumes;
  }

  async getMontageImages(subjectId: string): Promise<Array<{ id: string; name: string; path: string }>> {
    const res = await api<{ montages: Array<{ id: string; name: string; path: string }> }>(
      `/subjects/${encodeURIComponent(subjectId)}/montages`
    );
    return res.montages;
  }

  async ensureSubjectArtifacts(
    subjectId: string,
    kind: 'all' | 'maps' | 'curves' | 'montages' = 'all'
  ): Promise<{ started: boolean; jobs: any[]; reason: string }> {
    return api<{ started: boolean; jobs: any[]; reason: string }>(
      `/subjects/${encodeURIComponent(subjectId)}/ensure?kind=${encodeURIComponent(kind)}`,
      { method: 'POST', body: JSON.stringify({}) }
    );
  }

  async getPatlakData(subjectId: string, region: string): Promise<PatlakData> {
    return api<PatlakData>(
      `/subjects/${encodeURIComponent(subjectId)}/patlak?region=${encodeURIComponent(region)}`
    );
  }

  async getToftsData(subjectId: string, region: string): Promise<ToftsData> {
    return api<ToftsData>(
      `/subjects/${encodeURIComponent(subjectId)}/tofts?region=${encodeURIComponent(region)}`
    );
  }

  async getDeconvolutionData(subjectId: string, region: string): Promise<DeconvolutionData> {
    return api<DeconvolutionData>(
      `/subjects/${encodeURIComponent(subjectId)}/deconvolution?region=${encodeURIComponent(region)}`
    );
  }

  async getMetricsTable(subjectId: string): Promise<MetricsTable> {
    return api<MetricsTable>(`/subjects/${encodeURIComponent(subjectId)}/metrics`);
  }
}
