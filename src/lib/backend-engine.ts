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

const STORAGE_KEY = 'pbrain.backendUrl';

function sanitizeUrl(raw: string | null): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (!/^https?:\/\//i.test(trimmed)) return null;
  return trimmed.replace(/\/$/, '');
}

function readBackendOverride(): string | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const raw = params.get('backend');
    const cleaned = sanitizeUrl(raw);
    if (cleaned) {
      try {
        window.localStorage.setItem(STORAGE_KEY, cleaned);
      } catch {
        /* ignore */
      }
      return cleaned;
    }
    return null;
  } catch {
    return null;
  }
}

function readStoredBackend(): string | null {
  try {
    return sanitizeUrl(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    return null;
  }
}

export function setBackendOverride(url: string): string | null {
  const cleaned = sanitizeUrl(url);
  if (!cleaned) return null;
  try {
    window.localStorage.setItem(STORAGE_KEY, cleaned);
  } catch {
    /* ignore */
  }
  return cleaned;
}

function backendUrl(): string | null {
  const override = readBackendOverride();
  if (override) return override;

  const envUrl = (import.meta as any).env?.VITE_BACKEND_URL as string | undefined;
  if (envUrl) return sanitizeUrl(envUrl);

  const stored = readStoredBackend();
  if (stored) return stored;

  // If hosted on GitHub Pages (https), avoid defaulting to localhost which will be blocked/refused.
  try {
    const host = (window.location.hostname || '').toLowerCase();
    const isGithubPages = host.endsWith('github.io');
    if (isGithubPages) return null;
  } catch {
    /* ignore */
  }

  // Fallbacks for local dev.
  try {
    if (window.location.protocol === 'https:') return 'https://127.0.0.1:8787';
  } catch {
    /* ignore */
  }
  return 'http://127.0.0.1:8787';
}

export function backendConfigured(): boolean {
  return !!backendUrl();
}

export function getBackendBaseUrl(): string {
  const url = backendUrl();
  if (!url) throw new Error('BACKEND_NOT_CONFIGURED');
  return url;
}

type HealthResponse = { ok: boolean };

let lastHealthCheckAt = 0;
let lastHealthOk: boolean | null = null;
let backoffMs = 0;

async function isBackendHealthy(): Promise<boolean> {
  const now = Date.now();
  const wait = Math.max(1200, backoffMs || 0);
  if (lastHealthOk !== null && now - lastHealthCheckAt < wait) return lastHealthOk;

  lastHealthCheckAt = now;
  try {
    const res = await fetch(`${backendUrl()}/health`, {
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });
    lastHealthOk = res.ok;
    backoffMs = 0;
    return lastHealthOk;
  } catch {
    lastHealthOk = false;
    backoffMs = Math.min(Math.max(backoffMs || 1200, 1200) * 2, 15000);
    return false;
  }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const url = backendUrl();
  if (!url) throw new Error('BACKEND_NOT_CONFIGURED');

  const res = await fetch(`${url}${path}`, {
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

  private lastLogLines = new Map<string, string[]>();

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
        if (!backendConfigured()) return;
        // Avoid spamming the network (and console) when the local backend is not reachable.
        const ok = await isBackendHealthy();
        if (!ok) return;

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

          const nextLines = payload.lines || [];
          const prevLines = this.lastLogLines.get(jobId) || [];

          // Fast path: log only grew.
          let startIdx = 0;
          if (prevLines.length > 0 && nextLines.length >= prevLines.length) {
            let isPrefix = true;
            for (let i = 0; i < prevLines.length; i++) {
              if (prevLines[i] !== nextLines[i]) {
                isPrefix = false;
                break;
              }
            }
            if (isPrefix) startIdx = prevLines.length;
          }

          // Fallback: try to find an overlap (handles tail=80 sliding window).
          if (startIdx === 0 && prevLines.length > 0 && nextLines.length > 0) {
            const maxLookback = Math.min(20, prevLines.length);
            let overlap = 0;
            for (let k = maxLookback; k >= 1; k--) {
              const prevTail = prevLines.slice(prevLines.length - k);
              let matches = true;
              for (let i = 0; i < k && i < nextLines.length; i++) {
                if (prevTail[i] !== nextLines[i]) {
                  matches = false;
                  break;
                }
              }
              if (matches) {
                overlap = k;
                break;
              }
            }
            if (overlap > 0) startIdx = overlap;
          }

          for (const line of nextLines.slice(startIdx)) {
            listeners.forEach(l => l(line));
          }
          this.lastLogLines.set(jobId, nextLines);
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
      if (set.size === 0) {
        this.logListeners.delete(jobId);
        this.lastLogLines.delete(jobId);
      }
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

  async scanProjectSubjects(projectId: string): Promise<{ subjects: Array<{ name: string; sourcePath: string }> }> {
    return api<{ subjects: Array<{ name: string; sourcePath: string }> }>(
      `/projects/${encodeURIComponent(projectId)}/scan-subject-folders`
    );
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
