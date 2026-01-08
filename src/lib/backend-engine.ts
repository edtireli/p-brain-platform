import type {
  AppSettings,
  Curve,
  DeconvolutionData,
  InstallPBrainRequirementsResponse,
  InstallPBrainResponse,
  Job,
  MapVolume,
  MetricsTable,
  PatlakData,
  Project,
  RoiOverlay,
  RoiMaskVolume,
  ScanSystemDepsResponse,
  Subject,
  SystemDeps,
  ToftsData,
  TractographyData,
  VolumeFile,
  VolumeInfo,
} from '@/types';
import type { StageId } from '@/types';

type Unsubscribe = () => void;

const STORAGE_KEY = 'pbrain.backendUrl';

let tauriDiscoveryStarted = false;
let tauriDiscoveryTimer: number | null = null;
let tauriDiscoveryAttempts = 0;

const TAURI_DISCOVERY_MAX_ATTEMPTS = 120; // ~2 minutes at 1s interval
const TAURI_DISCOVERY_INTERVAL_MS = 1000;

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

function startTauriBackendDiscovery(): void {
  if (tauriDiscoveryStarted) return;
  tauriDiscoveryStarted = true;

  const ports: number[] = [];
  for (let p = 8787; p <= 8887; p++) ports.push(p);

  const publishFound = (found: string) => {
    try {
      (globalThis as any).window.__PBRAIN_BACKEND_URL = found;
      window.dispatchEvent(new Event('pbrain-backend-ready'));
    } catch {
      // ignore
    }
  };

  const scanOnce = async (): Promise<boolean> => {
    const batchSize = 16;
    for (let i = 0; i < ports.length; i += batchSize) {
      const batch = ports.slice(i, i + batchSize);
      const bases = batch.map(p => `http://127.0.0.1:${p}`);
      const results = await Promise.all(bases.map(async b => ((await probeHealth(b, 250)) ? b : null)));
      const found = results.find(Boolean) as string | null;
      if (found) {
        publishFound(found);
        return true;
      }
    }
    return false;
  };

  const attempt = async () => {
    tauriDiscoveryAttempts += 1;
    const ok = await scanOnce();
    if (ok && tauriDiscoveryTimer !== null) {
      clearInterval(tauriDiscoveryTimer);
      tauriDiscoveryTimer = null;
    }
    if (!ok && tauriDiscoveryAttempts >= TAURI_DISCOVERY_MAX_ATTEMPTS && tauriDiscoveryTimer !== null) {
      clearInterval(tauriDiscoveryTimer);
      tauriDiscoveryTimer = null;
    }
  };

  // Start immediately, then keep retrying while the backend spins up.
  void attempt();
  tauriDiscoveryTimer = window.setInterval(() => {
    void attempt();
  }, TAURI_DISCOVERY_INTERVAL_MS);
}

function sanitizeUrl(raw: string | null): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (!/^https?:\/\//i.test(trimmed)) return null;
  const noTrailingSlash = trimmed.replace(/\/$/, '');
  // Accept common convention of passing an API base that includes `/api`.
  // p-brain-web backend routes are mounted at the root.
  return noTrailingSlash.endsWith('/api') ? noTrailingSlash.slice(0, -4) : noTrailingSlash;
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
    const cleaned = sanitizeUrl(window.localStorage.getItem(STORAGE_KEY));
    if (!cleaned) return null;

    return cleaned;
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
  const hasTauri = typeof (globalThis as any)?.window?.__TAURI__ !== 'undefined';

  // Launchers (Electron/Tauri) can inject a concrete backend base URL.
  // This is the most reliable option when the UI isn't served by the backend.
  try {
    const injected = (globalThis as any)?.window?.__PBRAIN_BACKEND_URL;
    const cleaned = sanitizeUrl(typeof injected === 'string' ? injected : null);
    if (cleaned) return cleaned;
  } catch {
    // ignore
  }

  // In the packaged Tauri app, the backend port can change every launch.
  // Never fall back to a stale localStorage value; wait for the launcher.
  // (Allow explicit URL override via query string for debugging only.)
  if (hasTauri) {
    // Safety net: if injection doesn't arrive (or races webview init), discover by probing ports.
    startTauriBackendDiscovery();
    const override = readBackendOverride();
    if (override) return override;
    return null;
  }

  const override = readBackendOverride();
  if (override) return override;

  const env = (import.meta as any).env as Record<string, string | undefined> | undefined;
  const envUrl = env?.VITE_API_BASE_URL || env?.VITE_BACKEND_URL;
  if (envUrl) return sanitizeUrl(envUrl);

  const stored = readStoredBackend();
  if (stored) return stored;

  // Local app default: if UI is served from the local backend, use same-origin.
  try {
    const host = window.location.hostname;
    if (host === '127.0.0.1' || host === 'localhost') {
      return sanitizeUrl(window.location.origin);
    }
  } catch {
    // ignore
  }

  return null;
}

export function backendConfigured(): boolean {
  return !!backendUrl();
}

export function getBackendBaseUrl(): string {
  const url = backendUrl();
  if (!url) throw new Error('BACKEND_NOT_CONFIGURED');
  return url;
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

  async getSubjectRoiOverlays(subjectId: string): Promise<RoiOverlay[]> {
    const payload = await api<{ overlays: RoiOverlay[] }>(`/subjects/${encodeURIComponent(subjectId)}/roi-overlays`);
    return Array.isArray(payload?.overlays) ? payload.overlays : [];
  }

  async getSubjectRoiMasks(subjectId: string): Promise<RoiMaskVolume[]> {
    const payload = await api<{ masks: RoiMaskVolume[] }>(`/subjects/${encodeURIComponent(subjectId)}/roi-masks`);
    return Array.isArray(payload?.masks) ? payload.masks : [];
  }

  async getSubjectTractography(subjectId: string): Promise<TractographyData> {
    return api<TractographyData>(`/subjects/${encodeURIComponent(subjectId)}/tractography`);
  }

  // ----------------------------
  // App settings (onboarding)
  // ----------------------------

  async getSettings(): Promise<AppSettings> {
    return api<AppSettings>('/settings');
  }

  async updateSettings(patch: Partial<AppSettings>): Promise<AppSettings> {
    return api<AppSettings>('/settings', { method: 'PATCH', body: JSON.stringify(patch) });
  }

  async getSystemDeps(): Promise<SystemDeps> {
    return api<SystemDeps>('/system/deps');
  }

  async scanSystemDeps(apply: boolean = true): Promise<ScanSystemDepsResponse> {
    return api<ScanSystemDepsResponse>('/system/deps/scan', {
      method: 'POST',
      body: JSON.stringify({ apply }),
    });
  }

  async installPBrain(installDir: string): Promise<InstallPBrainResponse> {
    return api<InstallPBrainResponse>('/system/deps/pbrain/install', {
      method: 'POST',
      body: JSON.stringify({ installDir }),
    });
  }

  async installPBrainRequirements(pbrainDir?: string): Promise<InstallPBrainRequirementsResponse> {
    return api<InstallPBrainRequirementsResponse>('/system/deps/pbrain/requirements/install', {
      method: 'POST',
      body: JSON.stringify({ pbrainDir: pbrainDir || null }),
    });
  }

  async warmBackend(): Promise<{ started: boolean; done?: boolean; error?: string; steps?: Record<string, number> }> {
    return api('/system/warm', { method: 'POST', body: JSON.stringify({}) });
  }

  async installFastSurfer(installDir: string): Promise<{ ok: boolean; fastsurferDir: string }> {
    return api<{ ok: boolean; fastsurferDir: string }>('/system/deps/fastsurfer/install', {
      method: 'POST',
      body: JSON.stringify({ installDir }),
    });
  }

  async getProject(id: string): Promise<Project | undefined> {
    return api<Project>(`/projects/${encodeURIComponent(id)}`);
  }

  async createProject(data: { name: string; storagePath: string; copyDataIntoProject: boolean }): Promise<Project> {
    return api<Project>('/projects', { method: 'POST', body: JSON.stringify(data) });
  }

  async updateProject(
    projectId: string,
    data: { name?: string; storagePath?: string; copyDataIntoProject?: boolean }
  ): Promise<Project> {
    return api<Project>(`/projects/${encodeURIComponent(projectId)}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
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

  async runSubjectStage(subjectId: string, stageId: StageId): Promise<Job> {
    return api<Job>(`/subjects/${encodeURIComponent(subjectId)}/run-stage`, {
      method: 'POST',
      body: JSON.stringify({ stageId }),
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

  async cancelAllJobs(filters: { projectId?: string } = {}): Promise<{ cancelled: number; terminated: number }> {
    const params = new URLSearchParams();
    if (filters.projectId) params.set('projectId', filters.projectId);
    const qs = params.toString();
    return api<{ cancelled: number; terminated: number }>(`/jobs/cancel-all${qs ? `?${qs}` : ''}`, { method: 'POST' });
  }

  async retryJob(jobId: string): Promise<Job> {
    return api<Job>(`/jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST' });
  }

  async resolveDefaultVolume(
    subjectId: string,
    kind: 'dce' | 't1' | 't2' | 'flair' | 'diffusion' | 'map' = 'dce'
  ): Promise<string> {
    const effectiveKind = kind === 'map' ? 'dce' : kind;
    const res = await api<{ path: string }>(
      `/subjects/${encodeURIComponent(subjectId)}/default-volume?kind=${encodeURIComponent(effectiveKind)}`
    );
    return res.path;
  }

  // ----------------------------
  // Volumes (real backend)
  // ----------------------------

  async getVolumeInfo(path: string, subjectId?: string, kind?: string): Promise<VolumeInfo> {
    if (!subjectId) throw new Error('BackendEngineAPI.getVolumeInfo requires subjectId');

    return api<VolumeInfo>(`/volumes/info?subjectId=${encodeURIComponent(subjectId)}`, {
      method: 'POST',
      body: JSON.stringify({ path, kind }),
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
    kind: 'all' | 'maps' | 'curves' | 'montages' | 'roi' = 'all'
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

  async getMetricsTable(subjectId: string, view: 'atlas' | 'tissue' = 'atlas'): Promise<MetricsTable> {
    return api<MetricsTable>(
      `/subjects/${encodeURIComponent(subjectId)}/metrics?view=${encodeURIComponent(view)}`
    );
  }
}
