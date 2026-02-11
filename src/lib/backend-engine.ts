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
  ConnectomeData,
  VolumeFile,
  VolumeInfo,
  ProjectAnalysisDataset,
  ProjectAnalysisView,
  ProjectAnalysisPearsonResponse,
  ProjectAnalysisGroupCompareResponse,
  ProjectAnalysisOlsResponse,
  InputFunctionForces,
  ForcedRoiRef,
} from '@/types';
import type { StageId } from '@/types';

type Unsubscribe = () => void;

const STORAGE_KEY = 'pbrain.backendUrl';

let tauriDiscoveryStarted = false;
let tauriDiscoveryTimer: number | null = null;
let tauriDiscoveryAttempts = 0;
let tauriDiscoveryInFlight = false;

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

  const stop = () => {
    if (tauriDiscoveryTimer !== null) {
      clearTimeout(tauriDiscoveryTimer);
      tauriDiscoveryTimer = null;
    }
  };

  const scheduleNext = (delayMs: number) => {
    stop();
    tauriDiscoveryTimer = window.setTimeout(() => {
      void attempt();
    }, delayMs);
  };

  const attempt = async () => {
    if (tauriDiscoveryInFlight) return;
    tauriDiscoveryInFlight = true;
    try {
      tauriDiscoveryAttempts += 1;
      const ok = await scanOnce();
      if (ok) {
        stop();
        return;
      }
      if (tauriDiscoveryAttempts >= TAURI_DISCOVERY_MAX_ATTEMPTS) {
        stop();
        return;
      }
      scheduleNext(TAURI_DISCOVERY_INTERVAL_MS);
    } finally {
      tauriDiscoveryInFlight = false;
    }
  };

  // Start immediately, then retry while the backend spins up.
  void attempt();
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

  const timeoutMs = 30_000;
  const timeoutController = new AbortController();
  const timeout = window.setTimeout(() => timeoutController.abort(), timeoutMs);
  const signals: AbortSignal[] = [];
  if (init?.signal) signals.push(init.signal);
  signals.push(timeoutController.signal);

  let signal: AbortSignal | undefined;
  if (signals.length === 1) {
    signal = signals[0];
  } else {
    const anyFn = (AbortSignal as any)?.any as ((signals: AbortSignal[]) => AbortSignal) | undefined;
    if (typeof anyFn === 'function') {
      signal = anyFn(signals);
    } else {
      const ctrl = new AbortController();
      const onAbort = () => ctrl.abort();
      for (const s of signals) {
        if (s.aborted) {
          ctrl.abort();
          break;
        }
        s.addEventListener('abort', onAbort, { once: true });
      }
      signal = ctrl.signal;
    }
  }

  try {
    const res = await fetch(`${url}${path}`, {
      ...init,
      signal,
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
  } finally {
    clearTimeout(timeout);
  }
}

export class BackendEngineAPI {
  private jobListeners = new Set<(job: Job) => void>();
  private statusListeners = new Set<(update: { subjectId: string; stageId: any; status: any }) => void>();
  private logListeners = new Map<string, Set<(log: string) => void>>();

  private lastLogLines = new Map<string, string[]>();

  private pollTimer: number | null = null;
  private pollAbort: AbortController | null = null;
  private pollInFlight = false;
  private disposed = false;

  private lastStatusPollAt = 0;
  private lastJobs = new Map<string, Job>();
  private lastSubjects = new Map<string, Subject>();

  constructor() {}

  async restartBackend(): Promise<{ ok: boolean; willExit?: boolean; delayMs?: number }> {
    return api<{ ok: boolean; willExit?: boolean; delayMs?: number }>(`/system/backend/restart`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
  }

  async refreshStageRunners(): Promise<{ ok: boolean; refreshed: string[]; skipped: string[] }> {
    return api<{ ok: boolean; refreshed: string[]; skipped: string[] }>(`/system/backend/refresh-runners`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
  }

  async getSubjectConnectome(subjectId: string): Promise<ConnectomeData> {
    return api<ConnectomeData>(`/subjects/${encodeURIComponent(subjectId)}/connectome`);
  }

  dispose(): void {
    this.disposed = true;
    this.stopPolling();
    this.jobListeners.clear();
    this.statusListeners.clear();
    this.logListeners.clear();
    this.lastJobs.clear();
    this.lastSubjects.clear();
    this.lastLogLines.clear();
  }

  private hasLiveListeners(): boolean {
    if (this.jobListeners.size > 0) return true;
    if (this.statusListeners.size > 0) return true;
    for (const listeners of this.logListeners.values()) {
      if (listeners.size > 0) return true;
    }
    return false;
  }

  private ensurePolling(): void {
    if (this.disposed) return;
    if (!this.hasLiveListeners()) return;
    if (this.pollTimer != null) return;
    this.scheduleNextPoll(0);
  }

  private stopPolling(): void {
    if (this.pollTimer != null) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.pollAbort) {
      try {
        this.pollAbort.abort();
      } catch {
        // ignore
      }
      this.pollAbort = null;
    }
    this.pollInFlight = false;
  }

  private maybeStopPolling(): void {
    if (this.hasLiveListeners()) return;
    this.stopPolling();
  }

  private scheduleNextPoll(delayMs: number): void {
    if (this.disposed) return;
    if (this.pollTimer != null) clearTimeout(this.pollTimer);
    this.pollTimer = window.setTimeout(() => {
      void this.pollOnce();
    }, delayMs);
  }

  private jobsEqual(a: Job, b: Job): boolean {
    return (
      a.status === b.status &&
      a.progress === b.progress &&
      a.currentStep === b.currentStep &&
      a.estimatedTimeRemaining === b.estimatedTimeRemaining &&
      a.startTime === b.startTime &&
      a.endTime === b.endTime &&
      a.error === b.error
    );
  }

  private async pollOnce(): Promise<void> {
    if (this.disposed) return;
    if (!this.hasLiveListeners()) {
      this.stopPolling();
      return;
    }
    if (this.pollInFlight) {
      // Avoid overlapping polls (can dogpile async fetches).
      this.scheduleNextPoll(250);
      return;
    }

    this.pollInFlight = true;
    this.pollAbort = new AbortController();
    const signal = this.pollAbort.signal;

    try {
      if (!backendConfigured()) {
        this.scheduleNextPoll(300);
        return;
      }

      // Poll jobs only if someone is listening.
      if (this.jobListeners.size > 0) {
        const jobs = await api<Job[]>('/jobs', { signal });
        for (const job of jobs) {
          const prev = this.lastJobs.get(job.id);
          if (!prev || !this.jobsEqual(prev, job)) {
            this.jobListeners.forEach(l => l(job));
            this.lastJobs.set(job.id, job);
          }
        }
      }

      // Emit status updates by diffing subjects, but do it less often and only when needed.
      if (this.statusListeners.size > 0) {
        const now = Date.now();
        if (now - this.lastStatusPollAt >= 5000) {
          this.lastStatusPollAt = now;
          const projects = await api<Project[]>('/projects', { signal });
          for (const p of projects) {
            const subjects = await api<Subject[]>(`/projects/${encodeURIComponent(p.id)}/subjects`, { signal });
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
        }
      }

      // Poll logs for any jobs that have listeners.
      for (const [jobId, listeners] of this.logListeners.entries()) {
        if (listeners.size === 0) continue;
        const payload = await api<{ lines: string[] }>(`/jobs/${encodeURIComponent(jobId)}/logs?tail=80`, { signal });

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
    } finally {
      this.pollInFlight = false;
      this.pollAbort = null;
      this.scheduleNextPoll(1200);
    }
  }

  onJobUpdate(listener: (job: Job) => void): Unsubscribe {
    this.jobListeners.add(listener);
    this.ensurePolling();
    return () => {
      this.jobListeners.delete(listener);
      this.maybeStopPolling();
    };
  }

  onStatusUpdate(listener: (update: { subjectId: string; stageId: any; status: any }) => void): Unsubscribe {
    this.statusListeners.add(listener);
    this.ensurePolling();
    return () => {
      this.statusListeners.delete(listener);
      this.maybeStopPolling();
    };
  }

  onJobLogs(jobId: string, listener: (log: string) => void): Unsubscribe {
    if (!this.logListeners.has(jobId)) this.logListeners.set(jobId, new Set());
    this.logListeners.get(jobId)!.add(listener);
    this.ensurePolling();
    return () => {
      const set = this.logListeners.get(jobId);
      if (!set) return;
      set.delete(listener);
      if (set.size === 0) {
        this.logListeners.delete(jobId);
        this.lastLogLines.delete(jobId);
      }
      this.maybeStopPolling();
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

  async getSubjectInputFunctionForces(subjectId: string): Promise<InputFunctionForces> {
    const payload = await api<InputFunctionForces>(
      `/subjects/${encodeURIComponent(subjectId)}/input-function-forces`
    );
    return {
      forcedAif: (payload as any)?.forcedAif ?? null,
      forcedVif: (payload as any)?.forcedVif ?? null,
    };
  }

  async setSubjectInputFunctionForces(
    subjectId: string,
    forces: { forcedAif: ForcedRoiRef | null; forcedVif: ForcedRoiRef | null }
  ): Promise<InputFunctionForces> {
    return api<InputFunctionForces>(`/subjects/${encodeURIComponent(subjectId)}/input-function-forces`, {
      method: 'PUT',
      body: JSON.stringify(forces),
    });
  }

  async saveSubjectRoiVoxels(
    subjectId: string,
    payload: { roiType: string; roiSubType: string; sliceIndex: number; frameIndex: number; voxels: Array<[number, number]> }
  ): Promise<{ ok: boolean; savedVoxelCount?: number }> {
    return api<{ ok: boolean; savedVoxelCount?: number }>(`/subjects/${encodeURIComponent(subjectId)}/roi-voxels`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
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

  async getProjectAnalysisDataset(projectId: string, view: ProjectAnalysisView): Promise<ProjectAnalysisDataset> {
    const v = encodeURIComponent(view || 'total');
    return api<ProjectAnalysisDataset>(`/projects/${encodeURIComponent(projectId)}/analysis/dataset?view=${v}`);
  }

  async analysisPearson(x: number[], y: number[]): Promise<ProjectAnalysisPearsonResponse> {
    return api<ProjectAnalysisPearsonResponse>('/analysis/stats/pearson', {
      method: 'POST',
      body: JSON.stringify({ x, y }),
    });
  }

  async analysisGroupCompare(a: number[], b: number[]): Promise<ProjectAnalysisGroupCompareResponse> {
    return api<ProjectAnalysisGroupCompareResponse>('/analysis/stats/group-compare', {
      method: 'POST',
      body: JSON.stringify({ a, b }),
    });
  }

  async analysisOls(y: number[], X: number[][], columns: string[]): Promise<ProjectAnalysisOlsResponse> {
    return api<ProjectAnalysisOlsResponse>('/analysis/stats/ols', {
      method: 'POST',
      body: JSON.stringify({ y, X, columns }),
    });
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

  async runSubjectStage(
    subjectId: string,
    stageId: StageId,
    opts?: { runDependencies?: boolean; envOverrides?: Record<string, string> }
  ): Promise<Job> {
    return api<Job>(`/subjects/${encodeURIComponent(subjectId)}/run-stage`, {
      method: 'POST',
      body: JSON.stringify({
        stageId,
        ...(opts?.runDependencies === false ? { runDependencies: false } : {}),
        ...(opts?.envOverrides && Object.keys(opts.envOverrides).length ? { envOverrides: opts.envOverrides } : {}),
      }),
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

  async clearSubjectData(subjectId: string): Promise<{
    ok: boolean;
    cancelledJobs: number;
    terminatedProcesses: number;
    deleted: { analysis: boolean; images: boolean; runner: boolean };
    subject: Subject;
  }> {
    return api<{
      ok: boolean;
      cancelledJobs: number;
      terminatedProcesses: number;
      deleted: { analysis: boolean; images: boolean; runner: boolean };
      subject: Subject;
    }>(
      `/subjects/${encodeURIComponent(subjectId)}/clear-data`,
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
