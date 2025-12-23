import type {
  Curve,
  DeconvolutionData,
  Job,
  JobStatus,
  MapVolume,
  MetricsTable,
  PatlakData,
  Project,
  StageId,
  StageStatus,
  Subject,
  ToftsData,
  VolumeFile,
  VolumeInfo,
} from '@/types';
import { DEFAULT_CONFIG } from '@/types';
import { supabase } from '@/lib/supabase';
import { cachedLoadNifti, sliceZ } from '@/lib/nifti-cache';

type Unsubscribe = () => void;

type StatusUpdate = { subjectId: string; stageId: StageId; status: StageStatus };

const STAGES: StageId[] = [
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

const _RAW_STORAGE_BUCKET = (import.meta as any).env?.VITE_SUPABASE_STORAGE_BUCKET as string | undefined;
const STORAGE_BUCKET: string = _RAW_STORAGE_BUCKET && _RAW_STORAGE_BUCKET.trim().length > 0 ? _RAW_STORAGE_BUCKET.trim() : 'pbrain';

const _missingObjectCache = new Map<string, number>();
// When artifacts are missing (common while jobs are still running), avoid repeatedly
// hammering Storage with signed-url attempts.
const _MISSING_TTL_MS = 5 * 60_000;

function _isRecentlyMissing(path: string): boolean {
  const ts = _missingObjectCache.get(path);
  return typeof ts === 'number' && Date.now() - ts < _MISSING_TTL_MS;
}

function _markMissing(path: string) {
  _missingObjectCache.set(path, Date.now());
}

// The browser cannot run the real Python pipeline; a worker must run elsewhere.
// Keep the old simulated in-browser worker opt-in only.
const _RAW_ENABLE_DEMO_WORKER = (import.meta as any).env?.VITE_ENABLE_DEMO_WORKER as string | undefined;
const ENABLE_DEMO_WORKER = (_RAW_ENABLE_DEMO_WORKER ?? '').trim().toLowerCase() === 'true' || (_RAW_ENABLE_DEMO_WORKER ?? '').trim() === '1';

function nowIso(): string {
  return new Date().toISOString();
}

function emptyStageStatuses(): Record<StageId, StageStatus> {
  return {
    import: 'not_run',
    t1_fit: 'not_run',
    input_functions: 'not_run',
    time_shift: 'not_run',
    segmentation: 'not_run',
    tissue_ctc: 'not_run',
    modelling: 'not_run',
    diffusion: 'not_run',
    montage_qc: 'not_run',
  };
}

function mapProjectRow(row: any): Project {
  return {
    id: row.id,
    name: row.name,
    storagePath: row.storage_path ?? '',
    createdAt: row.created_at ?? nowIso(),
    updatedAt: row.updated_at ?? row.created_at ?? nowIso(),
    copyDataIntoProject: !!row.copy_data_into_project,
    config: (row.config ?? DEFAULT_CONFIG) as any,
  };
}

function mapSubjectRow(row: any): Subject {
  return {
    id: row.id,
    projectId: row.project_id,
    name: row.name,
    sourcePath: row.source_path ?? '',
    createdAt: row.created_at ?? nowIso(),
    updatedAt: row.updated_at ?? row.created_at ?? nowIso(),
    hasT1: !!row.has_t1,
    hasDCE: !!row.has_dce,
    hasDiffusion: !!row.has_diffusion,
    stageStatuses: (row.stage_statuses ?? emptyStageStatuses()) as any,
  };
}

function mapJobRow(row: any): Job {
  let status = (row.status ?? 'queued') as string;
  // Back-compat with older values that leaked into the DB.
  if (status === 'done') status = 'completed';
  if (status === 'canceled') status = 'cancelled';

  return {
    id: row.id,
    projectId: row.project_id,
    subjectId: row.subject_id,
    stageId: (row.stage_id ?? 'import') as StageId,
    status: status as JobStatus,
    progress: Number(row.progress ?? 0),
    currentStep: row.current_step ?? '',
    startTime: row.start_time ?? undefined,
    endTime: row.end_time ?? undefined,
    estimatedTimeRemaining: row.estimated_time_remaining ?? undefined,
    error: row.error ?? undefined,
    logPath: row.log_path ?? undefined,
  };
}

async function safe<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try {
    if (!supabase) return fallback;
    return await fn();
  } catch (e) {
    // Keep the UI resilient; callers often poll without try/catch.
    console.warn('[SupabaseEngineAPI] op failed', e);
    return fallback;
  }
}

function isLikelyMacOsJunkFile(name: string): boolean {
  return name === '.DS_Store' || name.startsWith('._');
}

async function downloadJson<T>(path: string): Promise<T | null> {
  if (!supabase) return null;
  if (_isRecentlyMissing(path)) return null;
  const sb = supabase as any;
  if (!sb.storage) return null;

  const parseLoose = (text: string): T | null => {
    try {
      return JSON.parse(text) as T;
    } catch {
      // p-brain JSON can contain bare NaN/Infinity tokens.
      // Convert them to null so JSON.parse can proceed.
      const sanitized = String(text)
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/\b-Infinity\b/g, 'null');
      try {
        return JSON.parse(sanitized) as T;
      } catch {
        return null;
      }
    }
  };

  // Prefer SDK download (adds auth headers when a session exists).
  // Some Storage policies reject unauthenticated reads; in that case, fall back
  // to a signed URL fetch if we can create one.
  try {
    const { data, error } = await sb.storage.from(STORAGE_BUCKET).download(path);
    if (error) {
      // Supabase Storage returns 400 for missing objects in some configurations.
      // Avoid spamming the network with follow-up signed-url attempts.
      const status = Number((error as any)?.statusCode ?? (error as any)?.status ?? 0);
      if (status === 400 || status === 404) {
        _markMissing(path);
        return null;
      }
    }

    if (!error && data) {
      const text = await (data as Blob).text();
      return parseLoose(text);
    }
  } catch {
    // ignore and try signed URL fallback
  }

  const { data: signed, error: sErr } = await sb.storage.from(STORAGE_BUCKET).createSignedUrl(path, 60 * 60);
  if (sErr) {
    const status = Number((sErr as any)?.statusCode ?? (sErr as any)?.status ?? 0);
    if (status === 400 || status === 404) _markMissing(path);
  }
  if (sErr || !signed?.signedUrl) return null;

  const res = await fetch(signed.signedUrl);
  if (!res.ok) return null;
  const text = await res.text();
  return parseLoose(text);
}

function _asFiniteNumber(v: any): number | undefined {
  const n = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN;
  return Number.isFinite(n) ? n : undefined;
}

async function toObjectUrl(path: string): Promise<string> {
  if (!supabase) return '';
  if (_isRecentlyMissing(path)) return '';
  const sb = supabase as any;
  if (!sb.storage) return '';

  // Prefer signed URLs (works for private buckets while user is authed).
  const { data, error } = await sb.storage.from(STORAGE_BUCKET).createSignedUrl(path, 60 * 60);
  if (error) {
    const status = Number((error as any)?.statusCode ?? (error as any)?.status ?? 0);
    if (status === 400 || status === 404) {
      _markMissing(path);
      return '';
    }
  }
  if (!error && data?.signedUrl) return data.signedUrl;

  // Only works if the bucket is public.
  const pub = sb.storage.from(STORAGE_BUCKET).getPublicUrl(path);
  return pub?.data?.publicUrl ?? '';
}

async function listObjects(prefix: string): Promise<Array<{ name: string; fullPath: string }>> {
  if (!supabase) return [];
  const sb = supabase as any;
  if (!sb.storage) return [];

  const { data, error } = await sb.storage.from(STORAGE_BUCKET).list(prefix, {
    limit: 200,
    sortBy: { column: 'name', order: 'asc' },
  });
  if (error) return [];

  return (data || [])
    .filter((o: any) => o?.name && !isLikelyMacOsJunkFile(o.name))
    .map((o: any) => ({ name: o.name as string, fullPath: `${prefix}/${o.name}` }));
}

function basename(p: string): string {
  const parts = p.split('/').filter(Boolean);
  return parts[parts.length - 1] ?? p;
}

function _stripQuery(p: string): string {
  return (p || '').split('?')[0] || '';
}

function _lowerBase(p: string): string {
  return basename(_stripQuery(p)).toLowerCase();
}

function _looksAxialVariant(nameLower: string): boolean {
  return nameLower.startsWith('ax') || nameLower.startsWith('ax_') || nameLower.startsWith('ax-');
}

function inferVolumeKindFromFilename(filenameOrPath: string): string {
  const n = _lowerBase(filenameOrPath);
  // p-brain utils/parameters.py conventions
  if (n.includes('wipcs_t1w_3d_tfe_32channel')) return 't1';
  if (n.includes('wipcs_3d_brain_view_t2_32chshc') || n.includes('wipaxt2tsematrix')) return 't2';
  if (n.includes('wipcs_3d_brain_view_flair_shc')) return 'flair';
  if (n.includes('wipdelrec-hperf120long') || n.includes('wiphperf120long')) return 'dce';
  if (n.includes('reg-dwinysense') || n.includes('isodwib-1000') || n.includes('wipdti_rsi') || n.includes('wipdwi_rsi')) return 'diffusion';
  return '';
}

const _DIFFUSION_PRIORITY_BASENAMES = [
  'reg-dwinysense.nii',
  'reg-dwinysense.nii.gz',
  'reg-dwinysense_adc.nii',
  'reg-dwinysense_adc.nii.gz',
  'isodwib-1000.nii',
  'isodwib-1000.nii.gz',
  'wipdti_rsi_p.nii',
  'wipdti_rsi_p.nii.gz',
  'wipdti_rsi_a.nii',
  'wipdti_rsi_a.nii.gz',
  'wipdwi_rsi_p.nii',
  'wipdwi_rsi_p.nii.gz',
];

type ArtifactIndex = {
  paths?: string[];
  volumes?: Array<{ id: string; name: string; path: string; kind?: string }>;
  maps?: Array<{ id: string; name: string; path: string }>;
};

async function listPngObjectPathsWithIndexFallback(prefix: string, projectId: string, subjectId: string): Promise<string[]> {
  // First try Storage list() (fast path).
  const objs = await listObjects(prefix);
  const listed = objs.filter(o => /\.png$/i.test(o.name)).map(o => o.fullPath);
  if (listed.length > 0) return listed;

  // Fallback: read the uploaded artifacts index (doesn't require listing folders).
  const indexPath = `projects/${projectId}/subjects/${subjectId}/artifacts/index.json`;
  const index = await downloadJson<ArtifactIndex>(indexPath);
  const paths = (index?.paths ?? []).filter(p => typeof p === 'string');
  const withPrefix = `${prefix}/`;
  return paths.filter(p => p.startsWith(withPrefix) && /\.png$/i.test(p));
}

export class SupabaseEngineAPI {
  private jobListeners = new Set<(job: Job) => void>();
  private statusListeners = new Set<(update: StatusUpdate) => void>();
  private jobsChannel: any | undefined;
  private subjectsChannel: any | undefined;

  private workerAbort = false;
  private workerPromise: Promise<void> | undefined;
  private authUnsub: any | undefined;

  constructor() {
    if (typeof window === 'undefined' || !supabase) return;
    const sb = supabase as any;

    if (ENABLE_DEMO_WORKER) {
      // Start/stop the simulated local worker based on auth state.
      sb.auth
        .getSession()
        .then(({ data }: any) => {
          if (data?.session?.user) this.startLocalWorker();
        })
        .catch(() => {});

      const { data } = sb.auth.onAuthStateChange((_event: any, session: any) => {
        if (session?.user) this.startLocalWorker();
        else this.stopLocalWorker();
      });
      this.authUnsub = data?.subscription;
    } else {
      // No in-browser worker. Jobs must be processed by an external worker.
      this.stopLocalWorker();
    }
  }

  private broadcastJob(job: Job) {
    this.jobListeners.forEach(l => l(job));
  }

  private broadcastStatus(update: StatusUpdate) {
    this.statusListeners.forEach(l => l(update));
  }

  onJobUpdate(listener: (job: Job) => void): Unsubscribe {
    this.jobListeners.add(listener);
    this.ensureJobsRealtime();
    return () => {
      this.jobListeners.delete(listener);
      if (this.jobListeners.size === 0) {
        this.jobsChannel?.unsubscribe?.();
        this.jobsChannel = undefined;
      }
    };
  }

  onStatusUpdate(listener: (update: StatusUpdate) => void): Unsubscribe {
    this.statusListeners.add(listener);
    this.ensureSubjectsRealtime();
    return () => {
      this.statusListeners.delete(listener);
      if (this.statusListeners.size === 0) {
        this.subjectsChannel?.unsubscribe?.();
        this.subjectsChannel = undefined;
      }
    };
  }

  private ensureJobsRealtime() {
    if (!supabase || this.jobsChannel) return;
    const sb = supabase as any;
    if (!sb.channel) return;

    this.jobsChannel = sb
      .channel('jobs-changes')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'jobs' },
        (payload: any) => {
          const row = payload?.new ?? payload?.old;
          if (!row) return;
          this.broadcastJob(mapJobRow(row));
        }
      )
      .subscribe();
  }

  private ensureSubjectsRealtime() {
    if (!supabase || this.subjectsChannel) return;
    const sb = supabase as any;
    if (!sb.channel) return;

    this.subjectsChannel = sb
      .channel('subjects-changes')
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'subjects' },
        (payload: any) => {
          const next = payload?.new;
          const prev = payload?.old;
          if (!next?.id) return;
          const nextStages = (next.stage_statuses ?? {}) as Record<string, StageStatus>;
          const prevStages = (prev?.stage_statuses ?? {}) as Record<string, StageStatus>;

          const stageIds = new Set<string>([...Object.keys(nextStages), ...Object.keys(prevStages)]);
          stageIds.forEach(stageId => {
            const a = prevStages[stageId];
            const b = nextStages[stageId];
            if (a !== b && b) {
              this.broadcastStatus({ subjectId: next.id, stageId: stageId as StageId, status: b });
            }
          });
        }
      )
      .subscribe();
  }

  onJobLogs(jobId: string, listener: (log: string) => void): Unsubscribe {
    // Poll job_events for log lines; emits only new lines per listener.
    if (!supabase) return () => {};
    const sb = supabase as any;

    let cancelled = false;
    let lastCount = 0;

    const tick = async () => {
      try {
        const { data, error } = await sb
          .from('job_events')
          .select('ts, level, message')
          .eq('job_id', jobId)
          .order('ts', { ascending: true })
          .limit(200);
        if (cancelled) return;
        if (error || !data) return;

        if (data.length > lastCount) {
          const fresh = data.slice(lastCount);
          fresh.forEach((row: any) => {
            const ts = row.ts ? new Date(row.ts).toISOString() : '';
            listener(`${ts} [${row.level ?? 'info'}] ${row.message ?? ''}`);
          });
          lastCount = data.length;
        }
      } catch {
        /* ignore */
      }
    };

    void tick();
    const t = window.setInterval(tick, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }

  private startLocalWorker() {
    if (this.workerPromise || !supabase) return;
    this.workerAbort = false;
    this.workerPromise = this.localWorkerLoop().finally(() => {
      this.workerPromise = undefined;
    });
  }

  private stopLocalWorker() {
    this.workerAbort = true;
    this.workerPromise = undefined;
  }

  private async sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private async localWorkerLoop() {
    while (!this.workerAbort) {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        await this.sleep(3000);
        continue;
      }

      try {
        const didWork = await this.processNextQueuedJob();
        await this.sleep(didWork ? 250 : 1500);
      } catch {
        await this.sleep(2000);
      }
    }
  }

  private async processNextQueuedJob(): Promise<boolean> {
    if (!supabase) return false;
    const sb = supabase as any;

    // Pick one queued job (RLS scopes to current user).
    const { data: queued, error: qErr } = await sb
      .from('jobs')
      .select('*')
      .eq('status', 'queued')
      .order('created_at', { ascending: true })
      .limit(1);
    if (qErr) throw qErr;
    const jobRow = (queued || [])[0];
    if (!jobRow) return false;

    // Claim it atomically.
    const { data: claimed, error: cErr } = await sb
      .from('jobs')
      .update({
        status: 'running',
        progress: 0,
        current_step: 'Starting',
        start_time: nowIso(),
        end_time: null,
        error: null,
      })
      .eq('id', jobRow.id)
      .eq('status', 'queued')
      .select('*')
      .maybeSingle();
    if (cErr) throw cErr;
    if (!claimed) return true; // Someone else claimed it.

    const job = mapJobRow(claimed);
    this.broadcastJob(job);

    // Mark subject stage as running.
    await this.updateSubjectStage(job.subjectId, job.stageId, 'running');

    const steps = [
      { p: 0.15, label: 'Preparing data', eta: 12 },
      { p: 0.45, label: 'Running pipeline', eta: 8 },
      { p: 0.75, label: 'Generating outputs', eta: 4 },
      { p: 0.95, label: 'Finalizing', eta: 1 },
    ];

    for (const s of steps) {
      if (this.workerAbort) return true;

      const { data: updated, error: uErr } = await sb
        .from('jobs')
        .update({
          progress: s.p,
          current_step: s.label,
          estimated_time_remaining: s.eta,
        })
        .eq('id', job.id)
        .eq('status', 'running')
        .select('*')
        .maybeSingle();
      if (uErr) throw uErr;
      if (!updated) {
        // cancelled/failed elsewhere
        return true;
      }
      this.broadcastJob(mapJobRow(updated));
      await this.sleep(900);
    }

    const { data: completed, error: fErr } = await sb
      .from('jobs')
      .update({
        status: 'completed',
        progress: 1,
        current_step: 'Completed',
        estimated_time_remaining: 0,
        end_time: nowIso(),
      })
      .eq('id', job.id)
      .eq('status', 'running')
      .select('*')
      .maybeSingle();
    if (fErr) throw fErr;
    if (completed) {
      this.broadcastJob(mapJobRow(completed));
      await this.updateSubjectStage(job.subjectId, job.stageId, 'done');
    }

    return true;
  }

  private async updateSubjectStage(subjectId: string, stageId: StageId, status: StageStatus) {
    if (!supabase) return;
    const sb = supabase as any;

    // Fetch current stage_statuses, then update JSON.
    const { data: subj, error: sErr } = await sb
      .from('subjects')
      .select('id, stage_statuses')
      .eq('id', subjectId)
      .maybeSingle();
    if (sErr) throw sErr;
    if (!subj) return;

    const next = { ...(subj.stage_statuses ?? emptyStageStatuses()), [stageId]: status };
    const { error: uErr } = await sb.from('subjects').update({ stage_statuses: next }).eq('id', subjectId);
    if (uErr) throw uErr;

    this.broadcastStatus({ subjectId, stageId, status });
  }

  async getProjects(): Promise<Project[]> {
    return safe(async () => {
      const sb = supabase! as any;
      const { data, error } = await sb
        .from('projects')
        .select('*')
        .order('created_at', { ascending: false });
      if (error) throw error;
      return (data || []).map(mapProjectRow);
    }, []);
  }

  async getProject(id: string): Promise<Project | undefined> {
    return safe(async () => {
      const sb = supabase! as any;
      const { data, error } = await sb.from('projects').select('*').eq('id', id).maybeSingle();
      if (error) throw error;
      return data ? mapProjectRow(data) : undefined;
    }, undefined);
  }

  async createProject(data: { name: string; storagePath: string; copyDataIntoProject: boolean }): Promise<Project> {
    const created = await safe(async () => {
      const sb = supabase! as any;
      const payload = {
        name: data.name,
        storage_path: data.storagePath,
        copy_data_into_project: !!data.copyDataIntoProject,
        config: DEFAULT_CONFIG,
      };
      const { data: row, error } = await sb.from('projects').insert(payload).select('*').single();
      if (error) throw error;
      return mapProjectRow(row);
    }, null as any);

    // If Supabase isn't configured, mimic the old behavior by throwing a typed-ish error.
    if (!created) throw new Error('SUPABASE_NOT_CONFIGURED');
    return created;
  }

  async updateProject(
    projectId: string,
    data: { name?: string; storagePath?: string; copyDataIntoProject?: boolean }
  ): Promise<Project> {
    const updated = await safe(async () => {
      const sb = supabase! as any;
      const payload: any = {};
      if (typeof data.name === 'string') payload.name = data.name;
      if (typeof data.storagePath === 'string') payload.storage_path = data.storagePath;
      if (typeof data.copyDataIntoProject === 'boolean') payload.copy_data_into_project = data.copyDataIntoProject;

      const { data: row, error } = await sb.from('projects').update(payload).eq('id', projectId).select('*').single();
      if (error) throw error;
      return mapProjectRow(row);
    }, null as any);

    if (!updated) throw new Error('SUPABASE_NOT_CONFIGURED');
    return updated;
  }

  async deleteProject(projectId: string): Promise<void> {
    await safe(async () => {
      const sb = supabase! as any;
      const { error } = await sb.from('projects').delete().eq('id', projectId);
      if (error) throw error;
      return undefined;
    }, undefined);
  }

  async updateProjectConfig(projectId: string, configUpdate: any): Promise<Project | undefined> {
    return safe(async () => {
      const sb = supabase! as any;
      const current = await this.getProject(projectId);
      const nextConfig = { ...(current?.config || DEFAULT_CONFIG), ...(configUpdate || {}) };
      const { data: row, error } = await sb
        .from('projects')
        .update({ config: nextConfig })
        .eq('id', projectId)
        .select('*')
        .maybeSingle();
      if (error) throw error;
      return row ? mapProjectRow(row) : undefined;
    }, undefined);
  }

  async getSubjects(projectId: string): Promise<Subject[]> {
    return safe(async () => {
      const sb = supabase! as any;
      const { data, error } = await sb
        .from('subjects')
        .select('*')
        .eq('project_id', projectId)
        .order('created_at', { ascending: true });
      if (error) throw error;
      return (data || []).map(mapSubjectRow);
    }, []);
  }

  async getSubject(id: string): Promise<Subject | undefined> {
    return safe(async () => {
      const sb = supabase! as any;
      const { data, error } = await sb.from('subjects').select('*').eq('id', id).maybeSingle();
      if (error) throw error;
      return data ? mapSubjectRow(data) : undefined;
    }, undefined);
  }

  async importSubjects(projectId: string, subjects: Array<{ name: string; sourcePath: string }>): Promise<Subject[]> {
    return safe(async () => {
      const sb = supabase! as any;
      const rows = subjects.map(s => ({
        project_id: projectId,
        name: s.name,
        source_path: s.sourcePath,
        has_t1: false,
        has_dce: false,
        has_diffusion: false,
        stage_statuses: emptyStageStatuses(),
      }));
      const { data, error } = await sb.from('subjects').insert(rows).select('*');
      if (error) throw error;
      return (data || []).map(mapSubjectRow);
    }, []);
  }

  async scanProjectSubjects(_projectId: string): Promise<{ subjects: Array<{ name: string; sourcePath: string }> }> {
    // No server-side filesystem scan in Supabase-only mode.
    return { subjects: [] };
  }

  async runFullPipeline(projectId: string, subjectIds: string[]): Promise<Job[]> {
    // Queue one job per subject with a payload the local runner can resolve.
    return safe(async () => {
      const sb = supabase! as any;
      const { data: subjects, error: sErr } = await sb
        .from('subjects')
        .select('id, source_path')
        .in('id', subjectIds);
      if (sErr) throw sErr;
      const rows = (subjects || []).map((s: any) => ({
        project_id: projectId,
        subject_id: s.id,
        stage_id: 'import',
        status: 'queued',
        progress: 0,
        current_step: 'Queued (waiting for runner)',
        payload: {
          relative_path: s.source_path ?? '',
          subject_id: s.id,
        },
      }));
      if (rows.length === 0) return [];

      // Persist a "running" indicator immediately so the dashboard dots reflect queued work.
      // (External runners may take a moment to claim jobs.)
      await Promise.all(
        subjectIds.map(async sid => {
          try {
            await this.updateSubjectStage(sid, 'import', 'running');
          } catch {
            /* ignore */
          }
        })
      );
      const { data, error } = await sb.from('jobs').insert(rows).select('*');
      if (error) throw error;
      return (data || []).map(mapJobRow);
    }, []);
  }

  async runSubjectPipeline(projectId: string, subjectId: string): Promise<void> {
    await this.runFullPipeline(projectId, [subjectId]);
  }

  async getJobEvents(jobId: string): Promise<Array<{ timestamp: Date; level: 'info' | 'warning' | 'error'; message: string }>> {
    return safe(async () => {
      const sb = supabase! as any;
      const { data, error } = await sb
        .from('job_events')
        .select('ts, level, message')
        .eq('job_id', jobId)
        .order('ts', { ascending: true })
        .limit(200);
      if (error || !data) return [];
      return data.map((row: any) => ({
        timestamp: row.ts ? new Date(row.ts) : new Date(),
        level: (row.level ?? 'info') as any,
        message: row.message ?? '',
      }));
    }, []);
  }

  async getJobs(filters: { projectId?: string; subjectId?: string; status?: string }): Promise<Job[]> {
    return safe(async () => {
      const sb = supabase! as any;
      let q = sb.from('jobs').select('*').order('created_at', { ascending: false });
      if (filters.projectId) q = q.eq('project_id', filters.projectId);
      if (filters.subjectId) q = q.eq('subject_id', filters.subjectId);
      if (filters.status) q = q.eq('status', filters.status);
      const { data, error } = await q;
      if (error) throw error;
      return (data || []).map(mapJobRow);
    }, []);
  }

  async getRunnerHeartbeats(): Promise<Array<{ workerId: string; lastSeen: string; hostname?: string; meta?: any }>> {
    return safe(async () => {
      const sb = supabase! as any;
      const { data, error } = await sb
        .from('worker_heartbeats')
        .select('worker_id,last_seen,hostname,meta')
        .order('last_seen', { ascending: false })
        .limit(10);
      if (error || !data) return [];
      return (data || []).map((r: any) => ({
        workerId: String(r.worker_id ?? ''),
        lastSeen: String(r.last_seen ?? ''),
        hostname: r.hostname ?? undefined,
        meta: r.meta ?? undefined,
      }));
    }, []);
  }

  async cancelJob(jobId: string): Promise<void> {
    await safe(async () => {
      const sb = supabase! as any;
      const { error } = await sb.from('jobs').update({ status: 'cancelled', end_time: nowIso() }).eq('id', jobId);
      if (error) throw error;
      return undefined;
    }, undefined);
  }

  async retryJob(jobId: string): Promise<Job> {
    const retried = await safe(async () => {
      const sb = supabase! as any;
      const { data: existing, error: e1 } = await sb.from('jobs').select('*').eq('id', jobId).maybeSingle();
      if (e1) throw e1;
      if (!existing) throw new Error('JOB_NOT_FOUND');

		let nextPayload: any = existing.payload && typeof existing.payload === 'object' ? { ...existing.payload } : {};
		if (!nextPayload.subject_id && existing.subject_id) nextPayload.subject_id = existing.subject_id;
		if (!nextPayload.relative_path && existing.subject_id) {
			const { data: subj, error: sErr } = await sb
				.from('subjects')
				.select('source_path')
				.eq('id', existing.subject_id)
				.maybeSingle();
			if (sErr) throw sErr;
			if (subj?.source_path) nextPayload.relative_path = subj.source_path;
		}

      const payload = {
        project_id: existing.project_id,
        subject_id: existing.subject_id,
        stage_id: existing.stage_id ?? 'import',
        status: 'queued',
        progress: 0,
        current_step: 'Queued (retry)',
			payload: nextPayload,
      };
      const { data: row, error: e2 } = await sb.from('jobs').insert(payload).select('*').single();
      if (e2) throw e2;
      return mapJobRow(row);
    }, null as any);

    if (!retried) throw new Error('SUPABASE_NOT_CONFIGURED');
    return retried;
  }

  async resolveDefaultVolume(
    _subjectId: string,
    _kind: 'dce' | 't1' | 't2' | 'flair' | 'diffusion' | 'map' = 'dce'
  ): Promise<string> {
    const kindLower0 = String(_kind).toLowerCase();
    if (kindLower0 === 'map') {
      const maps = await this.getMapVolumes(_subjectId);
      return maps[0]?.path ?? '';
    }

    const vols = await this.getVolumes(_subjectId);
    const kindLower = kindLower0;

    // 1) Prefer explicit kind if present.
    const explicit = vols.find(v => String(v.kind || '').toLowerCase() === kindLower)?.path;
    if (explicit) return explicit;

    // 2) Try filename inference.
    const inferredHit = vols.find(v => inferVolumeKindFromFilename(v.name || v.path) === kindLower)?.path;
    if (inferredHit) return inferredHit;

    // 3) Diffusion: honor ordered candidates (utils/parameters.py)
    if (kindLower === 'diffusion') {
      const byBase = new Map(vols.map(v => [_lowerBase(v.name || v.path), v.path] as const));
      for (const b of _DIFFUSION_PRIORITY_BASENAMES) {
        const hit = byBase.get(b);
        if (hit) return hit;
      }
    }

    // 4) DCE: prefer DelRec fallback when present.
    if (kindLower === 'dce') {
      const fallback = vols.find(v => _lowerBase(v.name || v.path).includes('wipdelrec-hperf120long'))?.path;
      if (fallback) return fallback;
      const primary = vols.find(v => _lowerBase(v.name || v.path).includes('wiphperf120long'))?.path;
      if (primary) return primary;
    }

    // 5) T2/FLAIR: prefer non-axial variant when both exist.
    if (kindLower === 't2' || kindLower === 'flair' || kindLower === 't1') {
      const key = kindLower === 't1'
        ? 'wipcs_t1w_3d_tfe_32channel'
        : kindLower === 't2'
          ? 'wipcs_3d_brain_view_t2_32chshc'
          : 'wipcs_3d_brain_view_flair_shc';
      const matches = vols.filter(v => _lowerBase(v.name || v.path).includes(key));
      const nonAx = matches.find(v => !_looksAxialVariant(_lowerBase(v.name || v.path)))?.path;
      if (nonAx) return nonAx;
      const ax = matches[0]?.path;
      if (ax) return ax;
    }

    return vols[0]?.path ?? '';
  }

  async getVolumeInfo(_path: string, _subjectId?: string): Promise<VolumeInfo> {
    const fallback: VolumeInfo = {
      path: _path,
      dimensions: [0, 0, 0, 0],
      voxelSize: [0, 0, 0],
      dataType: 'unknown',
      min: 0,
      max: 0,
    };

    return safe(async () => {
      const url = await toObjectUrl(_path);
      if (!url) return fallback;
      const vol = await cachedLoadNifti(url);
      return {
        path: _path,
        dimensions: vol.dims,
        voxelSize: vol.pixDims,
        dataType: 'float32',
        min: vol.min,
        max: vol.max,
      };
    }, fallback);
  }

  async getSliceData(_path: string, _z: number, _t: number = 0, _subjectId?: string): Promise<{ data: number[][]; min: number; max: number }> {
    const fallback = { data: [[0]], min: 0, max: 0 };
    return safe(async () => {
      const url = await toObjectUrl(_path);
      if (!url) return fallback;
      const vol = await cachedLoadNifti(url);
      const data = sliceZ(vol, _z, _t);
      return { data, min: vol.min, max: vol.max };
    }, fallback);
  }

  async getCurves(_subjectId: string): Promise<Curve[]> {
    return safe(async () => {
      const subj = await this.getSubject(_subjectId);
      if (!subj) return [];

      const prefix = `projects/${subj.projectId}/subjects/${subj.id}/curves`;
      const curves = await downloadJson<{ curves: Curve[] }>(`${prefix}/curves.json`);
      return curves?.curves ?? [];
    }, []);
  }

  async getMapVolumes(_subjectId: string): Promise<MapVolume[]> {
    return safe(async () => {
      const subj = await this.getSubject(_subjectId);
      if (!subj) return [];

      // Prefer the artifacts index produced by scripts/sync-artifacts.mjs.
      const indexPath = `projects/${subj.projectId}/subjects/${subj.id}/artifacts/index.json`;
      const index = await downloadJson<ArtifactIndex>(indexPath);

      const indexMaps = (index?.maps ?? [])
        .map(m => ({
          id: basename(m.path),
          name: basename(m.path).replace(/\.(nii|nii\.gz)$/i, ''),
          unit: '',
          path: m.path,
          group: /\/analysis\/diffusion\//i.test(m.path) || /^fa_|^md_|^ad_|^rd_|^mo_|tensor_residual_/i.test(basename(m.path))
            ? ('diffusion' as const)
            : ('modelling' as const),
        }))
        .filter(m => /\.(nii|nii\.gz)$/i.test(m.path));

      if (indexMaps.length > 0) return indexMaps;

      // Fallback: list any NIfTI outputs under analysis/.
      const prefix = `projects/${subj.projectId}/subjects/${subj.id}/analysis`;
      const objs = await listObjects(prefix);
      const nifti = objs.filter(o => /\.(nii|nii\.gz)$/i.test(o.name));
      return nifti.map(o => ({
        id: o.name,
        name: o.name.replace(/\.(nii|nii\.gz)$/i, ''),
        unit: '',
        path: o.fullPath,
        group: /\/diffusion\//i.test(o.fullPath) || /^fa_|^md_|^ad_|^rd_|^mo_|tensor_residual_/i.test(o.name)
          ? ('diffusion' as const)
          : ('modelling' as const),
      }));
    }, []);
  }

  async getVolumes(_subjectId: string): Promise<VolumeFile[]> {
    return safe(async () => {
      const subj = await this.getSubject(_subjectId);
      if (!subj) return [];

      const indexPath = `projects/${subj.projectId}/subjects/${subj.id}/artifacts/index.json`;
      const index = await downloadJson<ArtifactIndex>(indexPath);
      const vols = (index?.volumes ?? [])
        .filter(v => typeof v?.path === 'string' && /\.(nii|nii\.gz)$/i.test(v.path))
        .map(v => ({
          id: v.id || basename(v.path),
          name: v.name || basename(v.path),
          path: v.path,
				kind: v.kind || inferVolumeKindFromFilename(v.name || v.path) || 'source',
        }));
      if (vols.length > 0) return vols;

      const prefix = `projects/${subj.projectId}/subjects/${subj.id}/volumes/source`;
      const objs = await listObjects(prefix);
      const source = objs
        .filter(o => /\.(nii|nii\.gz)$/i.test(o.name))
        .map(o => ({ id: o.name, name: o.name, path: o.fullPath, kind: (inferVolumeKindFromFilename(o.name) || 'source') as any }));
      if (source.length > 0) return source;

      // If source volumes are too large to upload (Supabase Storage size limits),
      // fall back to analysis maps so the Viewer can still display something.
      const indexMaps = (index?.maps ?? []).filter(m => typeof m?.path === 'string' && /\.(nii|nii\.gz)$/i.test(m.path));
      if (indexMaps.length > 0) {
        return indexMaps.map(m => ({
          id: basename(m.path),
          name: basename(m.path),
          path: m.path,
          kind: 'analysis',
        }));
      }

      const analysisPrefix = `projects/${subj.projectId}/subjects/${subj.id}/analysis`;
      const analysisObjs = await listObjects(analysisPrefix);
      return analysisObjs
        .filter(o => /\.(nii|nii\.gz)$/i.test(o.name))
        .map(o => ({ id: o.name, name: o.name, path: o.fullPath, kind: 'analysis' }));
    }, []);
  }

  async getMontageImages(_subjectId: string): Promise<Array<{ id: string; name: string; path: string }>> {
    return safe(async () => {
      const subj = await this.getSubject(_subjectId);
      if (!subj) return [];

      const prefix = `projects/${subj.projectId}/subjects/${subj.id}/images/ai/montages`;

      const pngPaths = await listPngObjectPathsWithIndexFallback(prefix, subj.projectId, subj.id);
      const withUrls = await Promise.all(
        pngPaths.map(async fullPath => {
          const file = basename(fullPath);
          return {
            id: file,
            name: file,
            path: await toObjectUrl(fullPath),
          };
        })
      );

      return withUrls.filter(m => !!m.path);
    }, []);
  }

  async ensureSubjectArtifacts(_subjectId: string, _kind: 'all' | 'maps' | 'curves' | 'montages' = 'all'): Promise<{ started: boolean; jobs: any[]; reason: string }> {
    return { started: false, jobs: [], reason: 'External worker required (the browser cannot run p-brain)' };
  }

  async getPatlakData(_subjectId: string, _region: string): Promise<PatlakData> {
    return { x: [], y: [], Ki: 0, vp: 0, r2: 0, fitLineX: [], fitLineY: [], windowStart: 0 };
  }

  async getToftsData(_subjectId: string, _region: string): Promise<ToftsData> {
    return { timePoints: [], measured: [], fitted: [], Ktrans: 0, ve: 0, vp: 0, residuals: [] };
  }

  async getDeconvolutionData(_subjectId: string, _region: string): Promise<DeconvolutionData> {
    return { timePoints: [], residue: [], h_t: [], CBF: 0, MTT: 0, CTH: 0 };
  }

  async getMetricsTable(
    _subjectId: string,
    view: 'atlas' | 'tissue' = 'atlas'
  ): Promise<MetricsTable> {
    return safe(async () => {
      const subj = await this.getSubject(_subjectId);
      if (!subj) return { rows: [] };

      const prefix = `projects/${subj.projectId}/subjects/${subj.id}/analysis/metrics`;

      if (view === 'atlas') {
        type AtlasEntry = {
          Ki?: unknown;
          vp?: unknown;
          CBF_tikhonov?: unknown;
          MTT_tikhonov?: unknown;
          CTH_tikhonov?: unknown;
        };

        const atlas =
          (await downloadJson<Record<string, AtlasEntry>>(`${prefix}/Ki_values_atlas_patlak.json`)) ||
          (await downloadJson<Record<string, AtlasEntry>>(`${prefix}/Ki_values_atlas_tikhonov.json`));
        if (!atlas) return { rows: [] };

        const rows = Object.entries(atlas)
          .map(([region, v]) => ({
            region,
            Ki: _asFiniteNumber((v as any)?.Ki),
            vp: _asFiniteNumber((v as any)?.vp),
            CBF: _asFiniteNumber((v as any)?.CBF_tikhonov),
            MTT: _asFiniteNumber((v as any)?.MTT_tikhonov),
            CTH: _asFiniteNumber((v as any)?.CTH_tikhonov),
          }))
          .sort((a, b) => (a.region || '').localeCompare(b.region || ''));

        return { rows };
      }

      type TissueSliceEntry = {
        Ki?: unknown;
        vp?: unknown;
        CBF_tikhonov?: unknown;
        MTT_tikhonov?: unknown;
        CTH_tikhonov?: unknown;
        voxel_count?: unknown;
      };
      type TissueSlice = {
        white_matter_median?: TissueSliceEntry;
        cortical_gray_matter_median?: TissueSliceEntry;
        subcortical_gray_matter_median?: TissueSliceEntry;
      };

      const tissueSlices =
        (await downloadJson<TissueSlice[]>(`${prefix}/AI_values_median_tikhonov.json`)) ||
        (await downloadJson<TissueSlice[]>(`${prefix}/AI_values_median_patlak.json`));
      if (!tissueSlices || !Array.isArray(tissueSlices)) return { rows: [] };

      const defs: Array<{ key: keyof TissueSlice; region: string }> = [
        { key: 'white_matter_median', region: 'White Matter' },
        { key: 'cortical_gray_matter_median', region: 'Cortical Gray Matter' },
        { key: 'subcortical_gray_matter_median', region: 'Subcortical Gray Matter' },
      ];

      const rows = defs
        .map(({ key, region }) => {
          const acc: Record<string, { sum: number; w: number }> = {
            Ki: { sum: 0, w: 0 },
            vp: { sum: 0, w: 0 },
            CBF: { sum: 0, w: 0 },
            MTT: { sum: 0, w: 0 },
            CTH: { sum: 0, w: 0 },
          };

          for (const slice of tissueSlices) {
            const entry = (slice as any)?.[key] as TissueSliceEntry | undefined;
            if (!entry) continue;
            const w = _asFiniteNumber((entry as any).voxel_count) ?? 0;
            if (!(w > 0)) continue;

            const Ki = _asFiniteNumber((entry as any).Ki);
            const vp = _asFiniteNumber((entry as any).vp);
            const CBF = _asFiniteNumber((entry as any).CBF_tikhonov);
            const MTT = _asFiniteNumber((entry as any).MTT_tikhonov);
            const CTH = _asFiniteNumber((entry as any).CTH_tikhonov);

            if (Ki !== undefined) {
              acc.Ki.sum += Ki * w;
              acc.Ki.w += w;
            }
            if (vp !== undefined) {
              acc.vp.sum += vp * w;
              acc.vp.w += w;
            }
            if (CBF !== undefined) {
              acc.CBF.sum += CBF * w;
              acc.CBF.w += w;
            }
            if (MTT !== undefined) {
              acc.MTT.sum += MTT * w;
              acc.MTT.w += w;
            }
            if (CTH !== undefined) {
              acc.CTH.sum += CTH * w;
              acc.CTH.w += w;
            }
          }

          const row = {
            region,
            Ki: acc.Ki.w ? acc.Ki.sum / acc.Ki.w : undefined,
            vp: acc.vp.w ? acc.vp.sum / acc.vp.w : undefined,
            CBF: acc.CBF.w ? acc.CBF.sum / acc.CBF.w : undefined,
            MTT: acc.MTT.w ? acc.MTT.sum / acc.MTT.w : undefined,
            CTH: acc.CTH.w ? acc.CTH.sum / acc.CTH.w : undefined,
          };

          const hasAny = Object.values(row).some(v => typeof v === 'number' && Number.isFinite(v));
          return hasAny ? row : null;
        })
        .filter((r): r is NonNullable<typeof r> => !!r);

      return { rows };
    }, { rows: [] });
  }
}
