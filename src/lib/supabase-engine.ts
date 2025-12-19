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
  const sb = supabase as any;
  if (!sb.storage) return null;

  const { data, error } = await sb.storage.from(STORAGE_BUCKET).download(path);
  if (error || !data) return null;

  const text = await (data as Blob).text();
  return JSON.parse(text) as T;
}

async function toObjectUrl(path: string): Promise<string> {
  if (!supabase) return '';
  const sb = supabase as any;
  if (!sb.storage) return '';

  // Prefer signed URLs (works for private buckets while user is authed).
  const { data, error } = await sb.storage.from(STORAGE_BUCKET).createSignedUrl(path, 60 * 60);
  if (!error && data?.signedUrl) return data.signedUrl;

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

type ArtifactIndex = {
  paths?: string[];
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

    // Start/stop the local worker based on auth state.
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
    // Minimal: synthesize logs by polling the job row.
    // This avoids a blank "Waiting for logs…" experience without requiring new tables.
    if (!supabase) return () => {};
    const sb = supabase as any;

    let cancelled = false;
    let last = '';

    const tick = async () => {
      try {
        const { data, error } = await sb.from('jobs').select('*').eq('id', jobId).maybeSingle();
        if (cancelled) return;
        if (error || !data) return;

        const job = mapJobRow(data);
        const lines = [
          `status: ${job.status}`,
          `stage: ${job.stageId}`,
          `progress: ${Math.round((job.progress ?? 0) * 100)}%`,
          `step: ${job.currentStep || ''}`,
          job.startTime ? `start: ${job.startTime}` : '',
          job.endTime ? `end: ${job.endTime}` : '',
          job.error ? `error: ${job.error}` : '',
        ].filter(Boolean);
        const next = lines.join('\n');

        if (next !== last) {
          last = next;
          listener(next);
        }
      } catch {
        // ignore
      }
    };

    void tick();
    const t = window.setInterval(tick, 1000);
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
    // Create queued jobs for all stages; an external worker can pick them up.
    return safe(async () => {
      const sb = supabase! as any;
      const rows = subjectIds.flatMap(subjectId =>
        STAGES.map(stageId => ({
          project_id: projectId,
          subject_id: subjectId,
          stage_id: stageId,
          status: 'queued',
          progress: 0,
          current_step: 'Queued',
        }))
      );
      const { data, error } = await sb.from('jobs').insert(rows).select('*');
      if (error) throw error;
      return (data || []).map(mapJobRow);
    }, []);
  }

  async runSubjectPipeline(projectId: string, subjectId: string): Promise<void> {
    await this.runFullPipeline(projectId, [subjectId]);
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

      const payload = {
        project_id: existing.project_id,
        subject_id: existing.subject_id,
        stage_id: existing.stage_id ?? 'import',
        status: 'queued',
        progress: 0,
        current_step: 'Queued (retry)',
      };
      const { data: row, error: e2 } = await sb.from('jobs').insert(payload).select('*').single();
      if (e2) throw e2;
      return mapJobRow(row);
    }, null as any);

    if (!retried) throw new Error('SUPABASE_NOT_CONFIGURED');
    return retried;
  }

  async resolveDefaultVolume(_subjectId: string, _kind: 'dce' | 't1' | 'diffusion' = 'dce'): Promise<string> {
    return '';
  }

  async getVolumeInfo(_path: string, _subjectId?: string): Promise<VolumeInfo> {
    return {
      path: _path,
      dimensions: [0, 0, 0],
      voxelSize: [0, 0, 0],
      dataType: 'unknown',
      min: 0,
      max: 0,
    };
  }

  async getSliceData(_path: string, _z: number, _t: number = 0, _subjectId?: string): Promise<{ data: number[][]; min: number; max: number }> {
    return { data: [[0]], min: 0, max: 0 };
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

      const prefix = `projects/${subj.projectId}/subjects/${subj.id}/images/fit`;

      const pngPaths = await listPngObjectPathsWithIndexFallback(prefix, subj.projectId, subj.id);
      const withUrls = await Promise.all(
        pngPaths.map(async fullPath => {
          const file = basename(fullPath);
          return {
            id: file,
            name: file.replace(/\.png$/i, ''),
            unit: '',
            path: await toObjectUrl(fullPath),
            group: 'modelling' as const,
          };
        })
      );

      return withUrls.filter(m => !!m.path);
    }, []);
  }

  async getVolumes(_subjectId: string): Promise<VolumeFile[]> {
    return [];
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
    return { started: false, jobs: [], reason: 'No worker configured' };
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

  async getMetricsTable(_subjectId: string): Promise<MetricsTable> {
    return { rows: [] };
  }
}
