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
  return {
    id: row.id,
    projectId: row.project_id,
    subjectId: row.subject_id,
    stageId: (row.stage_id ?? 'import') as StageId,
    status: (row.status ?? 'queued') as JobStatus,
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

export class SupabaseEngineAPI {
  onJobUpdate(_listener: (job: Job) => void): Unsubscribe {
    // Optional: implement realtime later. For now, polling pages will refresh jobs.
    return () => {};
  }

  onStatusUpdate(_listener: (update: { subjectId: string; stageId: any; status: any }) => void): Unsubscribe {
    // Optional: implement realtime later.
    return () => {};
  }

  onJobLogs(_jobId: string, _listener: (log: string) => void): Unsubscribe {
    // No server-side log streaming in Supabase-only mode.
    return () => {};
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
    // Create queued jobs; a separate worker can pick them up.
    return safe(async () => {
      const sb = supabase! as any;
      const rows = subjectIds.map(subjectId => ({
        project_id: projectId,
        subject_id: subjectId,
        stage_id: 'import',
        status: 'queued',
        progress: 0,
        current_step: 'Queued',
      }));
      const { data, error } = await sb.from('jobs').insert(rows).select('*');
      if (error) throw error;
      return (data || []).map(mapJobRow);
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
    return [];
  }

  async getMapVolumes(_subjectId: string): Promise<MapVolume[]> {
    return [];
  }

  async getVolumes(_subjectId: string): Promise<VolumeFile[]> {
    return [];
  }

  async getMontageImages(_subjectId: string): Promise<Array<{ id: string; name: string; path: string }>> {
    return [];
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
