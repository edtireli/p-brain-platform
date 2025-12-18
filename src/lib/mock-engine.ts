import type {
  Project,
  Subject,
  Job,
  PipelineConfig,
  StageId,
  JobStatus,
  StageStatus,
  Artifact,
  VolumeInfo,
  Curve,
  PatlakData,
  ToftsData,
  DeconvolutionData,
  MetricsTable,
} from '@/types';
import { DEFAULT_CONFIG, STAGE_DEPENDENCIES, STAGE_NAMES } from '@/types';

class MockEngineAPI {
  private db = {
    projects: [] as Project[],
    subjects: [] as Subject[],
    jobs: [] as Job[],
    artifacts: [] as Artifact[],
  };

  private jobListeners: Set<(job: Job) => void> = new Set();
  private statusListeners: Set<(update: { subjectId: string; stageId: StageId; status: StageStatus }) => void> = new Set();
  private logListeners: Map<string, Set<(log: string) => void>> = new Map();

  constructor() {
    this.loadFromStorage();
    this.initializeDemoData();
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem('pbrain_db');
      if (stored) {
        this.db = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load from storage:', error);
    }
  }

  private saveToStorage() {
    try {
      localStorage.setItem('pbrain_db', JSON.stringify(this.db));
    } catch (error) {
      console.error('Failed to save to storage:', error);
    }
  }

  private initializeDemoData() {
    if (this.db.projects.length === 0) {
      const demoProject: Project = {
        id: 'demo_proj_001',
        name: 'DCE-MRI Study 2024',
        storagePath: '/Users/researcher/pbrain-projects/dce-study-2024',
        createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date().toISOString(),
        copyDataIntoProject: false,
        config: DEFAULT_CONFIG,
      };

      this.db.projects.push(demoProject);

      const demoSubjects: Subject[] = [
        {
          id: 'demo_subj_001',
          projectId: demoProject.id,
          name: 'subject_001',
          sourcePath: '/data/subjects/subject_001',
          createdAt: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
          updatedAt: new Date().toISOString(),
          hasT1: true,
          hasDCE: true,
          hasDiffusion: true,
          stageStatuses: {
            import: 'done',
            t1_fit: 'done',
            input_functions: 'done',
            time_shift: 'done',
            segmentation: 'done',
            tissue_ctc: 'done',
            modelling: 'done',
            diffusion: 'done',
            montage_qc: 'done',
          },
        },
        {
          id: 'demo_subj_002',
          projectId: demoProject.id,
          name: 'subject_002',
          sourcePath: '/data/subjects/subject_002',
          createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          updatedAt: new Date().toISOString(),
          hasT1: true,
          hasDCE: true,
          hasDiffusion: false,
          stageStatuses: {
            import: 'done',
            t1_fit: 'done',
            input_functions: 'done',
            time_shift: 'done',
            segmentation: 'done',
            tissue_ctc: 'done',
            modelling: 'failed',
            diffusion: 'not_run',
            montage_qc: 'not_run',
          },
        },
        {
          id: 'demo_subj_003',
          projectId: demoProject.id,
          name: 'subject_003',
          sourcePath: '/data/subjects/subject_003',
          createdAt: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
          updatedAt: new Date().toISOString(),
          hasT1: true,
          hasDCE: true,
          hasDiffusion: true,
          stageStatuses: {
            import: 'done',
            t1_fit: 'done',
            input_functions: 'running',
            time_shift: 'not_run',
            segmentation: 'not_run',
            tissue_ctc: 'not_run',
            modelling: 'not_run',
            diffusion: 'not_run',
            montage_qc: 'not_run',
          },
        },
      ];

      this.db.subjects.push(...demoSubjects);
      this.saveToStorage();
    }
  }

  onJobUpdate(listener: (job: Job) => void) {
    this.jobListeners.add(listener);
    return () => this.jobListeners.delete(listener);
  }

  onStatusUpdate(listener: (update: { subjectId: string; stageId: StageId; status: StageStatus }) => void) {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  onJobLogs(jobId: string, listener: (log: string) => void) {
    if (!this.logListeners.has(jobId)) {
      this.logListeners.set(jobId, new Set());
    }
    this.logListeners.get(jobId)!.add(listener);
    return () => {
      const listeners = this.logListeners.get(jobId);
      if (listeners) {
        listeners.delete(listener);
        if (listeners.size === 0) {
          this.logListeners.delete(jobId);
        }
      }
    };
  }

  private notifyJobUpdate(job: Job) {
    this.jobListeners.forEach(listener => listener(job));
  }

  private notifyStatusUpdate(subjectId: string, stageId: StageId, status: StageStatus) {
    this.statusListeners.forEach(listener => listener({ subjectId, stageId, status }));
  }

  private notifyJobLog(jobId: string, log: string) {
    const listeners = this.logListeners.get(jobId);
    if (listeners) {
      listeners.forEach(listener => listener(log));
    }
  }

  async getProjects(): Promise<Project[]> {
    return [...this.db.projects];
  }

  async getProject(id: string): Promise<Project | undefined> {
    return this.db.projects.find(p => p.id === id);
  }

  async createProject(data: {
    name: string;
    storagePath: string;
    copyDataIntoProject: boolean;
  }): Promise<Project> {
    const project: Project = {
      id: `proj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: data.name,
      storagePath: data.storagePath,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      copyDataIntoProject: data.copyDataIntoProject,
      config: DEFAULT_CONFIG,
    };

    this.db.projects.push(project);
    this.saveToStorage();
    return project;
  }

  async updateProjectConfig(
    projectId: string,
    configUpdate: Partial<PipelineConfig>
  ): Promise<Project | undefined> {
    const project = this.db.projects.find(p => p.id === projectId);
    if (project) {
      project.config = { ...project.config, ...configUpdate };
      project.updatedAt = new Date().toISOString();
      this.saveToStorage();
    }
    return project;
  }

  async getSubjects(projectId: string): Promise<Subject[]> {
    return this.db.subjects.filter(s => s.projectId === projectId);
  }

  async getSubject(id: string): Promise<Subject | undefined> {
    return this.db.subjects.find(s => s.id === id);
  }

  async importSubjects(
    projectId: string,
    subjects: Array<{ name: string; sourcePath: string }>
  ): Promise<Subject[]> {
    const imported: Subject[] = [];

    for (const subjectData of subjects) {
      const hasDCE = Math.random() > 0.1;
      const hasT1 = Math.random() > 0.05;
      const hasDiffusion = Math.random() > 0.3;

      const subject: Subject = {
        id: `subj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        projectId,
        name: subjectData.name,
        sourcePath: subjectData.sourcePath,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        hasT1,
        hasDCE,
        hasDiffusion,
        stageStatuses: {
          import: 'not_run',
          t1_fit: 'not_run',
          input_functions: 'not_run',
          time_shift: 'not_run',
          segmentation: 'not_run',
          tissue_ctc: 'not_run',
          modelling: 'not_run',
          diffusion: 'not_run',
          montage_qc: 'not_run',
        },
      };

      this.db.subjects.push(subject);
      imported.push(subject);
    }

    this.saveToStorage();
    return imported;
  }

  async createJob(data: {
    projectId: string;
    subjectId: string;
    stageId: StageId;
    parametersOverride?: Partial<PipelineConfig>;
  }): Promise<Job> {
    const job: Job = {
      id: `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      projectId: data.projectId,
      subjectId: data.subjectId,
      stageId: data.stageId,
      status: 'queued',
      progress: 0,
      currentStep: 'Initializing...',
      startTime: new Date().toISOString(),
    };

    this.db.jobs.push(job);
    this.saveToStorage();

    this.executeJob(job);

    return job;
  }

  private async executeJob(job: Job) {
    await new Promise(resolve => setTimeout(resolve, 500));

    const subject = this.db.subjects.find(s => s.id === job.subjectId);
    if (!subject) {
      job.status = 'failed';
      job.error = 'Subject not found';
      this.notifyJobUpdate(job);
      this.notifyJobLog(job.id, `[ERROR] Subject not found: ${job.subjectId}`);
      this.saveToStorage();
      return;
    }

    subject.stageStatuses[job.stageId] = 'running';
    this.notifyStatusUpdate(job.subjectId, job.stageId, 'running');
    this.saveToStorage();

    job.status = 'running';
    this.notifyJobUpdate(job);
    this.notifyJobLog(job.id, `[INFO] Starting ${STAGE_NAMES[job.stageId]} for subject ${subject.name}`);

    const steps = this.getStageSteps(job.stageId);
    const stepDuration = 2000;
    const totalDuration = steps.length * stepDuration;

    for (let i = 0; i < steps.length; i++) {
      job.progress = ((i + 1) / steps.length) * 100;
      job.currentStep = steps[i];
      job.estimatedTimeRemaining = Math.round(((steps.length - i - 1) * stepDuration) / 1000);
      this.notifyJobUpdate(job);
      this.notifyJobLog(job.id, `[PROGRESS] ${Math.round(job.progress)}% - ${steps[i]}`);

      await new Promise(resolve => setTimeout(resolve, stepDuration));

      if (Math.random() < 0.05 && i === Math.floor(steps.length / 2)) {
        job.status = 'failed';
        job.error = `Simulated error in step: ${steps[i]}`;
        job.endTime = new Date().toISOString();
        job.estimatedTimeRemaining = undefined;
        subject.stageStatuses[job.stageId] = 'failed';
        this.notifyJobUpdate(job);
        this.notifyStatusUpdate(job.subjectId, job.stageId, 'failed');
        this.notifyJobLog(job.id, `[ERROR] ${job.error}`);
        this.saveToStorage();
        return;
      }
    }

    job.status = 'completed';
    job.progress = 100;
    job.currentStep = 'Complete';
    job.endTime = new Date().toISOString();
    job.estimatedTimeRemaining = 0;
    subject.stageStatuses[job.stageId] = 'done';

    this.notifyJobUpdate(job);
    this.notifyStatusUpdate(job.subjectId, job.stageId, 'done');
    this.notifyJobLog(job.id, `[SUCCESS] ${STAGE_NAMES[job.stageId]} completed successfully`);
    this.saveToStorage();
  }

  private getStageSteps(stageId: StageId): string[] {
    const steps: Record<StageId, string[]> = {
      import: [
        'Scanning directory structure',
        'Validating NIfTI headers',
        'Indexing sequences',
        'Writing metadata',
      ],
      t1_fit: [
        'Loading IR series',
        'Computing magnitude model',
        'Fitting T1 per voxel',
        'Deriving M0 map',
        'Generating brain mask',
      ],
      input_functions: [
        'Loading concentration maps',
        'Identifying vessel candidates',
        'Extracting AIF',
        'Extracting VIF',
        'Validating curves',
      ],
      time_shift: [
        'Computing cross-correlation',
        'Determining optimal shift',
        'Applying time shift',
        'Rescaling to arterial peak',
        'Writing adjusted curves',
      ],
      segmentation: [
        'Checking FastSurfer availability',
        'Running tissue segmentation',
        'Registering to DCE space',
        'Extracting GM/WM masks',
        'Creating parcel ROIs',
      ],
      tissue_ctc: [
        'Loading tissue masks',
        'Extracting GM curve',
        'Extracting WM curve',
        'Computing parcel means',
        'Writing curves to JSON',
      ],
      modelling: [
        'Loading tissue curves',
        'Running Patlak analysis',
        'Running Extended Tofts',
        'Running deconvolution',
        'Computing perfusion metrics',
        'Writing parameter maps',
      ],
      diffusion: [
        'Loading diffusion volume',
        'Tensor fitting',
        'Computing FA/MD',
        'Writing metric maps',
      ],
      montage_qc: [
        'Rendering slice montages',
        'Overlaying segmentation',
        'Computing histograms',
        'Writing QC report',
      ],
    };

    return steps[stageId] || ['Processing...'];
  }

  async getJobs(filters?: {
    projectId?: string;
    subjectId?: string;
    status?: JobStatus;
  }): Promise<Job[]> {
    let jobs = [...this.db.jobs];

    if (filters?.projectId) {
      jobs = jobs.filter(j => j.projectId === filters.projectId);
    }
    if (filters?.subjectId) {
      jobs = jobs.filter(j => j.subjectId === filters.subjectId);
    }
    if (filters?.status) {
      jobs = jobs.filter(j => j.status === filters.status);
    }

    return jobs.sort((a, b) => {
      const aTime = a.startTime || '';
      const bTime = b.startTime || '';
      return bTime.localeCompare(aTime);
    });
  }

  async cancelJob(jobId: string): Promise<void> {
    const job = this.db.jobs.find(j => j.id === jobId);
    if (job && (job.status === 'queued' || job.status === 'running')) {
      job.status = 'cancelled';
      job.endTime = new Date().toISOString();
      this.notifyJobUpdate(job);
      this.saveToStorage();
    }
  }

  async retryJob(jobId: string): Promise<Job> {
    const oldJob = this.db.jobs.find(j => j.id === jobId);
    if (!oldJob) {
      throw new Error('Job not found');
    }

    return this.createJob({
      projectId: oldJob.projectId,
      subjectId: oldJob.subjectId,
      stageId: oldJob.stageId,
    });
  }

  async runFullPipeline(projectId: string, subjectIds: string[]): Promise<Job[]> {
    const jobs: Job[] = [];
    const stages: StageId[] = [
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

    for (const subjectId of subjectIds) {
      for (const stageId of stages) {
        const job = await this.createJob({ projectId, subjectId, stageId });
        jobs.push(job);
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    return jobs;
  }

  async runSubjectPipeline(projectId: string, subjectId: string): Promise<Job[]> {
    const stages: StageId[] = [
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

    const jobs: Job[] = [];
    for (const stageId of stages) {
      const job = await this.createJob({ projectId, subjectId, stageId });
      jobs.push(job);
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    return jobs;
  }

  async getVolumeInfo(path: string): Promise<VolumeInfo> {
    return {
      path,
      dimensions: [256, 256, 64, 80],
      voxelSize: [1.0, 1.0, 2.5],
      dataType: 'float32',
      min: 0,
      max: 4095,
    };
  }

  async getSliceData(
    path: string,
    z: number,
    t: number = 0
  ): Promise<{ data: number[][]; min: number; max: number }> {
    const size = 256;
    const data: number[][] = [];

    for (let y = 0; y < size; y++) {
      const row: number[] = [];
      for (let x = 0; x < size; x++) {
        const centerX = size / 2;
        const centerY = size / 2;
        const dx = x - centerX;
        const dy = y - centerY;
        const r = Math.sqrt(dx * dx + dy * dy);
        
        const brainRadius = size * 0.35;
        const value = r < brainRadius ? 1000 + Math.sin(r / 10 + z * 0.1 + t * 0.05) * 500 : 0;
        
        row.push(value);
      }
      data.push(row);
    }

    return { data, min: 0, max: 2000 };
  }

  async getCurves(subjectId: string): Promise<Curve[]> {
    const timePoints = Array.from({ length: 80 }, (_, i) => i * 2.5);
    
    const aif = timePoints.map(t => {
      const peak = 30;
      const amplitude = 5;
      const decay = 0.05;
      return Math.max(0, amplitude * Math.exp(-((t - peak) ** 2) / 100) * Math.exp(-decay * Math.max(0, t - peak)));
    });

    const vif = timePoints.map((t, i) => aif[i] * 0.6 + Math.random() * 0.1);
    
    const gmCurve = timePoints.map(t => {
      const peak = 40;
      const amplitude = 2;
      return Math.max(0, amplitude * (1 - Math.exp(-0.1 * Math.max(0, t - 10))));
    });

    const wmCurve = gmCurve.map(v => v * 0.4);

    return [
      {
        id: 'aif',
        name: 'Arterial Input Function (AIF)',
        timePoints,
        values: aif,
        unit: 'mM',
      },
      {
        id: 'vif',
        name: 'Venous Input Function (VIF)',
        timePoints,
        values: vif,
        unit: 'mM',
      },
      {
        id: 'gm',
        name: 'Gray Matter',
        timePoints,
        values: gmCurve,
        unit: 'mM',
      },
      {
        id: 'wm',
        name: 'White Matter',
        timePoints,
        values: wmCurve,
        unit: 'mM',
      },
    ];
  }

  async getPatlakData(subjectId: string, region: string): Promise<PatlakData> {
    const timePoints = Array.from({ length: 80 }, (_, i) => i * 2.5);
    const Ki = 0.02 + Math.random() * 0.01;
    const vp = 0.05 + Math.random() * 0.03;

    const x = timePoints.map((_, i) => i * 0.5 + Math.random() * 0.1);
    const y = x.map(xi => Ki * xi + vp + (Math.random() - 0.5) * 0.01);

    const windowStart = Math.floor(x.length * 0.4);
    const fitX = x.slice(windowStart);
    const fitY = y.slice(windowStart);

    const n = fitX.length;
    const sumX = fitX.reduce((a, b) => a + b, 0);
    const sumY = fitY.reduce((a, b) => a + b, 0);
    const sumXY = fitX.reduce((sum, xi, i) => sum + xi * fitY[i], 0);
    const sumX2 = fitX.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const fitLineX = [Math.min(...fitX), Math.max(...fitX)];
    const fitLineY = fitLineX.map(xi => slope * xi + intercept);

    const yMean = sumY / n;
    const ssTot = fitY.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0);
    const ssRes = fitY.reduce((sum, yi, i) => sum + (yi - (slope * fitX[i] + intercept)) ** 2, 0);
    const r2 = 1 - ssRes / ssTot;

    return {
      x,
      y,
      Ki: slope * 6000,
      vp: intercept,
      r2,
      fitLineX,
      fitLineY,
      windowStart,
    };
  }

  async getToftsData(subjectId: string, region: string): Promise<ToftsData> {
    const timePoints = Array.from({ length: 80 }, (_, i) => i * 2.5);
    const Ktrans = 0.1 + Math.random() * 0.05;
    const ve = 0.2 + Math.random() * 0.1;
    const vp = 0.05 + Math.random() * 0.02;

    const measured = timePoints.map(t => {
      const base = vp * 2 * Math.exp(-0.05 * t);
      const extracellular = Ktrans * t * Math.exp(-Ktrans * t / ve);
      return base + extracellular + (Math.random() - 0.5) * 0.05;
    });

    const fitted = timePoints.map(t => {
      const base = vp * 2 * Math.exp(-0.05 * t);
      const extracellular = Ktrans * t * Math.exp(-Ktrans * t / ve);
      return base + extracellular;
    });

    const residuals = measured.map((m, i) => m - fitted[i]);

    return {
      timePoints,
      measured,
      fitted,
      Ktrans,
      ve,
      vp,
      residuals,
    };
  }

  async getDeconvolutionData(subjectId: string, region: string): Promise<DeconvolutionData> {
    const timePoints = Array.from({ length: 80 }, (_, i) => i * 2.5);
    
    const residue = timePoints.map(t => Math.exp(-t / 15));
    
    const peak = residue[0];
    const normalized = residue.map(r => r / peak);
    
    const dR = normalized.map((r, i) => {
      if (i === 0) return 0;
      return -(r - normalized[i - 1]) / 2.5;
    });
    
    const integral = dR.reduce((sum, dr) => sum + dr * 2.5, 0);
    const h_t = dR.map(dr => dr / integral);
    
    const MTT = timePoints.reduce((sum, t, i) => sum + t * normalized[i] * 2.5, 0);
    
    const CTH = Math.sqrt(
      timePoints.reduce((sum, t, i) => sum + ((t - MTT) ** 2) * Math.abs(h_t[i]) * 2.5, 0)
    );

    const CBF = 60 + Math.random() * 20;

    return {
      timePoints,
      residue: normalized,
      h_t,
      CBF,
      MTT,
      CTH,
    };
  }

  async getMetricsTable(subjectId: string): Promise<MetricsTable> {
    const regions = [
      'Gray Matter',
      'White Matter',
      'Frontal Lobe',
      'Parietal Lobe',
      'Temporal Lobe',
      'Occipital Lobe',
    ];

    const rows = regions.map(region => ({
      region,
      Ki: 10 + Math.random() * 5,
      vp: 0.04 + Math.random() * 0.02,
      Ktrans: 0.08 + Math.random() * 0.04,
      ve: 0.18 + Math.random() * 0.08,
      CBF: 50 + Math.random() * 30,
      MTT: 4 + Math.random() * 2,
      CTH: 1.5 + Math.random() * 0.5,
    }));

    return { rows };
  }
}

export const mockEngine = new MockEngineAPI();
