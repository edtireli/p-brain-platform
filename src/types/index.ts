export type StageId =
  | 'import'
  | 't1_fit'
  | 'input_functions'
  | 'time_shift'
  | 'segmentation'
  | 'tissue_ctc'
  | 'modelling'
  | 'diffusion'
  | 'montage_qc';

export type StageStatus = 'not_run' | 'running' | 'done' | 'failed';

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface Project {
  id: string;
  name: string;
  storagePath: string;
  createdAt: string;
  updatedAt: string;
  copyDataIntoProject: boolean;
  config: PipelineConfig;
}

export interface Subject {
  id: string;
  projectId: string;
  name: string;
  sourcePath: string;
  createdAt: string;
  updatedAt: string;
  hasT1: boolean;
  hasDCE: boolean;
  hasDiffusion: boolean;
  stageStatuses: Record<StageId, StageStatus>;
}

export interface FolderStructureConfig {
  subjectFolderPattern: string;
  t1Pattern: string;
  dcePattern: string;
  diffusionPattern: string;
  niftiSubfolder: string;
  useNestedStructure: boolean;
  aiModelsPath: string;
}

export interface PipelineConfig {
  physiological: {
    r1: number;
    hematocrit: number;
    tissueDensity: number;
  };
  model: {
    lambdaTikhonov: number;
    autoLambda: boolean;
    patlakWindowStartFraction: number;
    patlakMinR2: number;
  };
  inputFunction: {
    source: 'aif' | 'adjusted_vif';
  };
  voxelwise: {
    enabled: boolean;
    writeMTT: boolean;
    writeCTH: boolean;
  };
  externalTools: {
    dcm2niixPath?: string;
    fastsurferPath?: string;
    flirtPath?: string;
    mrtrixPath?: string;
  };
  aiModels: {
    aifModelPath?: string;
  };
  folderStructure: FolderStructureConfig;
}

export interface Job {
  id: string;
  projectId: string;
  subjectId: string;
  stageId: StageId;
  status: JobStatus;
  progress: number;
  currentStep: string;
  startTime?: string;
  endTime?: string;
  estimatedTimeRemaining?: number;
  error?: string;
  logPath?: string;
  logs?: Array<{
    timestamp: Date;
    level: 'info' | 'warning' | 'error';
    message: string;
  }>;
}

export interface Artifact {
  id: string;
  subjectId: string;
  stageId: StageId;
  type: 'volume' | 'curve' | 'table' | 'plot' | 'mask' | 'config';
  path: string;
  metadata: Record<string, any>;
  createdAt: string;
}

export interface VolumeInfo {
  path: string;
  dimensions: [number, number, number, number?];
  voxelSize: [number, number, number];
  dataType: string;
  min: number;
  max: number;
}

export interface VolumeFile {
  id: string;
  name: string;
  path: string;
  kind?: 'dce' | 't1' | 'diffusion' | string;
}

export interface MapVolume {
  id: string;
  name: string;
  unit: string;
  path: string;
  group: 'modelling' | 'diffusion';
}

export interface Curve {
  id: string;
  name: string;
  timePoints: number[];
  values: number[];
  unit: string;
  metadata?: Record<string, any>;
}

export interface PatlakData {
  x: number[];
  y: number[];
  Ki: number;
  vp: number;
  r2: number;
  fitLineX: number[];
  fitLineY: number[];
  windowStart: number;
}

export interface ToftsData {
  timePoints: number[];
  measured: number[];
  fitted: number[];
  Ktrans: number;
  ve: number;
  vp: number;
  residuals: number[];
}

export interface DeconvolutionData {
  timePoints: number[];
  residue: number[];
  h_t: number[];
  CBF: number;
  MTT: number;
  CTH: number;
}

export interface MetricsTable {
  rows: Array<{
    region: string;
    Ki?: number;
    vp?: number;
    Ktrans?: number;
    ve?: number;
    CBF?: number;
    MTT?: number;
    CTH?: number;
  }>;
}

export const STAGE_NAMES: Record<StageId, string> = {
  import: 'Import & Index',
  t1_fit: 'T1/M0 Fitting',
  input_functions: 'AIF/VIF Extraction',
  time_shift: 'Time Shifting',
  segmentation: 'Segmentation',
  tissue_ctc: 'Tissue Curves',
  modelling: 'Pharmacokinetic Modelling',
  diffusion: 'Diffusion Analysis',
  montage_qc: 'QC Montage',
};

export const STAGE_DEPENDENCIES: Record<StageId, StageId[]> = {
  import: [],
  t1_fit: ['import'],
  input_functions: ['t1_fit'],
  time_shift: ['input_functions'],
  segmentation: ['t1_fit'],
  tissue_ctc: ['segmentation', 'time_shift'],
  modelling: ['tissue_ctc'],
  diffusion: ['import'],
  montage_qc: ['modelling', 'segmentation'],
};

export const DEFAULT_FOLDER_STRUCTURE: FolderStructureConfig = {
  subjectFolderPattern: '{subject_id}',
  // Defaults aligned with p-brain `utils/parameters.py` conventions.
  // Patterns are comma-separated fallbacks (first match wins).
  t1Pattern: 'WIPcs_T1W_3D_TFE_32channel.nii*,*T1*.nii*',
  dcePattern: 'WIPDelRec-hperf120long.nii*,WIPhperf120long.nii*,*DCE*.nii*',
  diffusionPattern: 'Reg-DWInySENSE*.nii*,isoDWIb-1000*.nii*,WIPDTI_RSI_*.nii*,WIPDWI_RSI_*.nii*,*DTI*.nii*',
  niftiSubfolder: 'NIfTI',
  useNestedStructure: true,
  aiModelsPath: '/Users/edt/Desktop/p-brain/AI',
};

export const DEFAULT_CONFIG: PipelineConfig = {
  physiological: {
    r1: 4.39,
    hematocrit: 0.42,
    tissueDensity: 1.04,
  },
  model: {
    lambdaTikhonov: 0.1,
    autoLambda: false,
    patlakWindowStartFraction: 0.4,
    patlakMinR2: 0.85,
  },
  inputFunction: {
    source: 'aif',
  },
  voxelwise: {
    enabled: false,
    writeMTT: true,
    writeCTH: true,
  },
  externalTools: {},
  aiModels: {},
  folderStructure: DEFAULT_FOLDER_STRUCTURE,
};
