export type StageId =
  | 'import'
  | 't1_fit'
  | 'input_functions'
  | 'time_shift'
  | 'segmentation'
  | 'tissue_ctc'
  | 'modelling'
  | 'diffusion'
  | 'tractography'
  | 'connectome';

export type StageStatus = 'not_run' | 'running' | 'done' | 'failed' | 'waiting';

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface AppSettings {
  firstName: string;
  onboardingCompleted: boolean;
  pbrainMainPy: string;
  fastsurferDir: string;
  freesurferHome: string;
}

export interface SystemDeps {
  pbrainMainPy: {
    configured: string;
    exists: boolean;
  };
  freesurfer: {
    reconAll: string;
    freesurferHome: string;
    ok: boolean;
  };
  fastsurfer: {
    fastsurferDir: string;
    runScript: string;
    ok: boolean;
  };
}

export interface ScanSystemDepsResponse {
  ok: boolean;
  applied: boolean;
  settingsPatch: Partial<AppSettings>;
  found: {
    pbrainMainPy: string;
    fastsurferDir: string;
    freesurferHome: string;
  };
  deps: SystemDeps;
}

export interface InstallPBrainResponse {
  ok: boolean;
  pbrainDir: string;
  pbrainMainPy: string;
}

export interface InstallPBrainRequirementsResponse {
  ok: boolean;
  command: string;
  pbrainDir: string;
  outputTail: string;
}

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
  t2Pattern: string;
  flairPattern: string;
  dcePattern: string;
  diffusionPattern: string;
  niftiSubfolder: string;
  useNestedStructure: boolean;
  aiModelsPath: string;
}

export type CtcModel = 'saturation' | 'turboflash' | 'advanced';

export type PkModel =
  | 'patlak'
  | 'tikhonov'
  | 'both';

export type T1FitMode = 'auto' | 'ir' | 'vfa' | 'none';

export interface CtcConfig {
  model: CtcModel;
  // TurboFLASH ky=0 index / turbo factor.
  // Use null/undefined to rely on NIfTI JSON sidecar metadata (auto).
  turboNph?: number | null;
  numberOfPeaks: number;

  // When selecting the TSCC/CTC "Max" curve, skip curves that have a forked/split apex.
  // Maps to p-brain Defaults JSON key: skipForkedMaxCtcPeaks
  skipForkedMaxCtcPeaks?: boolean;

  // Controls whether p-brain writes CTC peak/AUC maps and full 4D CTC exports.
  // Maps to p-brain Defaults JSON keys: writeCtcMaps, writeCtc4d, ctcMapSlice
  writeCtcMaps?: boolean;
  writeCtc4d?: boolean;
  ctcMapSlice?: number;

  // Global peak-rescale threshold for concentration curves.
  // When peak concentration exceeds this value, p-brain rescales the curve.
  peakRescaleThreshold?: number | null;
}

export type FlipAngleSetting = 'auto' | number;

export interface PBrainExecutionConfig {
  // Enforce metadata-first behavior and error when critical acquisition
  // metadata is missing (instead of silently defaulting).
  strictMetadata: boolean;

  // Multiprocessing toggle for p-brain subprocess runs.
  multiprocessing: boolean;

  // Core selection policy. Supports "auto" (80% cores), integer counts, or
  // a fraction in (0,1] (e.g. 0.8).
  cores: string;

  // Optional flip angle override for signal->concentration conversion.
  // Use 'auto' to rely on NIfTI sidecar metadata.
  flipAngle: FlipAngleSetting;

  // T1/M0 fit method selection.
  // Matches p-brain env: P_BRAIN_T1_FIT=auto|ir|vfa|none
  t1Fit?: T1FitMode;

  // Optional discovery overrides for non-standard acquisition naming.
  // These map to p-brain env vars used by series discovery:
  // - P_BRAIN_VFA_GLOB (comma-separated glob(s) relative to NIfTI dir)
  // - P_BRAIN_IR_PREFIXES (comma-separated filename prefixes)
  // - P_BRAIN_IR_TI (comma-separated TI values)
  vfaGlob?: string;
  irPrefixes?: string;
  irTi?: string;
}

export interface PipelineConfig {
  physiological: {
    r1: number;
    hematocrit: number;
    tissueDensity: number;
  };
  ctc: CtcConfig;
  pbrain: PBrainExecutionConfig;
  model: {
    pkModel?: PkModel;
    lambdaTikhonov: number;
    autoLambda: boolean;
    tikhonovPenalty?: 'identity' | 'derivative';
    residueEnforceNonneg?: boolean;
    residueEnforceMonotone?: boolean;
    patlakWindowStartFraction: number;
    patlakMinR2: number;
  };
  inputFunction: {
    // Which input-function curve to use for modelling.
    // - aif: pure arterial curve
    // - vif: pure venous curve (SSS)
    // - adjusted_vif: SSS-derived TSCC (time-shifted/rescaled)
    source: 'aif' | 'vif' | 'adjusted_vif';

    // How to summarize a vascular ROI mask into a representative curve.
    // Matches p-brain env vars:
    // - P_BRAIN_VASCULAR_ROI_CURVE_METHOD=max|mean
    // - P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX=1|0
    vascularRoiCurveMethod?: 'max' | 'mean' | 'median';
    vascularRoiAdaptiveMax?: boolean;

    ai?: {
      // AI confidence thresholds for slice selection.
      // Values are fractions in [0,1] (e.g. 0.5 = 50%).
      sliceConfStart?: number;
      sliceConfMin?: number;
      sliceConfStep?: number;

      // If AI fails to find AIF/VIF after the threshold sweep:
      // - 'deterministic': fall back automatically
      // - 'roi': require user interaction (stage becomes 'waiting')
      missingFallback?: 'deterministic' | 'roi';
    };
  };
  tissue?: {
    // How to summarize tissue/segmentation ROI voxels into representative curves/values.
    // Matches p-brain env: P_BRAIN_TISSUE_ROI_AGGREGATION=mean|median
    roiAggregation?: 'mean' | 'median';
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

	// Optional: a recommended default window for display (e.g. montage-style 2–98%).
	displayMin?: number;
	displayMax?: number;
}

export interface VolumeFile {
  id: string;
  name: string;
  path: string;
	kind?: 'dce' | 't1' | 't2' | 'flair' | 'diffusion' | 'analysis' | 'source' | string;
}

export interface RoiOverlay {
  id: string;
  roiType: string;
  roiSubType: string;
  sliceIndex: number;
  frameIndex?: number | null;

  // ROI voxel coordinates are saved as (row, col).
  row0: number;
  row1: number;
  col0: number;
  col1: number;
}

export interface ForcedRoiRef {
  roiType: 'Artery' | 'Vein';
  roiSubType: string;
  sliceIndex: number;
}

export interface InputFunctionForces {
  forcedAif: ForcedRoiRef | null;
  forcedVif: ForcedRoiRef | null;
}

export interface RoiMaskVolume {
  id: string;
  name: string;
  path: string;
  roiType: string;
  roiSubType: string;
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

export interface ConnectomeData {
  available: boolean;
  files: {
    matrix: string | null;
    labels: string | null;
    metrics: string | null;
    image?: string | null;
  };
  metrics: any | null;
  error?: string;
}

export interface TractographyData {
  path: string;
  // Array of streamlines, each a list of [x,y,z] points.
  streamlines: number[][][];
  // Optional per-streamline RGB colours (0-1 range), parallel to streamlines.
  colors?: number[][];
  totalStreamlines?: number;
  returned?: number;
  error?: string;
}

export interface ConnectomeData {
  available: boolean;
  // Absolute paths (backend-side) for debugging/inspection.
  files: {
    matrix: string | null;
    labels: string | null;
    metrics: string | null;
  };
  // Parsed JSON content from connectome_metrics.json
  metrics: Record<string, any> | null;
  error?: string;
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
  tractography: 'Tractography',
  connectome: 'Network Connectome',
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
  tractography: ['diffusion'],
  connectome: ['tractography'],
};

export const DEFAULT_FOLDER_STRUCTURE: FolderStructureConfig = {
  subjectFolderPattern: '{subject_id}',
  // Defaults aligned with p-brain `utils/parameters.py` conventions.
  // Patterns are comma-separated fallbacks (first match wins).
  t1Pattern: 'WIPcs_T1W_3D_TFE_32channel.nii*,*T1*.nii*',
  t2Pattern: 'WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,ax*WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,WIPAxT2TSEmatrix.nii*,*T2*.nii*',
  flairPattern: 'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,ax*WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,*FLAIR*.nii*',
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
  ctc: {
    model: 'turboflash',
    turboNph: null,
    numberOfPeaks: 2,
    peakRescaleThreshold: 4.0,
  },
  pbrain: {
    strictMetadata: false,
    multiprocessing: true,
    cores: 'auto',
    flipAngle: 'auto',
    t1Fit: 'ir',
    vfaGlob: '*VFA*.nii*',
    irPrefixes: 'WIPTI_,WIPDelRec-TI_',
    irTi: '00120,00300,00600,01000,02000,04000,10000',
  },
  model: {
    pkModel: 'both',
    lambdaTikhonov: 0.1,
    autoLambda: true,
    tikhonovPenalty: 'derivative',
    residueEnforceNonneg: true,
    residueEnforceMonotone: true,
    patlakWindowStartFraction: 0.4,
    patlakMinR2: 0.85,
  },
  inputFunction: {
    source: 'adjusted_vif',
    ai: {
      sliceConfStart: 0.5,
      sliceConfMin: 0.1,
      sliceConfStep: 0.05,
      missingFallback: 'deterministic',
    },
  },
  tissue: {
    roiAggregation: 'median',
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

export type ProjectAnalysisView = 'total' | 'atlas';

export interface ProjectAnalysisDatasetRow {
  subjectId: string;
  subjectName: string;
  region: string;
  [key: string]: any;
}

export interface ProjectAnalysisDataset {
  view: ProjectAnalysisView;
  rows: ProjectAnalysisDatasetRow[];
  regions: string[];
  metrics: string[];
}

export interface ProjectAnalysisPearsonResponse {
  n: number;
  r: number;
  p: number;
}

export interface ProjectAnalysisGroupCompareResponse {
  na: number;
  nb: number;
  meanA: number;
  meanB: number;
  t: number;
  t_p: number;
  mw_u: number;
  mw_p: number;
  cohen_d: number;
  shapiroA_p: number | null;
  shapiroB_p: number | null;
}

export interface ProjectAnalysisOlsCoefficient {
  name: string;
  beta: number;
  se: number;
  t: number;
  p: number;
}

export interface ProjectAnalysisOlsResponse {
  n: number;
  df_resid: number;
  r2: number;
  residual_shapiro_p: number | null;
  coefficients: ProjectAnalysisOlsCoefficient[];
}
