import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, UserPlus, ArrowLeft, X, List, CheckSquare, Square, MinusSquare, FolderOpen, Trash, Check } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Checkbox } from '@/components/ui/checkbox';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { engine } from '@/lib/engine';
import { playSuccessSound, playErrorSound, resumeAudioContext } from '@/lib/sounds';
import type { Project, Subject, StageId, StageStatus, Job, FolderStructureConfig, CtcModel, PkModel, T1FitMode } from '@/types';
import { STAGE_NAMES } from '@/types';
import { toast } from 'sonner';
import { JobMonitorPanel } from './JobMonitorPanel';
import { FolderStructureConfig as FolderStructureConfigComponent } from './FolderStructureConfig';
import { motion, AnimatePresence } from 'framer-motion';
import { DEFAULT_FOLDER_STRUCTURE } from '@/types';

interface DetectedSubject {
  name: string;
  path: string;
  selected: boolean;
}

interface ProjectDashboardProps {
  projectId: string;
  onBack: () => void;
  onSelectSubject: (subjectId: string) => void;
  onOpenAnalysis: () => void;
}

const STAGES: StageId[] = [
  'import',
  't1_fit',
  'input_functions',
  'time_shift',
  'segmentation',
  'tissue_ctc',
  'modelling',
  'diffusion',
  'tractography',
  'connectome',
];

function normalizeStageStatuses(stageStatuses: Partial<Record<StageId, StageStatus>> | undefined | null) {
  const out: Record<StageId, StageStatus> = {} as any;
  for (const s of STAGES) out[s] = 'not_run';
  if (stageStatuses) {
    for (const [k, v] of Object.entries(stageStatuses)) {
      if (!(STAGES as readonly string[]).includes(k) || !v) continue;
      // Back-compat: older values leaked into the DB.
      if (v === ('pending' as any)) {
        out[k as StageId] = 'not_run';
        continue;
      }
      out[k as StageId] = v as StageStatus;
    }
  }
  return out;
}

function mergeStageStatusesFromJobs(
  subject: Subject,
  jobs: Job[]
): Record<StageId, StageStatus> {
  const next = normalizeStageStatuses(subject.stageStatuses);

  const perStage = new Map<StageId, Job[]>();
  for (const j of jobs) {
    if (!j.subjectId) continue;
    if (j.subjectId !== subject.id) continue;
    const stageId = j.stageId as StageId;
    if (!STAGES.includes(stageId)) continue;
    const arr = perStage.get(stageId) || [];
    arr.push(j);
    perStage.set(stageId, arr);
  }

  // Precedence: running > failed > completed.
  // Do NOT treat queued jobs as running; queued work should not overwrite persisted stage_statuses.
  // Important: never let historical failures override a persisted `done` state (common after retries).
  for (const stageId of STAGES) {
    const arr = perStage.get(stageId) || [];
    if (arr.some(j => j.status === 'running')) {
      next[stageId] = 'running';
      continue;
    }

    // If p-brain signaled that this stage is waiting for user interaction,
    // keep that persisted state regardless of historical job outcomes.
    if (next[stageId] === 'waiting') {
      continue;
    }

    const isPersistedDone = next[stageId] === 'done';
    if (!isPersistedDone) {
      if (arr.some(j => j.status === 'failed')) {
        next[stageId] = 'failed';
        continue;
      }
      if (arr.some(j => j.status === 'cancelled')) {
        next[stageId] = 'failed';
        continue;
      }
    }

    if (arr.some(j => j.status === 'completed')) {
      next[stageId] = 'done';
      continue;
    }
  }

  return next;
}

export function ProjectDashboard({ projectId, onBack, onSelectSubject, onOpenAnalysis }: ProjectDashboardProps) {
  const [project, setProject] = useState<Project | null>(null);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isJobMonitorOpen, setIsJobMonitorOpen] = useState(false);
  const [isCtcDialogOpen, setIsCtcDialogOpen] = useState(false);
  const [draftCtcModel, setDraftCtcModel] = useState<CtcModel>('advanced');
  const [draftTurboNphAuto, setDraftTurboNphAuto] = useState<boolean>(true);
  const [draftTurboNph, setDraftTurboNph] = useState<number>(1);
  const [draftNumberOfPeaks, setDraftNumberOfPeaks] = useState<number>(2);
  const [draftPeakRescaleThreshold, setDraftPeakRescaleThreshold] = useState<number>(4.0);
  const [draftSkipForkedMaxCtcPeaks, setDraftSkipForkedMaxCtcPeaks] = useState<boolean>(true);
  const [draftWriteCtcMaps, setDraftWriteCtcMaps] = useState<boolean>(true);
  const [draftWriteCtc4d, setDraftWriteCtc4d] = useState<boolean>(true);
  const [draftCtcMapSlice, setDraftCtcMapSlice] = useState<number>(5);
  const [draftStrictMetadata, setDraftStrictMetadata] = useState<boolean>(false);
  const [draftMultiprocessing, setDraftMultiprocessing] = useState<boolean>(true);
  const [draftCoresAuto, setDraftCoresAuto] = useState<boolean>(true);
  const [draftCoresPercent, setDraftCoresPercent] = useState<number>(80);
  const [draftFlipAngleAuto, setDraftFlipAngleAuto] = useState<boolean>(true);
  const [draftFlipAngleDeg, setDraftFlipAngleDeg] = useState<number>(30);
  const [draftT1FitMode, setDraftT1FitMode] = useState<T1FitMode>('auto');
  const [draftAiSliceConfStartPct, setDraftAiSliceConfStartPct] = useState<number>(50);
  const [draftAiSliceConfMinPct, setDraftAiSliceConfMinPct] = useState<number>(10);
  const [draftAiSliceConfStepPct, setDraftAiSliceConfStepPct] = useState<number>(5);
  const [draftAiMissingFallback, setDraftAiMissingFallback] = useState<'deterministic' | 'roi'>('deterministic');
  const [draftInputFunctionSource, setDraftInputFunctionSource] = useState<'aif' | 'vif' | 'adjusted_vif'>('adjusted_vif');
  const [draftVascularRoiCurveMethod, setDraftVascularRoiCurveMethod] = useState<'max' | 'mean' | 'median'>('max');
  const [draftVascularRoiAdaptiveMax, setDraftVascularRoiAdaptiveMax] = useState<boolean>(true);
  const [draftTissueRoiAggregation, setDraftTissueRoiAggregation] = useState<'mean' | 'median'>('median');
  const [draftAutoLambda, setDraftAutoLambda] = useState<boolean>(false);
  const [draftPkModel, setDraftPkModel] = useState<PkModel>('both');
  const [activeJobsCount, setActiveJobsCount] = useState(0);
  const [runningSubjectIds, setRunningSubjectIds] = useState<Set<string>>(new Set());
  const [selectedSubjectIds, setSelectedSubjectIds] = useState<Set<string>>(new Set());
  const previousJobStatusesRef = useRef<Map<string, Job['status']>>(new Map());
  const lastSelectedIndexRef = useRef<number | null>(null);
  
  const [isDragging, setIsDragging] = useState(false);
  const [detectedSubjects, setDetectedSubjects] = useState<DetectedSubject[]>([]);
  const [droppedFolderName, setDroppedFolderName] = useState<string>('');
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const [draftFolderStructure, setDraftFolderStructure] = useState<FolderStructureConfig>(DEFAULT_FOLDER_STRUCTURE);

  const [isScanning, setIsScanning] = useState(false);

  useEffect(() => {
    loadProject();
    loadSubjects();

    const refreshActiveJobs = async () => {
      const jobs = await engine.getJobs({ projectId });
      const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
      setActiveJobsCount(active);

      // Keep stage dots correct even if realtime status updates are missed.
      setSubjects(prev =>
        prev.map(s => ({ ...s, stageStatuses: mergeStageStatusesFromJobs(s, jobs) }))
      );

      // Persist the per-subject running indicator across navigation by deriving it
      // from the current jobs table, not transient component state.
      const running = new Set(
        jobs
          .filter(j => j.status === 'running' || j.status === 'queued')
          .map(j => j.subjectId)
          .filter(Boolean)
      );
      setRunningSubjectIds(running);

      jobs.forEach(job => {
        previousJobStatusesRef.current.set(job.id, job.status);
      });

    };
    const unsubscribeStatus = engine.onStatusUpdate((update: { subjectId: string; stageId: any; status: any }) => {
      setSubjects(prev =>
        prev.map(s =>
          s.id === update.subjectId
            ? { ...s, stageStatuses: { ...s.stageStatuses, [update.stageId]: update.status } }
            : s
        )
      );
    });
    const unsubscribeJob = engine.onJobUpdate((job: Job) => {
      const previousStatus = previousJobStatusesRef.current.get(job.id);
      
      if (previousStatus && previousStatus !== job.status) {
        if (job.status === 'completed' && previousStatus === 'running') {
          playSuccessSound();
        } else if (job.status === 'failed' && previousStatus === 'running') {
          playErrorSound();
        }
      }
      
      previousJobStatusesRef.current.set(job.id, job.status);

      // Keep the per-subject running indicator in sync (queued/running => active).
      if (job.status === 'running' || job.status === 'queued') {
        setRunningSubjectIds(prev => {
          const next = new Set(prev);
          next.add(job.subjectId);
          return next;
        });
      }

      // Keep active job count in sync without aggressive polling.
      setActiveJobsCount(() => {
        const m = previousJobStatusesRef.current;
        m.set(job.id, job.status);
        let active = 0;
        m.forEach(s => {
          if (s === 'running' || s === 'queued') active += 1;
        });
        return active;
      });

      if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
        setRunningSubjectIds(prev => {
          const next = new Set(prev);
          next.delete(job.subjectId);
          return next;
        });
      }
    });

    const interval = setInterval(() => {
      if (document.visibilityState !== 'visible') return;
      refreshActiveJobs();
    }, 20000);

    refreshActiveJobs();

    return () => {
      unsubscribeStatus();
      unsubscribeJob();
      clearInterval(interval);
    };
  }, [projectId]);

  useEffect(() => {
    if (!project) return;
    const cfg: any = project.config || {};
    const ctc: any = cfg?.ctc || {};
    const modelCfg: any = cfg?.model || {};
    const inputFn: any = cfg?.inputFunction || {};

    {
      const raw = String(inputFn?.source || 'adjusted_vif').trim().toLowerCase();
      if (raw === 'aif') setDraftInputFunctionSource('aif');
      else if (raw === 'vif') setDraftInputFunctionSource('vif');
      else setDraftInputFunctionSource('adjusted_vif');
    }

    {
      const rawMethod = String(inputFn?.vascularRoiCurveMethod || 'max').trim().toLowerCase();
      if (rawMethod === 'mean') setDraftVascularRoiCurveMethod('mean');
      else if (rawMethod === 'median') setDraftVascularRoiCurveMethod('median');
      else setDraftVascularRoiCurveMethod('max');
      // Default true (matches p-brain default).
      setDraftVascularRoiAdaptiveMax(Boolean(inputFn?.vascularRoiAdaptiveMax ?? true));
    }
    const model = String(ctc?.model || 'advanced').toLowerCase();
    if (model === 'turboflash') setDraftCtcModel('turboflash');
    else if (model === 'advanced') setDraftCtcModel('advanced');
    else setDraftCtcModel('saturation');
    {
      // Auto when the override is missing/null/empty.
      const hasOverride = ctc && Object.prototype.hasOwnProperty.call(ctc, 'turboNph') && ctc.turboNph !== null && String(ctc.turboNph).trim() !== '';
      if (!hasOverride) {
        setDraftTurboNphAuto(true);
        setDraftTurboNph(1);
      } else {
        const rawNph = Number(ctc?.turboNph);
        const nph = Number.isFinite(rawNph) && rawNph >= 1 ? Math.floor(rawNph) : 1;
        setDraftTurboNphAuto(false);
        setDraftTurboNph(nph);
      }
    }
    const rawPeaks = Number(ctc?.numberOfPeaks);
    const peaks = Number.isFinite(rawPeaks) && rawPeaks >= 1 ? Math.floor(rawPeaks) : 2;
    setDraftNumberOfPeaks(peaks);

    const rawPeakThresh = Number(ctc?.peakRescaleThreshold);
    setDraftPeakRescaleThreshold(Number.isFinite(rawPeakThresh) && rawPeakThresh >= 0 ? rawPeakThresh : 4.0);

    // Default true (matches p-brain default).
    setDraftSkipForkedMaxCtcPeaks(Boolean(ctc?.skipForkedMaxCtcPeaks ?? true));
    setDraftWriteCtcMaps(Boolean(ctc?.writeCtcMaps ?? true));
    setDraftWriteCtc4d(Boolean(ctc?.writeCtc4d ?? true));
    {
      const rawSlice = Number(ctc?.ctcMapSlice);
      const slice = Number.isFinite(rawSlice) && rawSlice >= 1 ? Math.floor(rawSlice) : 5;
      setDraftCtcMapSlice(slice);
    }

    const pb = (project as any)?.config?.pbrain;
    setDraftStrictMetadata(Boolean(pb?.strictMetadata ?? false));
    setDraftMultiprocessing(Boolean(pb?.multiprocessing ?? true));
    {
      const raw = String(pb?.t1Fit ?? 'auto').trim().toLowerCase();
      if (raw === 'ir') setDraftT1FitMode('ir');
      else if (raw === 'vfa') setDraftT1FitMode('vfa');
      else if (raw === 'none') setDraftT1FitMode('none');
      else setDraftT1FitMode('auto');
    }
    {
      const raw = String(pb?.cores ?? 'auto').trim().toLowerCase();
      if (!raw || raw === 'auto') {
        setDraftCoresAuto(true);
        setDraftCoresPercent(80);
      } else {
        const v = Number(raw);
        setDraftCoresAuto(false);
        if (Number.isFinite(v)) {
          if (v > 0 && v <= 1) setDraftCoresPercent(Math.max(1, Math.min(100, Math.round(v * 100))));
          else if (v > 1 && v <= 100) setDraftCoresPercent(Math.max(1, Math.min(100, Math.round(v))));
          else setDraftCoresPercent(100);
        } else {
          setDraftCoresPercent(80);
        }
      }
    }
    {
      const raw = pb?.flipAngle;
      if (raw === undefined || raw === null || String(raw).trim() === '' || String(raw).trim().toLowerCase() === 'auto') {
        setDraftFlipAngleAuto(true);
      } else {
        const v = Number(raw);
        if (Number.isFinite(v) && v > 0) {
          setDraftFlipAngleAuto(false);
          setDraftFlipAngleDeg(v);
        } else {
          setDraftFlipAngleAuto(true);
        }
      }
    }

    setDraftAutoLambda(Boolean(modelCfg?.autoLambda ?? false));
    {
      const pkRaw = String(modelCfg?.pkModel ?? 'both').trim().toLowerCase();
      // Map legacy/removed options onto the single validated set.
      if (
        pkRaw === 'all' ||
        pkRaw === 'both' ||
        pkRaw === 'patlak_tikhonov' ||
        pkRaw === 'patlak_tikhonov_fast' ||
        pkRaw === 'patlak-then-tikhonov' ||
        pkRaw === 'patlak-then-tikhonov-fast' ||
        pkRaw === 'patlak_then_tikhonov' ||
        pkRaw === 'patlak_then_tikhonov_fast'
      ) {
        setDraftPkModel('both');
      } else if (pkRaw === 'patlak') {
        setDraftPkModel('patlak');
      } else if (
        pkRaw === 'tikhonov' ||
        pkRaw === 'tikhonov_only' ||
        pkRaw === 'tikhonov-only' ||
        pkRaw === 'tikhonov_fast' ||
        pkRaw === 'tikhonov-only-fast' ||
        pkRaw === 'tikhonov_only_fast' ||
        pkRaw === 'tik_fast' ||
        pkRaw === 'tik-fast' ||
        pkRaw === 'tikfast' ||
        pkRaw === 'tikh-fast'
      ) {
        setDraftPkModel('tikhonov');
      } else if (pkRaw === 'two_compartment' || pkRaw === '2comp' || pkRaw === 'two-comp' || pkRaw === 'two-compartment') {
        // Removed; keep behaviour deterministic.
        setDraftPkModel('tikhonov');
      } else {
        setDraftPkModel('both');
      }
    }

    {
      const ai: any = inputFn?.ai || {};
      const toPct = (v: any, fallback: number) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return fallback;
        // Accept either 0-1 fractions or 0-100 percentages.
        const pct = n <= 1 ? n * 100 : n;
        return Math.max(1, Math.min(100, Math.round(pct * 100) / 100));
      };
      setDraftAiSliceConfStartPct(toPct(ai?.sliceConfStart, 50));
      setDraftAiSliceConfMinPct(toPct(ai?.sliceConfMin, 10));
      setDraftAiSliceConfStepPct(toPct(ai?.sliceConfStep, 5));

      const fb = String(ai?.missingFallback || 'deterministic').trim().toLowerCase();
      setDraftAiMissingFallback(fb === 'roi' ? 'roi' : 'deterministic');
    }

    {
      const tissue: any = (project?.config as any)?.tissue || {};
      const agg = String(tissue?.roiAggregation || 'median').trim().toLowerCase();
      setDraftTissueRoiAggregation(agg === 'mean' ? 'mean' : 'median');
    }
  }, [project?.id, project?.updatedAt]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'a') {
        const target = e.target as HTMLElement;
        const isInputField = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;
        
        if (!isInputField && subjects.length > 0) {
          e.preventDefault();
          if (selectedSubjectIds.size === subjects.length) {
            setSelectedSubjectIds(new Set());
          } else {
            setSelectedSubjectIds(new Set(subjects.map(s => s.id)));
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [subjects, selectedSubjectIds]);

  const loadProject = async () => {
    const data = await engine.getProject(projectId);
    if (data) setProject(data);
  };

  const loadSubjects = async () => {
    const data = await engine.getSubjects(projectId);
    setSubjects(data.map(s => ({ ...s, stageStatuses: normalizeStageStatuses(s.stageStatuses) })));
  };

  const handleSaveFolderConfig = async (
    config: FolderStructureConfig,
    pbrainOverrides?: { vfaGlob?: string; irPrefixes?: string; irTi?: string }
  ) => {
    try {
      const patch: any = { folderStructure: config };
      if (pbrainOverrides && typeof pbrainOverrides === 'object') {
        const existing = (project as any)?.config?.pbrain || {};
        patch.pbrain = { ...existing, ...pbrainOverrides };
      }
      const updated = await engine.updateProjectConfig(projectId, patch);
      if (updated) {
        setProject(updated);
      }
    } catch (error) {
      toast.error('Failed to save folder configuration');
      console.error(error);
    }
  };

  const handleSaveCtcConfig = async () => {
    try {
      const nextNph = draftTurboNphAuto
        ? null
        : (Number.isFinite(draftTurboNph) && draftTurboNph >= 1 ? Math.floor(draftTurboNph) : 1);
      const nextPeaks = Number.isFinite(draftNumberOfPeaks) && draftNumberOfPeaks >= 1 ? Math.floor(draftNumberOfPeaks) : 2;
      const nextPeakRescaleThreshold = Number.isFinite(draftPeakRescaleThreshold) && draftPeakRescaleThreshold >= 0
        ? draftPeakRescaleThreshold
        : 4.0;
      const cores = (() => {
        if (!draftMultiprocessing) return '1';
        if (draftCoresAuto) return 'auto';
        const pct = Number(draftCoresPercent);
        const clamped = Number.isFinite(pct) ? Math.max(1, Math.min(100, pct)) : 80;
        const frac = clamped / 100;
        const rounded = Math.round(frac * 1000) / 1000;
        return String(rounded);
      })();

      const updated = await engine.updateProjectConfig(projectId, {
        ctc: {
          model: draftCtcModel,
          turboNph: nextNph,
          numberOfPeaks: nextPeaks,
          peakRescaleThreshold: nextPeakRescaleThreshold,
          skipForkedMaxCtcPeaks: Boolean(draftSkipForkedMaxCtcPeaks),
          writeCtcMaps: Boolean(draftWriteCtcMaps),
          writeCtc4d: Boolean(draftWriteCtc4d),
          ctcMapSlice: Math.max(1, Math.floor(Number(draftCtcMapSlice) || 5)),
        },
        pbrain: {
          strictMetadata: Boolean(draftStrictMetadata),
          multiprocessing: Boolean(draftMultiprocessing),
          cores,
          flipAngle: draftFlipAngleAuto ? 'auto' : Math.max(0.0001, Number(draftFlipAngleDeg) || 30),
          t1Fit: draftT1FitMode,
        },
        model: {
          pkModel: draftPkModel,
          autoLambda: Boolean(draftAutoLambda),
          tikhonovPenalty: 'derivative',
        },
        inputFunction: {
          source: (draftInputFunctionSource || 'adjusted_vif') as any,
          vascularRoiCurveMethod: (draftVascularRoiCurveMethod || 'max') as any,
          vascularRoiAdaptiveMax: Boolean(draftVascularRoiAdaptiveMax),
          ai: {
            sliceConfStart: Math.max(0.01, Math.min(1.0, (Number(draftAiSliceConfStartPct) || 50) / 100)),
            sliceConfMin: Math.max(0.01, Math.min(1.0, (Number(draftAiSliceConfMinPct) || 10) / 100)),
            sliceConfStep: Math.max(0.01, Math.min(0.5, (Number(draftAiSliceConfStepPct) || 5) / 100)),
            missingFallback: draftAiMissingFallback,
          },
        },
        tissue: {
          roiAggregation: draftTissueRoiAggregation,
        },
      });
      if (updated) {
        setProject(updated);
      }
      toast.success('Saved Defaults');
      setIsCtcDialogOpen(false);
    } catch (error) {
      toast.error('Failed to save Defaults');
      console.error(error);
    }
  };

  const processDroppedItems = useCallback(async (items: DataTransferItemList) => {
    const entries: FileSystemDirectoryEntry[] = [];
    
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file') {
        const entry = item.webkitGetAsEntry();
        if (entry?.isDirectory) {
          entries.push(entry as FileSystemDirectoryEntry);
        }
      }
    }

    if (entries.length === 0) {
      toast.error('Please drop a folder containing subject folders');
      return;
    }

    const rootEntry = entries[0];
    setDroppedFolderName(rootEntry.name);

    const subjectFolders: DetectedSubject[] = [];

    const reader = rootEntry.createReader();
    const readEntries = (): Promise<FileSystemEntry[]> => {
      return new Promise((resolve, reject) => {
        reader.readEntries(resolve, reject);
      });
    };

    try {
      const childEntries = await readEntries();
      
      for (const child of childEntries) {
        if (child.isDirectory && !child.name.startsWith('.')) {
          subjectFolders.push({
            name: child.name,
            path: child.name,
            selected: true,
          });
        }
      }

      subjectFolders.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

      if (subjectFolders.length === 0) {
        toast.error('No subject folders found in the dropped directory');
        return;
      }

      setDetectedSubjects(subjectFolders);
      toast.success(`Found ${subjectFolders.length} subject folder${subjectFolders.length > 1 ? 's' : ''}`);
    } catch (error) {
      console.error('Error reading directory:', error);
      toast.error('Failed to read folder contents');
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (dropZoneRef.current && !dropZoneRef.current.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.items) {
      processDroppedItems(e.dataTransfer.items);
    }
  }, [processDroppedItems]);

  const handleToggleSubject = (index: number) => {
    setDetectedSubjects(prev => 
      prev.map((s, i) => i === index ? { ...s, selected: !s.selected } : s)
    );
  };

  const handleSelectAllDetected = () => {
    const allSelected = detectedSubjects.every(s => s.selected);
    setDetectedSubjects(prev => prev.map(s => ({ ...s, selected: !allSelected })));
  };

  const handleImportSelected = async () => {
    const toImport = detectedSubjects.filter(s => s.selected);
    
    if (toImport.length === 0) {
      toast.error('No subjects selected for import');
      return;
    }

    const subjectsToImport = toImport.map(s => ({
      name: s.name,
      sourcePath: s.path,
    }));

    try {
      try {
        await engine.updateProjectConfig(projectId, { folderStructure: draftFolderStructure });
      } catch (err) {
        toast.error('Failed to save folder structure config');
        console.error(err);
      }
      await engine.importSubjects(projectId, subjectsToImport);
      toast.success(`Imported ${subjectsToImport.length} subject${subjectsToImport.length > 1 ? 's' : ''}`);
      setIsAddDialogOpen(false);
      setDetectedSubjects([]);
      setDroppedFolderName('');
      setDraftFolderStructure(DEFAULT_FOLDER_STRUCTURE);
      loadSubjects();
    } catch (error) {
      toast.error('Failed to import subjects');
      console.error(error);
    }
  };

  const handleClearDropped = () => {
    setDetectedSubjects([]);
    setDroppedFolderName('');
    setDraftFolderStructure(DEFAULT_FOLDER_STRUCTURE);
  };

  const handleScanAndImport = async () => {
    if (!project) return;
    setIsScanning(true);
    try {
      const result = await engine.scanProjectSubjects(projectId);
      const existing = new Set(subjects.map(s => s.name.toLowerCase()));
      const toImport = (result?.subjects || []).filter(s => !existing.has(s.name.toLowerCase()));

      if (toImport.length === 0) {
        toast.info('No new subject folders found in project storage path');
        return;
      }

      await engine.importSubjects(projectId, toImport);
      toast.success(`Imported ${toImport.length} subject${toImport.length > 1 ? 's' : ''} from storage`);
      loadSubjects();
    } catch (error) {
      toast.error('Failed to scan or import subjects');
      console.error(error);
    } finally {
      setIsScanning(false);
    }
  };

  const handleRunFullPipeline = async () => {
    if (subjects.length === 0) {
      toast.error('No subjects to process');
      return;
    }

    resumeAudioContext();
    setIsRunning(true);
    try {
      const subjectIds = subjects.map(s => s.id);
      setRunningSubjectIds(new Set(subjectIds));
      await engine.runFullPipeline(projectId, subjectIds);
      toast.success('Pipeline started for all subjects');
    } catch (error) {
      toast.error('Failed to start pipeline');
      console.error(error);
    } finally {
      setIsRunning(false);
    }
  };

  const handleRunSelectedPipeline = async () => {
    if (selectedSubjectIds.size === 0) {
      toast.error('No subjects selected');
      return;
    }

    resumeAudioContext();
    setIsRunning(true);
    try {
      const subjectIds = Array.from(selectedSubjectIds);
      setRunningSubjectIds(prev => new Set([...prev, ...subjectIds]));
      await engine.runFullPipeline(projectId, subjectIds);
      toast.success(`Pipeline started for ${subjectIds.length} subject${subjectIds.length > 1 ? 's' : ''}`);
      setSelectedSubjectIds(new Set());
    } catch (error) {
      toast.error('Failed to start pipeline');
      console.error(error);
    } finally {
      setIsRunning(false);
    }
  };

  const handleSelectSubjectToggle = (subjectId: string, index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    
    const isCtrlOrCmd = e.ctrlKey || e.metaKey;
    
    if (e.shiftKey && lastSelectedIndexRef.current !== null) {
      const start = Math.min(lastSelectedIndexRef.current, index);
      const end = Math.max(lastSelectedIndexRef.current, index);
      const rangeIds = subjects.slice(start, end + 1).map(s => s.id);
      
      setSelectedSubjectIds(prev => {
        const next = new Set(prev);
        rangeIds.forEach(id => next.add(id));
        return next;
      });
    } else if (isCtrlOrCmd) {
      setSelectedSubjectIds(prev => {
        const next = new Set(prev);
        if (next.has(subjectId)) {
          next.delete(subjectId);
        } else {
          next.add(subjectId);
        }
        return next;
      });
    } else {
      setSelectedSubjectIds(prev => {
        const next = new Set(prev);
        if (next.has(subjectId)) {
          next.delete(subjectId);
        } else {
          next.add(subjectId);
        }
        return next;
      });
      lastSelectedIndexRef.current = index;
    }
  };

  const handleSelectAll = () => {
    if (selectedSubjectIds.size === subjects.length) {
      setSelectedSubjectIds(new Set());
    } else {
      setSelectedSubjectIds(new Set(subjects.map(s => s.id)));
    }
  };

  const isAllSelected = subjects.length > 0 && selectedSubjectIds.size === subjects.length;
  const isSomeSelected = selectedSubjectIds.size > 0 && selectedSubjectIds.size < subjects.length;

  const handleRunSubjectPipeline = async (e: React.MouseEvent, subjectId: string, subjectName: string) => {
    e.stopPropagation();
    resumeAudioContext();
    
    setRunningSubjectIds(prev => new Set(prev).add(subjectId));
    try {
      await engine.runFullPipeline(projectId, [subjectId]);
      toast.success(`Pipeline started for ${subjectName}`);
    } catch (error) {
      toast.error(`Failed to start pipeline for ${subjectName}`);
      setRunningSubjectIds(prev => {
        const next = new Set(prev);
        next.delete(subjectId);
        return next;
      });
      console.error(error);
    }
  };

  const getStatusIndicator = (status: StageStatus) => {
    switch (status) {
      case 'done':
        return (
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-success/10">
            <div className="h-2 w-2 rounded-full bg-success" />
          </div>
        );
      case 'waiting':
        return (
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-warning/10">
            <div className="h-2 w-2 rounded-full bg-warning" />
          </div>
        );
      case 'failed':
        return (
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-destructive/10">
            <X size={12} weight="bold" className="text-destructive" />
          </div>
        );
      case 'running':
        return (
          <div className="relative flex h-6 w-6 items-center justify-center">
            <motion.div
              className="absolute h-6 w-6 rounded-full border-2 border-accent/30"
              animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0, 0.6] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div
              className="absolute h-4 w-4 rounded-full border-2 border-transparent border-t-accent"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
            <div className="h-2 w-2 rounded-full bg-accent" />
          </div>
        );
      default:
        return (
          <div className="flex h-6 w-6 items-center justify-center">
            <div className="h-1.5 w-1.5 rounded-full bg-border" />
          </div>
        );
    }
  };

  if (!project) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>;
  }

  const stages: StageId[] = STAGES;

  return (
    <div className="min-h-screen bg-background">
      <div className="border-b border-border bg-card">
        <div className="mx-auto max-w-full px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={onBack} className="gap-2">
                <ArrowLeft size={18} />
              </Button>
              <div>
                <h1 className="text-2xl font-medium tracking-tight">{project.name}</h1>
                <p className="mono text-xs text-muted-foreground mt-0.5">{project.storagePath}</p>
              </div>
            </div>

            <div className="flex gap-3 items-center">
              <AnimatePresence>
                {selectedSubjectIds.size > 0 && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.2 }}
                    className="flex items-center gap-3"
                  >
                    <span className="text-sm text-muted-foreground">
                      {selectedSubjectIds.size} selected
                    </span>
                    <Button 
                      onClick={handleRunSelectedPipeline} 
                      disabled={isRunning} 
                      className="gap-2 bg-accent hover:bg-accent/90"
                    >
                      <Play size={18} weight="fill" />
                      Run Selected ({selectedSubjectIds.size})
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedSubjectIds(new Set())}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Clear
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>

              <Button 
                variant="outline" 
                onClick={() => setIsJobMonitorOpen(true)}
                className="gap-2 relative"
              >
                <List size={20} weight="bold" />
                Jobs
                {activeJobsCount > 0 && (
                  <motion.span 
                    className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-xs font-medium text-accent-foreground"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, damping: 25 }}
                  >
                    <motion.span
                      animate={{ scale: [1, 1.1, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      {activeJobsCount}
                    </motion.span>
                  </motion.span>
                )}
              </Button>

              <Button
                variant="outline"
                onClick={onOpenAnalysis}
              >
                Analysis
              </Button>

              <Dialog open={isCtcDialogOpen} onOpenChange={setIsCtcDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline">
                    Defaults
                  </Button>
                </DialogTrigger>
                <DialogContent className="w-[95vw] max-w-2xl max-h-[85vh]">
                  <DialogHeader>
                    <DialogTitle>Defaults</DialogTitle>
                    <DialogDescription>
                      Project-level defaults applied to p-brain runs.
                    </DialogDescription>
                  </DialogHeader>

                  <ScrollArea className="max-h-[70vh] pr-4">
                    <div className="space-y-5 pb-2">
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">PK model</Label>
                          <div className="text-xs text-muted-foreground">Choose which pharmacokinetic model(s) to run.</div>
                        </div>
                        <Select value={draftPkModel} onValueChange={(v) => setDraftPkModel((v as PkModel) || 'both')}>
                          <SelectTrigger className="h-9 w-[200px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="both">Patlak + Tikhonov</SelectItem>
                            <SelectItem value="patlak">Patlak</SelectItem>
                            <SelectItem value="tikhonov">Tikhonov</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Auto lambda (L-curve)</Label>
                          <div className="text-xs text-muted-foreground">Automatically pick the Tikhonov lambda using an L-curve sweep.</div>
                        </div>
                        <Switch checked={draftAutoLambda} onCheckedChange={setDraftAutoLambda} />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Strict metadata</Label>
                          <div className="text-xs text-muted-foreground">Error when acquisition metadata is missing (no silent defaults).</div>
                        </div>
                        <Switch checked={draftStrictMetadata} onCheckedChange={setDraftStrictMetadata} />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Multiprocessing</Label>
                          <div className="text-xs text-muted-foreground">Enable multiprocessing in p-brain runs.</div>
                        </div>
                        <Switch checked={draftMultiprocessing} onCheckedChange={setDraftMultiprocessing} />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Cores</Label>
                          <div className="text-xs text-muted-foreground">Use Auto or set a percent of CPU cores.</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Auto</Label>
                            <Switch
                              checked={draftCoresAuto}
                              onCheckedChange={(v) => setDraftCoresAuto(Boolean(v))}
                              disabled={!draftMultiprocessing}
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <Slider
                              value={[Number.isFinite(draftCoresPercent) ? draftCoresPercent : 80]}
                              onValueChange={([v]) => {
                                const n = Number(v);
                                if (!Number.isFinite(n)) return;
                                setDraftCoresPercent(Math.max(1, Math.min(100, Math.round(n))));
                                setDraftCoresAuto(false);
                              }}
                              min={1}
                              max={100}
                              step={1}
                              className="w-[140px]"
                              disabled={!draftMultiprocessing || draftCoresAuto}
                            />
                            <div className="flex items-center gap-1">
                              <Input
                                type="number"
                                min={1}
                                max={100}
                                step={1}
                                value={String(Number.isFinite(draftCoresPercent) ? draftCoresPercent : 80)}
                                onChange={(e) => {
                                  const n = Number(e.target.value);
                                  if (!Number.isFinite(n)) return;
                                  setDraftCoresPercent(Math.max(1, Math.min(100, Math.round(n))));
                                  setDraftCoresAuto(false);
                                }}
                                className="h-9 w-[72px] text-right"
                                disabled={!draftMultiprocessing || draftCoresAuto}
                              />
                              <div className="text-xs text-muted-foreground">%</div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Flip angle</Label>
                          <div className="text-xs text-muted-foreground">Auto reads FlipAngle from the NIfTI JSON sidecar.</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Auto</Label>
                            <Switch checked={draftFlipAngleAuto} onCheckedChange={(v) => setDraftFlipAngleAuto(Boolean(v))} />
                          </div>
                          <div className="flex items-center gap-2">
                            <Input
                              type="number"
                              min={0}
                              step={0.1}
                              value={String(Number.isFinite(draftFlipAngleDeg) ? draftFlipAngleDeg : 30)}
                              onChange={(e) => setDraftFlipAngleDeg(Number(e.target.value) || 0)}
                              className="h-9 w-[120px] text-right"
                              disabled={draftFlipAngleAuto}
                            />
                            <div className="text-xs text-muted-foreground">deg</div>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">T1/M0 fit method</Label>
                          <div className="text-xs text-muted-foreground">Auto prefers IR if present, otherwise tries VFA.</div>
                        </div>
                        <Select value={draftT1FitMode} onValueChange={(v) => setDraftT1FitMode(((v as any) || 'auto') as T1FitMode)}>
                          <SelectTrigger className="h-9 w-[200px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="auto">auto</SelectItem>
                            <SelectItem value="ir">ir</SelectItem>
                            <SelectItem value="vfa">vfa</SelectItem>
                            <SelectItem value="none">none</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">AIF/VIF AI confidence sweep</Label>
                          <div className="text-xs text-muted-foreground">If AI can’t find AIF/VIF, lower the confidence threshold down to 10% (default).</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Start</Label>
                            <Input
                              type="number"
                              min={1}
                              max={100}
                              step={1}
                              value={String(Number.isFinite(draftAiSliceConfStartPct) ? draftAiSliceConfStartPct : 50)}
                              onChange={(e) => setDraftAiSliceConfStartPct(Math.max(1, Math.min(100, Math.round(Number(e.target.value) || 50))))}
                              className="h-9 w-[72px] text-right"
                            />
                            <div className="text-xs text-muted-foreground">%</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Min</Label>
                            <Input
                              type="number"
                              min={1}
                              max={100}
                              step={1}
                              value={String(Number.isFinite(draftAiSliceConfMinPct) ? draftAiSliceConfMinPct : 10)}
                              onChange={(e) => setDraftAiSliceConfMinPct(Math.max(1, Math.min(100, Math.round(Number(e.target.value) || 10))))}
                              className="h-9 w-[72px] text-right"
                            />
                            <div className="text-xs text-muted-foreground">%</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Step</Label>
                            <Input
                              type="number"
                              min={1}
                              max={50}
                              step={1}
                              value={String(Number.isFinite(draftAiSliceConfStepPct) ? draftAiSliceConfStepPct : 5)}
                              onChange={(e) => setDraftAiSliceConfStepPct(Math.max(1, Math.min(50, Math.round(Number(e.target.value) || 5))))}
                              className="h-9 w-[72px] text-right"
                            />
                            <div className="text-xs text-muted-foreground">%</div>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Missing AIF/VIF fallback</Label>
                          <div className="text-xs text-muted-foreground">If AI still can’t find AIF/VIF after the sweep.</div>
                        </div>
                        <Select value={draftAiMissingFallback} onValueChange={(v) => setDraftAiMissingFallback((v === 'roi' ? 'roi' : 'deterministic') as any)}>
                          <SelectTrigger className="h-9 w-[200px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="deterministic">deterministic</SelectItem>
                            <SelectItem value="roi">user ROI (wait)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Input function source</Label>
                          <div className="text-xs text-muted-foreground">Choose which curve p-brain uses as the modelling input.</div>
                        </div>
                        <Select
                          value={draftInputFunctionSource}
                          onValueChange={(v) => {
                            const raw = String(v || '').trim().toLowerCase();
                            if (raw === 'aif') setDraftInputFunctionSource('aif');
                            else if (raw === 'vif') setDraftInputFunctionSource('vif');
                            else setDraftInputFunctionSource('adjusted_vif');
                          }}
                        >
                          <SelectTrigger className="h-9 w-[200px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="aif">aif (arterial)</SelectItem>
                            <SelectItem value="vif">vif (venous)</SelectItem>
                              <SelectItem value="adjusted_vif">advanced VIF (TSCC)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Vascular ROI curve</Label>
                          <div className="text-xs text-muted-foreground">How to summarize the vascular ROI mask into a single curve.</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Select
                            value={draftVascularRoiCurveMethod}
                            onValueChange={(v) => {
                              const raw = String(v || '').trim().toLowerCase();
                              if (raw === 'mean') setDraftVascularRoiCurveMethod('mean');
                              else if (raw === 'median') setDraftVascularRoiCurveMethod('median');
                              else setDraftVascularRoiCurveMethod('max');
                            }}
                          >
                            <SelectTrigger className="h-9 w-[140px]">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="max">max</SelectItem>
                              <SelectItem value="mean">mean</SelectItem>
                              <SelectItem value="median">median</SelectItem>
                            </SelectContent>
                          </Select>

                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Adaptive max</Label>
                            <Switch
                              checked={draftVascularRoiAdaptiveMax}
                              onCheckedChange={(v) => setDraftVascularRoiAdaptiveMax(Boolean(v))}
                              disabled={draftVascularRoiCurveMethod !== 'max'}
                            />
                          </div>
                        </div>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Tissue/parcel aggregation</Label>
                          <div className="text-xs text-muted-foreground">Mean or median of normalized tissue/atlas ROI curves and values.</div>
                        </div>
                        <Select
                          value={draftTissueRoiAggregation}
                          onValueChange={(v) => setDraftTissueRoiAggregation((String(v).toLowerCase() === 'mean' ? 'mean' : 'median') as any)}
                        >
                          <SelectTrigger className="h-9 w-[200px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="median">median</SelectItem>
                            <SelectItem value="mean">mean</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">CTC model</Label>
                          <div className="text-xs text-muted-foreground">Signal-to-concentration conversion model.</div>
                        </div>
                        <Select
                          value={draftCtcModel}
                          onValueChange={(v) => {
                            const raw = String(v || '').toLowerCase();
                            if (raw === 'turboflash') setDraftCtcModel('turboflash');
                            else if (raw === 'advanced') setDraftCtcModel('advanced');
                            else setDraftCtcModel('saturation');
                          }}
                        >
                          <SelectTrigger className="h-9 w-[200px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="advanced">advanced (TurboFLASH case12)</SelectItem>
                            <SelectItem value="turboflash">turboflash</SelectItem>
                            <SelectItem value="saturation">saturation</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">TurboFLASH nph</Label>
                          <div className="text-xs text-muted-foreground">Auto reads nph from the NIfTI JSON sidecar (fallback 1).</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-2">
                            <Label className="text-[11px] text-muted-foreground">Auto</Label>
                            <Switch
                              checked={draftTurboNphAuto}
                              onCheckedChange={(v) => {
                                const nextAuto = Boolean(v);
                                setDraftTurboNphAuto(nextAuto);
                                if (!nextAuto) {
                                  setDraftTurboNph(prev => (Number.isFinite(prev) && prev >= 1 ? Math.floor(prev) : 1));
                                }
                              }}
                              disabled={!['turboflash', 'advanced'].includes(draftCtcModel)}
                            />
                          </div>
                          <Input
                            type="number"
                            min={1}
                            step={1}
                            value={Number.isFinite(draftTurboNph) ? String(draftTurboNph) : '1'}
                            onChange={(e) => setDraftTurboNph(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
                            disabled={!['turboflash', 'advanced'].includes(draftCtcModel) || draftTurboNphAuto}
                            className="h-9 w-[120px]"
                          />
                        </div>
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Number of peaks</Label>
                          <div className="text-xs text-muted-foreground">Used for peak-based alignment and choosing the TSCC “Max” curve.</div>
                        </div>
                        <Input
                          type="number"
                          min={1}
                          step={1}
                          value={Number.isFinite(draftNumberOfPeaks) ? String(draftNumberOfPeaks) : '2'}
                          onChange={(e) => setDraftNumberOfPeaks(Math.max(1, Math.floor(Number(e.target.value) || 2)))}
                          className="h-9 w-[200px]"
                        />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Skip forked Max TSCC/CTC peaks</Label>
                          <div className="text-xs text-muted-foreground">Exclude split/forked apex curves when selecting the “Max” TSCC/CTC curve.</div>
                        </div>
                        <Switch checked={draftSkipForkedMaxCtcPeaks} onCheckedChange={(v) => setDraftSkipForkedMaxCtcPeaks(Boolean(v))} />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Write CTC maps</Label>
                          <div className="text-xs text-muted-foreground">Export CTC peak/AUC maps (NIfTI/PNG/NPY) during tissue CTC stage.</div>
                        </div>
                        <Switch checked={draftWriteCtcMaps} onCheckedChange={(v) => setDraftWriteCtcMaps(Boolean(v))} />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Write CTC 4D</Label>
                          <div className="text-xs text-muted-foreground">Export full 4D concentration time-series as NIfTI/NPY during tissue CTC stage.</div>
                        </div>
                        <Switch checked={draftWriteCtc4d} onCheckedChange={(v) => setDraftWriteCtc4d(Boolean(v))} />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">CTC map slice</Label>
                          <div className="text-xs text-muted-foreground">1-based slice index used for map PNG exports.</div>
                        </div>
                        <Input
                          type="number"
                          min={1}
                          step={1}
                          value={String(Number.isFinite(draftCtcMapSlice) ? draftCtcMapSlice : 5)}
                          onChange={(e) => setDraftCtcMapSlice(Math.max(1, Math.floor(Number(e.target.value) || 5)))}
                          className="h-9 w-[200px]"
                          disabled={!draftWriteCtcMaps}
                        />
                      </div>

                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <Label className="text-xs font-medium">Peak rescale threshold</Label>
                          <div className="text-xs text-muted-foreground">If peak exceeds this, p-brain rescales the curve.</div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Slider
                            value={[Number.isFinite(draftPeakRescaleThreshold) ? draftPeakRescaleThreshold : 4.0]}
                            onValueChange={([v]) => {
                              const n = Number(v);
                              if (!Number.isFinite(n)) return;
                              setDraftPeakRescaleThreshold(Math.max(0, Math.min(10, n)));
                            }}
                            min={0}
                            max={10}
                            step={0.1}
                            className="w-[140px]"
                          />
                          <Input
                            type="number"
                            min={0}
                            step={0.1}
                            value={Number.isFinite(draftPeakRescaleThreshold) ? String(draftPeakRescaleThreshold) : '4.0'}
                            onChange={(e) => {
                              const n = Number(e.target.value);
                              if (!Number.isFinite(n)) return;
                              setDraftPeakRescaleThreshold(Math.max(0, Math.min(10, n)));
                            }}
                            className="h-9 w-[72px] text-right"
                          />
                        </div>
                      </div>

                      <div className="flex justify-end gap-2 pt-2">
                        <Button variant="outline" onClick={() => setIsCtcDialogOpen(false)}>
                          Cancel
                        </Button>
                        <Button onClick={handleSaveCtcConfig}>
                          Save
                        </Button>
                      </div>
                    </div>
                  </ScrollArea>
                </DialogContent>
              </Dialog>

              <FolderStructureConfigComponent 
                project={project}
                onSave={handleSaveFolderConfig}
              />

              <Dialog open={isAddDialogOpen} onOpenChange={(open) => {
                setIsAddDialogOpen(open);
                if (!open) {
                  setDetectedSubjects([]);
                  setDroppedFolderName('');
                  setIsDragging(false);
                }
              }}>
                <DialogContent className="max-w-lg">
                  <DialogHeader>
                    <DialogTitle>Import Subjects</DialogTitle>
                    <DialogDescription>
                      Drag and drop a folder containing subject folders to import them into the pipeline.
                    </DialogDescription>
                  </DialogHeader>

                  <div className="space-y-4">
                    {detectedSubjects.length === 0 ? (
                      <div
                        ref={dropZoneRef}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 transition-all ${
                          isDragging 
                            ? 'border-accent bg-accent/5 scale-[1.02]' 
                            : 'border-border hover:border-muted-foreground/50'
                        }`}
                      >
                        <AnimatePresence>
                          {isDragging && (
                            <motion.div 
                              className="absolute inset-0 rounded-lg bg-accent/10"
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                            />
                          )}
                        </AnimatePresence>
                        <FolderOpen 
                          size={48} 
                          weight={isDragging ? 'fill' : 'duotone'} 
                          className={`mb-3 transition-colors ${isDragging ? 'text-accent' : 'text-muted-foreground'}`} 
                        />
                        <p className="text-sm font-medium text-foreground">
                          {isDragging ? 'Drop folder here' : 'Drop a subjects folder here'}
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          The folder should contain individual subject folders
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <FolderOpen size={18} className="text-accent" />
                            <span className="mono text-sm font-medium">{droppedFolderName}</span>
                            <Badge variant="secondary" className="text-xs">
                              {detectedSubjects.filter(s => s.selected).length} / {detectedSubjects.length} selected
                            </Badge>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleClearDropped}
                            className="h-7 px-2 text-muted-foreground hover:text-foreground"
                          >
                            <Trash size={16} />
                          </Button>
                        </div>
                        
                        <div className="flex items-center gap-2 border-b border-border pb-2">
                          <Checkbox
                            id="select-all-detected"
                            checked={detectedSubjects.every(s => s.selected)}
                            onCheckedChange={handleSelectAllDetected}
                          />
                          <label 
                            htmlFor="select-all-detected" 
                            className="text-xs font-medium text-muted-foreground cursor-pointer"
                          >
                            Select all
                          </label>
                        </div>

                        <ScrollArea className="h-[280px] pr-3">
                          <div className="space-y-1">
                            {detectedSubjects.map((subject, index) => (
                              <motion.div
                                key={subject.path}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.02 }}
                                className={`flex items-center gap-3 rounded-md px-3 py-2 transition-colors ${
                                  subject.selected ? 'bg-accent/10' : 'hover:bg-muted/50'
                                }`}
                              >
                                <Checkbox
                                  id={`subject-${index}`}
                                  checked={subject.selected}
                                  onCheckedChange={() => handleToggleSubject(index)}
                                />
                                <label 
                                  htmlFor={`subject-${index}`}
                                  className="flex-1 cursor-pointer"
                                >
                                  <span className="mono text-sm">{subject.name}</span>
                                </label>
                                {subject.selected && (
                                  <Check size={14} className="text-success" />
                                )}
                              </motion.div>
                            ))}
                          </div>
                        </ScrollArea>

                        <div className="space-y-3 rounded-lg border border-border p-4">
                          <div className="space-y-1">
                            <div className="text-xs font-medium text-muted-foreground">Folder matching</div>
                            <div className="text-xs text-muted-foreground">
                              Configure how the worker finds NIfTI files inside each subject folder.
                            </div>
                          </div>

                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="space-y-1">
                              <Label className="text-xs text-muted-foreground">Subject folder pattern</Label>
                              <Input
                                value={draftFolderStructure.subjectFolderPattern}
                                onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, subjectFolderPattern: e.target.value }))}
                                placeholder="{subject_id}"
                                className="mono"
                              />
                            </div>
                            <div className="space-y-1">
                              <Label className="text-xs text-muted-foreground">NIfTI subfolder</Label>
                              <Input
                                value={draftFolderStructure.niftiSubfolder}
                                onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, niftiSubfolder: e.target.value }))}
                                placeholder="NIfTI"
                                className="mono"
                              />
                            </div>
                          </div>

                          <div className="flex items-start justify-between gap-4">
                            <div className="space-y-1">
                              <Label className="text-xs font-medium">Treat data as nested structure</Label>
                              <div className="text-xs text-muted-foreground">If enabled, the worker looks under the NIfTI subfolder for volumes.</div>
                            </div>
                            <Switch
                              checked={draftFolderStructure.useNestedStructure}
                              onCheckedChange={(checked) => setDraftFolderStructure(prev => ({ ...prev, useNestedStructure: checked }))}
                            />
                          </div>

                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">T1 filename patterns (comma-separated)</Label>
                            <Input
                              value={draftFolderStructure.t1Pattern}
                              onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, t1Pattern: e.target.value }))}
                              className="mono"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">DCE filename patterns (comma-separated)</Label>
                            <Input
                              value={draftFolderStructure.dcePattern}
                              onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, dcePattern: e.target.value }))}
                              className="mono"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">Diffusion filename patterns (comma-separated)</Label>
                            <Input
                              value={draftFolderStructure.diffusionPattern}
                              onChange={(e) => setDraftFolderStructure(prev => ({ ...prev, diffusionPattern: e.target.value }))}
                              className="mono"
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="flex justify-end gap-3 pt-2">
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={() => {
                          setIsAddDialogOpen(false);
                          setDetectedSubjects([]);
                          setDroppedFolderName('');
                        }}
                      >
                        Cancel
                      </Button>
                      <Button 
                        onClick={handleImportSelected}
                        disabled={detectedSubjects.filter(s => s.selected).length === 0}
                        className="gap-2"
                      >
                        <UserPlus size={18} />
                        Import {detectedSubjects.filter(s => s.selected).length > 0 
                          ? `(${detectedSubjects.filter(s => s.selected).length})` 
                          : ''}
                      </Button>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>

              <Button onClick={handleRunFullPipeline} disabled={isRunning || subjects.length === 0} className="gap-2">
                <Play size={20} weight="fill" />
                Run All
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-full p-6">
        {subjects.length === 0 ? (
          <Card
            className={`border-dashed ${isDragging ? 'border-accent bg-accent/5' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsDragging(false);
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsDragging(false);
              setIsAddDialogOpen(true);
              if (e.dataTransfer.items) processDroppedItems(e.dataTransfer.items);
            }}
          >
            <CardContent className="flex flex-col items-center justify-center py-16">
              <UserPlus size={64} className="mb-4 text-muted-foreground" />
              <h3 className="mb-2 text-base font-medium">No subjects yet</h3>
              <p className="mb-6 text-center text-sm text-muted-foreground">
                {isDragging ? 'Drop a subjects folder here' : 'Add subjects to begin neuroimaging analysis'}
              </p>
              <div className="flex items-center gap-3">
                <Button onClick={() => setIsAddDialogOpen(true)} className="gap-2">
                  <UserPlus size={20} weight="bold" />
                  Add Subjects
                </Button>
                <Button variant="secondary" onClick={handleScanAndImport} disabled={isScanning} className="gap-2">
                  <FolderOpen size={18} />
                  {isScanning ? 'Importing…' : 'Automatically Import subjects'}
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-0 shadow-sm">
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <TooltipProvider>
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="sticky left-0 z-10 bg-[oklch(0.92_0.005_250)] px-3 py-3 text-center w-10">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                onClick={handleSelectAll}
                                className="flex h-6 w-6 items-center justify-center rounded transition-colors hover:bg-muted"
                              >
                                {isAllSelected ? (
                                  <CheckSquare size={18} weight="fill" className="text-primary" />
                                ) : isSomeSelected ? (
                                  <MinusSquare size={18} weight="fill" className="text-primary" />
                                ) : (
                                  <Square size={18} className="text-muted-foreground" />
                                )}
                              </button>
                            </TooltipTrigger>
                            <TooltipContent>
                              {isAllSelected ? 'Deselect all' : 'Select all'}
                            </TooltipContent>
                          </Tooltip>
                        </th>
                        <th className="sticky left-10 z-10 bg-[oklch(0.92_0.005_250)] px-2 py-3 text-center text-xs font-medium uppercase tracking-wider w-12">
                          Run
                        </th>
                        <th className="sticky left-[5.5rem] z-10 bg-[oklch(0.92_0.005_250)] px-4 py-3 text-left text-xs font-medium uppercase tracking-wider shadow-[4px_0_6px_-4px_rgba(0,0,0,0.1)]">
                          Subject
                        </th>
                        {stages.map(stageId => (
                          <th
                            key={stageId}
                            className="px-3 py-3 text-center text-xs font-medium uppercase tracking-wider"
                          >
                            <div className="whitespace-nowrap">{STAGE_NAMES[stageId]}</div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {subjects.map((subject, index) => {
                        const isSubjectRunning = runningSubjectIds.has(subject.id) ||
                          Object.values(subject.stageStatuses || {}).some(s => s === 'running');
                        const hasEverRun = Object.values(subject.stageStatuses || {}).some(s => s === 'done' || s === 'failed' || s === 'waiting');
                        const isSelected = selectedSubjectIds.has(subject.id);
                        
                        return (
                          <motion.tr
                            key={subject.id}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.18, delay: Math.min(index * 0.015, 0.25) }}
                            className={`cursor-pointer border-b border-border transition-colors ${isSelected ? 'bg-[oklch(0.94_0.02_250)]' : 'hover:bg-muted/30'}`}
                            onClick={() => onSelectSubject(subject.id)}
                          >
                            <td className={`sticky left-0 z-10 px-3 py-3 ${isSelected ? 'bg-[oklch(0.94_0.02_250)]' : 'bg-card'}`}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={(e) => handleSelectSubjectToggle(subject.id, index, e)}
                                    className="flex h-6 w-6 items-center justify-center rounded transition-colors hover:bg-muted"
                                  >
                                    {isSelected ? (
                                      <CheckSquare size={18} weight="fill" className="text-primary" />
                                    ) : (
                                      <Square size={18} className="text-muted-foreground hover:text-foreground" />
                                    )}
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <span>Click to select</span>
                                  <span className="block text-xs text-muted-foreground">Shift+click for range · Ctrl+click to toggle</span>
                                </TooltipContent>
                              </Tooltip>
                            </td>
                            <td className={`sticky left-10 z-10 px-2 py-3 ${isSelected ? 'bg-[oklch(0.94_0.02_250)]' : 'bg-card'}`}>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <button
                                    onClick={(e) => handleRunSubjectPipeline(e, subject.id, subject.name)}
                                    disabled={isSubjectRunning}
                                    className={`flex h-8 w-8 items-center justify-center rounded-md transition-all ${
                                      isSubjectRunning 
                                        ? 'cursor-not-allowed opacity-50' 
                                        : 'hover:bg-primary hover:text-primary-foreground text-muted-foreground hover:scale-110'
                                    }`}
                                  >
                                    {isSubjectRunning ? (
                                      <motion.div
                                        className="h-4 w-4 rounded-full border-2 border-transparent border-t-accent"
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                      />
                                    ) : (
                                      <Play size={16} weight="fill" />
                                    )}
                                  </button>
                                </TooltipTrigger>
                                <TooltipContent side="right">
                                  {isSubjectRunning
                                    ? 'Pipeline running...'
                                    : hasEverRun
                                      ? 'Rerun full pipeline for this subject'
                                      : 'Run full pipeline for this subject'}
                                </TooltipContent>
                              </Tooltip>
                            </td>
                            <td className={`sticky left-[5.5rem] z-10 px-4 py-3 font-normal shadow-[4px_0_6px_-4px_rgba(0,0,0,0.1)] ${isSelected ? 'bg-[oklch(0.94_0.02_250)]' : 'bg-card'}`}>
                              <div className="flex flex-col gap-1">
                                <span className="text-sm">{subject.name}</span>
                                <div className="flex gap-2">
                                  {subject.hasT1 && (
                                    <Badge variant="secondary" className="text-xs font-normal px-1.5 py-0">
                                      T1
                                    </Badge>
                                  )}
                                  {subject.hasDCE && (
                                    <Badge variant="secondary" className="text-xs font-normal px-1.5 py-0">
                                      DCE
                                    </Badge>
                                  )}
                                  {subject.hasDiffusion && (
                                    <Badge variant="secondary" className="text-xs font-normal px-1.5 py-0">
                                      DTI
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </td>
                            {stages.map(stageId => {
                              const status = subject.stageStatuses[stageId];
                              return (
                                <td key={stageId} className="px-3 py-3">
                                  <div className="flex justify-center">
                                    {getStatusIndicator(status)}
                                  </div>
                                </td>
                              );
                            })}
                          </motion.tr>
                        );
                      })}
                    </tbody>
                  </table>
                </TooltipProvider>
              </div>

              <button
                type="button"
                onClick={() => setIsAddDialogOpen(true)}
                className="flex w-full items-center justify-center gap-2 border-t border-border bg-card px-4 py-3 text-sm font-medium text-foreground transition-colors hover:bg-muted/30"
              >
                <UserPlus size={18} weight="bold" />
                Add Subjects
              </button>
            </CardContent>
          </Card>
        )}
      </div>

      <JobMonitorPanel
        projectId={projectId}
        isOpen={isJobMonitorOpen}
        onClose={() => setIsJobMonitorOpen(false)}
      />
    </div>
  );
}
