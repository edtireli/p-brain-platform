import { useState, useEffect, useMemo } from 'react';
import { Folder, FolderOpen, FileCode, TreeStructure, FloppyDisk, ArrowCounterClockwise, CaretRight, CaretDown, Warning, CheckCircle } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'framer-motion';
import type { FolderStructureConfig as FolderConfig, Project } from '@/types';
import { DEFAULT_FOLDER_STRUCTURE } from '@/types';
import { engine } from '@/lib/engine';
import type { FolderStructurePreviewSubject } from '@/types';

interface PatternValidation {
  isValid: boolean;
  warnings: string[];
  suggestions: string[];
}

function validateGlobPattern(pattern: string, patternType: 'file' | 'folder'): PatternValidation {
  const warnings: string[] = [];
  const suggestions: string[] = [];
  let isValid = true;

  if (!pattern || pattern.trim() === '') {
    return { isValid: false, warnings: ['Pattern cannot be empty'], suggestions: ['Enter a valid pattern'] };
  }

  const trimmed = pattern.trim();

  if (trimmed.includes('**') && !trimmed.includes('/')) {
    warnings.push('Double asterisk (**) is typically used with paths');
    suggestions.push('Use single * for simple wildcards, or **/pattern for recursive matching');
  }

  if (/\*{3,}/.test(trimmed)) {
    warnings.push('Three or more consecutive asterisks are invalid');
    suggestions.push('Use * for single-level or ** for recursive matching');
    isValid = false;
  }

  const brackets = trimmed.match(/\[|\]/g) || [];
  const openBrackets = brackets.filter(b => b === '[').length;
  const closeBrackets = brackets.filter(b => b === ']').length;
  if (openBrackets !== closeBrackets) {
    warnings.push('Unmatched brackets in character class');
    suggestions.push('Ensure each [ has a matching ]');
    isValid = false;
  }

  const braces = trimmed.match(/\{|\}/g) || [];
  const openBraces = braces.filter(b => b === '{').length;
  const closeBraces = braces.filter(b => b === '}').length;
  if (openBraces !== closeBraces) {
    if (!trimmed.includes('{subject_id}')) {
      warnings.push('Unmatched braces in pattern');
      suggestions.push('Ensure each { has a matching }');
      isValid = false;
    }
  }

  if (patternType === 'file') {
    if (!trimmed.includes('.')) {
      warnings.push('No file extension specified');
      suggestions.push('Add extension like .nii.gz or .nii');
    }

    if (trimmed.includes('.nii') && !trimmed.endsWith('.nii') && !trimmed.endsWith('.nii.gz')) {
      warnings.push('NIfTI extension may be malformed');
      suggestions.push('Use .nii or .nii.gz as the file extension');
    }

    if (trimmed.startsWith('/')) {
      warnings.push('Pattern should not start with /');
      suggestions.push('Remove leading slash for relative path matching');
    }
  }

  if (patternType === 'folder') {
    if (trimmed.includes('.nii')) {
      warnings.push('Folder pattern should not contain file extensions');
      suggestions.push('Remove file extension from folder pattern');
    }
  }

  if (/[<>:"|?]/.test(trimmed)) {
    warnings.push('Pattern contains invalid path characters');
    suggestions.push('Remove characters: < > : " | ?');
    isValid = false;
  }

  if (trimmed.includes('//')) {
    warnings.push('Double slashes detected in pattern');
    suggestions.push('Use single slashes between path segments');
  }

  if (trimmed.endsWith('/') && patternType === 'file') {
    warnings.push('File pattern should not end with /');
    suggestions.push('Remove trailing slash');
  }

  return { isValid, warnings, suggestions };
}

interface PatternInputProps {
  id: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  patternType: 'file' | 'folder';
  label: string;
  colorIndicator?: string;
  optional?: boolean;
}

function PatternInput({ id, value, onChange, placeholder, patternType, label, colorIndicator, optional }: PatternInputProps) {
  const validation = useMemo(() => {
    if (optional && (!value || String(value).trim() === '')) {
      return { isValid: true, warnings: [], suggestions: [] };
    }
    return validateGlobPattern(value, patternType);
  }, [value, patternType, optional]);
  
  return (
    <div className="space-y-2">
      <Label htmlFor={id} className="text-xs flex items-center gap-2">
        {colorIndicator && <span className={`h-2 w-2 rounded-full ${colorIndicator}`} />}
        {label}
      </Label>
      <div className="relative">
        <Input
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={`mono text-sm pr-8 ${
            !validation.isValid 
              ? 'border-destructive focus-visible:ring-destructive/50' 
              : validation.warnings.length > 0 
                ? 'border-warning focus-visible:ring-warning/50' 
                : ''
          }`}
        />
        <div className="absolute right-2 top-1/2 -translate-y-1/2">
          {!validation.isValid ? (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Warning size={16} className="text-destructive" weight="fill" />
                </TooltipTrigger>
                <TooltipContent side="left" className="max-w-xs">
                  <div className="space-y-1">
                    {validation.warnings.map((w, i) => (
                      <p key={i} className="text-xs text-destructive-foreground">{w}</p>
                    ))}
                    {validation.suggestions.length > 0 && (
                      <div className="pt-1 border-t border-border mt-1">
                        {validation.suggestions.map((s, i) => (
                          <p key={i} className="text-xs text-muted-foreground">💡 {s}</p>
                        ))}
                      </div>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ) : validation.warnings.length > 0 ? (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Warning size={16} className="text-warning" weight="fill" />
                </TooltipTrigger>
                <TooltipContent side="left" className="max-w-xs">
                  <div className="space-y-1">
                    {validation.warnings.map((w, i) => (
                      <p key={i} className="text-xs">{w}</p>
                    ))}
                    {validation.suggestions.length > 0 && (
                      <div className="pt-1 border-t border-border mt-1">
                        {validation.suggestions.map((s, i) => (
                          <p key={i} className="text-xs text-muted-foreground">💡 {s}</p>
                        ))}
                      </div>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ) : value.length > 0 ? (
            <CheckCircle size={16} className="text-success" weight="fill" />
          ) : null}
        </div>
      </div>
      <AnimatePresence>
        {!validation.isValid && validation.warnings.length > 0 && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="text-[10px] text-destructive"
          >
            {validation.warnings[0]}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}

interface FolderStructureConfigProps {
  project: Project;
  onSave: (
    config: FolderConfig,
    pbrainOverrides?: { vfaGlob?: string; irPrefixes?: string; irTi?: string }
  ) => void;
}

const PRESET_STRUCTURES = [
  {
    id: 'pbrain-default',
    name: 'p-brain (Default)',
    description: 'Matches p-brain NIfTI folder + typical filename conventions',
    config: {
      subjectFolderPattern: '{subject_id}',
      // Comma-separated fallbacks (first match wins)
      t1Pattern: 'WIPcs_T1W_3D_TFE_32channel.nii*,*T1*.nii*',
      t2Pattern:
        'WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,ax*WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,WIPAxT2TSEmatrix.nii*,*T2*.nii*',
      flairPattern: 'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,ax*WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,*FLAIR*.nii*',
      dcePattern: 'WIPDelRec-hperf120long.nii*,WIPhperf120long.nii*,*DCE*.nii*',
      diffusionPattern: 'Reg-DWInySENSE*.nii*,isoDWIb-1000*.nii*,WIPDTI_RSI_*.nii*,WIPDWI_RSI_*.nii*,*DTI*.nii*',
      niftiSubfolder: 'NIfTI',
      useNestedStructure: true,
      aiModelsPath: DEFAULT_FOLDER_STRUCTURE.aiModelsPath,
    },
  },
  {
    id: 'bids',
    name: 'BIDS Standard',
    description: 'Brain Imaging Data Structure format',
    config: {
      subjectFolderPattern: 'sub-{subject_id}',
      t1Pattern: 'anat/*T1w*.nii.gz',
      t2Pattern: 'anat/*T2w*.nii.gz',
      flairPattern: 'anat/*FLAIR*.nii.gz',
      dcePattern: 'perf/*dce*.nii.gz',
      diffusionPattern: 'dwi/*dwi*.nii.gz',
      niftiSubfolder: '',
      useNestedStructure: true,
      aiModelsPath: DEFAULT_FOLDER_STRUCTURE.aiModelsPath,
    },
  },
  {
    id: 'flat',
    name: 'Flat Structure',
    description: 'All NIfTI files in subject folder',
    config: {
      subjectFolderPattern: '{subject_id}',
      t1Pattern: '*T1*.nii.gz',
      dcePattern: '*DCE*.nii.gz',
      diffusionPattern: '*DTI*.nii.gz',
      niftiSubfolder: '',
      useNestedStructure: false,
      aiModelsPath: DEFAULT_FOLDER_STRUCTURE.aiModelsPath,
    },
  },
  {
    id: 'nested',
    name: 'Nested by Modality',
    description: 'Organized into modality subfolders',
    config: {
      subjectFolderPattern: '{subject_id}',
      t1Pattern: '*.nii.gz',
      dcePattern: '*.nii.gz',
      diffusionPattern: '*.nii.gz',
      niftiSubfolder: 'nifti',
      useNestedStructure: true,
      aiModelsPath: DEFAULT_FOLDER_STRUCTURE.aiModelsPath,
    },
  },
  {
    id: 'custom',
    name: 'Custom',
    description: 'Define your own structure',
    config: DEFAULT_FOLDER_STRUCTURE,
  },
];

interface FolderTreeNodeProps {
  name: string;
  isFolder: boolean;
  isExpanded?: boolean;
  level: number;
  isHighlighted?: boolean;
  highlightColor?: 'success' | 'accent' | 'warning';
  children?: React.ReactNode;
  onToggle?: () => void;
}

const getHighlightClasses = (color?: 'success' | 'accent' | 'warning') => {
  switch (color) {
    case 'success':
      return 'bg-success/10 border border-success/30';
    case 'accent':
      return 'bg-accent/10 border border-accent/30';
    case 'warning':
      return 'bg-warning/10 border border-warning/30';
    default:
      return '';
  }
};

const getBadgeClasses = (color?: 'success' | 'accent' | 'warning') => {
  switch (color) {
    case 'success':
      return 'border-success/50 text-success';
    case 'accent':
      return 'border-accent/50 text-accent';
    case 'warning':
      return 'border-warning/50 text-warning';
    default:
      return '';
  }
};

const getBadgeLabel = (color?: 'success' | 'accent' | 'warning') => {
  switch (color) {
    case 'success':
      return 'T1';
    case 'accent':
      return 'DCE';
    case 'warning':
      return 'DTI';
    default:
      return '';
  }
};

function FolderTreeNode({ 
  name, 
  isFolder, 
  isExpanded = false, 
  level, 
  isHighlighted,
  highlightColor,
  children,
  onToggle,
}: FolderTreeNodeProps) {
  const hasChildren = !!children;
  
  return (
    <div className="select-none">
      <div 
        className={`flex items-center gap-1.5 py-1 px-2 rounded-md cursor-pointer transition-colors ${
          isHighlighted 
            ? getHighlightClasses(highlightColor)
            : 'hover:bg-muted/50'
        }`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={onToggle}
      >
        {hasChildren && isFolder ? (
          isExpanded ? (
            <CaretDown size={12} className="text-muted-foreground" />
          ) : (
            <CaretRight size={12} className="text-muted-foreground" />
          )
        ) : (
          <span className="w-3" />
        )}
        {isFolder ? (
          isExpanded ? (
            <FolderOpen size={16} weight="fill" className="text-accent" />
          ) : (
            <Folder size={16} weight="fill" className="text-accent" />
          )
        ) : (
          <FileCode size={16} className="text-muted-foreground" />
        )}
        <span className={`mono text-xs ${isHighlighted ? 'font-medium' : ''}`}>
          {name}
        </span>
        {isHighlighted && highlightColor && (
          <Badge 
            variant="outline" 
            className={`text-[10px] px-1.5 py-0 ml-2 ${getBadgeClasses(highlightColor)}`}
          >
            {getBadgeLabel(highlightColor)}
          </Badge>
        )}
      </div>
      <AnimatePresence>
        {isExpanded && children && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function FolderStructureConfig({ project, onSave }: FolderStructureConfigProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [tab, setTab] = useState<'presets' | 'patterns' | 'preview'>('presets');
  const [config, setConfig] = useState<FolderConfig>(
    { ...DEFAULT_FOLDER_STRUCTURE, ...(project.config.folderStructure || {}) }
  );
  const [pbrainVfaGlob, setPbrainVfaGlob] = useState<string>('*VFA*.nii*');
  const [pbrainIrPrefixes, setPbrainIrPrefixes] = useState<string>('WIPTI_,WIPDelRec-TI_');
  const [pbrainIrTi, setPbrainIrTi] = useState<string>('00120,00300,00600,01000,02000,04000,10000');
  const [selectedPreset, setSelectedPreset] = useState<string>('custom');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set(['root', 'subject', 'nifti', 'anat', 'perf', 'dwi'])
  );

  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewSubjects, setPreviewSubjects] = useState<FolderStructurePreviewSubject[]>([]);

  const validationState = useMemo(() => {
    const subjectValidation = validateGlobPattern(config.subjectFolderPattern, 'folder');
    const t1Validation = config.t1Pattern?.trim() ? validateGlobPattern(config.t1Pattern, 'file') : { isValid: true, warnings: [], suggestions: [] };
    const t2Validation = (config as any).t2Pattern?.trim() ? validateGlobPattern((config as any).t2Pattern, 'file') : { isValid: true, warnings: [], suggestions: [] };
    const dceValidation = config.dcePattern?.trim() ? validateGlobPattern(config.dcePattern, 'file') : { isValid: true, warnings: [], suggestions: [] };
    const diffusionValidation = config.diffusionPattern?.trim() ? validateGlobPattern(config.diffusionPattern, 'file') : { isValid: true, warnings: [], suggestions: [] };
    
    const allValid = subjectValidation.isValid && t1Validation.isValid && t2Validation.isValid &&
                     dceValidation.isValid && diffusionValidation.isValid;
    
    const totalWarnings = subjectValidation.warnings.length + t1Validation.warnings.length + t2Validation.warnings.length +
                          dceValidation.warnings.length + diffusionValidation.warnings.length;
    
    return { allValid, totalWarnings };
  }, [config]);

  useEffect(() => {
    const pb: any = (project as any)?.config?.pbrain || {};
    const vfa = String(pb?.vfaGlob ?? '*VFA*.nii*').trim();
    setPbrainVfaGlob(vfa || '*VFA*.nii*');
    const pref = String(pb?.irPrefixes ?? 'WIPTI_,WIPDelRec-TI_').trim();
    setPbrainIrPrefixes(pref || 'WIPTI_,WIPDelRec-TI_');
    const ti = String(pb?.irTi ?? '00120,00300,00600,01000,02000,04000,10000').trim();
    setPbrainIrTi(ti || '00120,00300,00600,01000,02000,04000,10000');
  }, [project?.id, project?.updatedAt]);

  useEffect(() => {
    if (!isOpen) return;
    if (tab !== 'preview') return;
    if (!project?.id) return;

    const t = window.setTimeout(() => {
      setPreviewLoading(true);
      setPreviewError(null);
      void (async () => {
        try {
          const res = await (engine as any).previewFolderStructure(project.id, config);
          const subjects = (res?.subjects || []) as FolderStructurePreviewSubject[];
          setPreviewSubjects(subjects);
          if (Array.isArray(res?.errors) && res.errors.length > 0) {
            setPreviewError(res.errors[0]);
          }
        } catch (e) {
          setPreviewSubjects([]);
          setPreviewError(e instanceof Error ? e.message : String(e));
        } finally {
          setPreviewLoading(false);
        }
      })();
    }, 250);

    return () => window.clearTimeout(t);
  }, [isOpen, tab, project?.id, config]);

  useEffect(() => {
    const comparable = { ...config, aiModelsPath: '' };
    const matchingPreset = PRESET_STRUCTURES.find(
      p => p.id !== 'custom' && 
        JSON.stringify({ ...(p.config as any), aiModelsPath: '' }) === JSON.stringify(comparable)
    );
    setSelectedPreset(matchingPreset?.id || 'custom');
  }, [config]);

  const handlePresetChange = (presetId: string) => {
    const preset = PRESET_STRUCTURES.find(p => p.id === presetId);
    if (preset) {
      setConfig(prev => ({
        ...preset.config,
        aiModelsPath: prev.aiModelsPath || DEFAULT_FOLDER_STRUCTURE.aiModelsPath,
      }));
      setSelectedPreset(presetId);
    }
  };

  const handleSave = () => {
    if (!validationState.allValid) {
      toast.error('Please fix invalid patterns before saving');
      return;
    }
    if (validationState.totalWarnings > 0) {
      toast.warning('Configuration saved with warnings', {
        description: `${validationState.totalWarnings} pattern warning(s) detected`
      });
    } else {
      toast.success('Folder structure configuration saved');
    }
    onSave(config, {
      vfaGlob: String(pbrainVfaGlob || '').trim(),
      irPrefixes: String(pbrainIrPrefixes || '').trim(),
      irTi: String(pbrainIrTi || '').trim(),
    });
    setIsOpen(false);
  };

  const handleReset = () => {
    setConfig({ ...DEFAULT_FOLDER_STRUCTURE, ...(project.config.folderStructure || {}) });
    toast.info('Configuration reset');
  };


  const toggleFolder = (folderId: string) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      if (next.has(folderId)) {
        next.delete(folderId);
      } else {
        next.add(folderId);
      }
      return next;
    });
  };

  const renderFolderPreview = () => {
    const subjectName = config.subjectFolderPattern.replace('{subject_id}', 'subject_001');
    const hasNiftiSubfolder = config.niftiSubfolder.length > 0;
    
    if (config.useNestedStructure) {
      if (selectedPreset === 'bids') {
        return (
          <>
            <FolderTreeNode 
              name="project_root" 
              isFolder 
              level={0}
              isExpanded={expandedFolders.has('root')}
              onToggle={() => toggleFolder('root')}
            >
              <FolderTreeNode 
                name={subjectName} 
                isFolder 
                level={1}
                isExpanded={expandedFolders.has('subject')}
                onToggle={() => toggleFolder('subject')}
              >
                <FolderTreeNode 
                  name="anat" 
                  isFolder 
                  level={2}
                  isExpanded={expandedFolders.has('anat')}
                  onToggle={() => toggleFolder('anat')}
                >
                  <FolderTreeNode 
                    name={`${subjectName}_T1w.nii.gz`}
                    isFolder={false}
                    level={3}
                    isHighlighted
                    highlightColor="success"
                  />
                </FolderTreeNode>
                <FolderTreeNode 
                  name="perf" 
                  isFolder 
                  level={2}
                  isExpanded={expandedFolders.has('perf')}
                  onToggle={() => toggleFolder('perf')}
                >
                  <FolderTreeNode 
                    name={`${subjectName}_dce.nii.gz`}
                    isFolder={false}
                    level={3}
                    isHighlighted
                    highlightColor="accent"
                  />
                </FolderTreeNode>
                <FolderTreeNode 
                  name="dwi" 
                  isFolder 
                  level={2}
                  isExpanded={expandedFolders.has('dwi')}
                  onToggle={() => toggleFolder('dwi')}
                >
                  <FolderTreeNode 
                    name={`${subjectName}_dwi.nii.gz`}
                    isFolder={false}
                    level={3}
                    isHighlighted
                    highlightColor="warning"
                  />
                </FolderTreeNode>
              </FolderTreeNode>
            </FolderTreeNode>
          </>
        );
      }
      
      return (
        <>
          <FolderTreeNode 
            name="project_root" 
            isFolder 
            level={0}
            isExpanded={expandedFolders.has('root')}
            onToggle={() => toggleFolder('root')}
          >
            <FolderTreeNode 
              name={subjectName} 
              isFolder 
              level={1}
              isExpanded={expandedFolders.has('subject')}
              onToggle={() => toggleFolder('subject')}
            >
              {hasNiftiSubfolder ? (
                <FolderTreeNode 
                  name={config.niftiSubfolder} 
                  isFolder 
                  level={2}
                  isExpanded={expandedFolders.has('nifti')}
                  onToggle={() => toggleFolder('nifti')}
                >
                  <FolderTreeNode 
                    name={config.t1Pattern.replace('*', subjectName + '_')}
                    isFolder={false}
                    level={3}
                    isHighlighted
                    highlightColor="success"
                  />
                  <FolderTreeNode 
                    name={config.dcePattern.replace('*', subjectName + '_')}
                    isFolder={false}
                    level={3}
                    isHighlighted
                    highlightColor="accent"
                  />
                  <FolderTreeNode 
                    name={config.diffusionPattern.replace('*', subjectName + '_')}
                    isFolder={false}
                    level={3}
                    isHighlighted
                    highlightColor="warning"
                  />
                </FolderTreeNode>
              ) : (
                <>
                  <FolderTreeNode 
                    name={config.t1Pattern.replace('*', subjectName + '_')}
                    isFolder={false}
                    level={2}
                    isHighlighted
                    highlightColor="success"
                  />
                  <FolderTreeNode 
                    name={config.dcePattern.replace('*', subjectName + '_')}
                    isFolder={false}
                    level={2}
                    isHighlighted
                    highlightColor="accent"
                  />
                  <FolderTreeNode 
                    name={config.diffusionPattern.replace('*', subjectName + '_')}
                    isFolder={false}
                    level={2}
                    isHighlighted
                    highlightColor="warning"
                  />
                </>
              )}
            </FolderTreeNode>
          </FolderTreeNode>
        </>
      );
    }
    
    return (
      <>
        <FolderTreeNode 
          name="project_root" 
          isFolder 
          level={0}
          isExpanded={expandedFolders.has('root')}
          onToggle={() => toggleFolder('root')}
        >
          <FolderTreeNode 
            name={subjectName} 
            isFolder 
            level={1}
            isExpanded={expandedFolders.has('subject')}
            onToggle={() => toggleFolder('subject')}
          >
            <FolderTreeNode 
              name={config.t1Pattern.replace('*', subjectName + '_')}
              isFolder={false}
              level={2}
              isHighlighted
              highlightColor="success"
            />
            <FolderTreeNode 
              name={config.dcePattern.replace('*', subjectName + '_')}
              isFolder={false}
              level={2}
              isHighlighted
              highlightColor="accent"
            />
            <FolderTreeNode 
              name={config.diffusionPattern.replace('*', subjectName + '_')}
              isFolder={false}
              level={2}
              isHighlighted
              highlightColor="warning"
            />
          </FolderTreeNode>
        </FolderTreeNode>
      </>
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <TreeStructure size={18} />
          Configure
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TreeStructure size={22} className="text-accent" />
            Folder Structure Configuration
          </DialogTitle>
          <DialogDescription>
            Define how subject folders are organized and where imaging files are located.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 min-h-0 overflow-hidden">
          <Tabs value={tab} onValueChange={(v) => setTab(v as any)} className="h-full min-h-0 flex flex-col">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="presets">Presets</TabsTrigger>
              <TabsTrigger value="patterns">File Patterns</TabsTrigger>
              <TabsTrigger value="preview">Preview</TabsTrigger>
            </TabsList>

            <div className="flex-1 min-h-0 overflow-hidden mt-4">
              <TabsContent value="presets" className="mt-0 h-full min-h-0">
                <ScrollArea className="h-full pr-4">
                  <div className="grid gap-4 sm:grid-cols-2 pb-4">
                    {PRESET_STRUCTURES.map((preset) => (
                      <Card 
                        key={preset.id}
                        className={`cursor-pointer transition-all hover:shadow-md ${
                          selectedPreset === preset.id 
                            ? 'border-accent ring-2 ring-accent/20' 
                            : 'hover:border-muted-foreground/30'
                        }`}
                        onClick={() => handlePresetChange(preset.id)}
                      >
                        <CardHeader className="pb-2">
                          <CardTitle className="flex items-center justify-between text-sm font-medium">
                            <span>{preset.name}</span>
                            {selectedPreset === preset.id && (
                              <Badge className="bg-accent text-accent-foreground text-[10px]">
                                Selected
                              </Badge>
                            )}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-xs text-muted-foreground mb-3">
                            {preset.description}
                          </p>
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 text-xs">
                              <Folder size={12} className="text-accent" />
                              <span className="mono text-muted-foreground">
                                {preset.config.subjectFolderPattern}
                              </span>
                            </div>
                            {preset.config.niftiSubfolder && (
                              <div className="flex items-center gap-2 text-xs">
                                <FolderOpen size={12} className="text-accent" />
                                <span className="mono text-muted-foreground">
                                  /{preset.config.niftiSubfolder}/
                                </span>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="patterns" className="mt-0 h-full min-h-0">
                <ScrollArea className="h-full pr-4">
                  <div className="space-y-6 pb-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Folder size={16} className="text-accent" />
                        Subject Folder Pattern
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <PatternInput
                          id="subjectPattern"
                          value={config.subjectFolderPattern}
                          onChange={(value) => setConfig(prev => ({ 
                            ...prev, 
                            subjectFolderPattern: value 
                          }))}
                          placeholder="sub-{subject_id}"
                          patternType="folder"
                          label="Pattern for subject folder names"
                        />
                        <p className="text-[10px] text-muted-foreground">
                          Use <code className="mono bg-muted px-1 rounded">{'{subject_id}'}</code> as a placeholder for the subject identifier.
                        </p>
                      </div>

                      <div className="flex items-center justify-between rounded-lg border border-border bg-card p-3">
                        <div className="space-y-0.5">
                          <Label htmlFor="nestedStructure" className="text-sm font-normal">
                            Use nested folder structure
                          </Label>
                          <p className="text-xs text-muted-foreground">
                            Enable if files are in subfolders within each subject
                          </p>
                        </div>
                        <Switch 
                          id="nestedStructure" 
                          checked={config.useNestedStructure}
                          onCheckedChange={(checked) => setConfig(prev => ({ 
                            ...prev, 
                            useNestedStructure: checked 
                          }))}
                        />
                      </div>

                      {config.useNestedStructure && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="space-y-2"
                        >
                          <Label htmlFor="niftiSubfolder" className="text-xs">
                            NIfTI subfolder name (leave empty for modality subfolders)
                          </Label>
                          <Input
                            id="niftiSubfolder"
                            value={config.niftiSubfolder}
                            onChange={(e) => setConfig(prev => ({ 
                              ...prev, 
                              niftiSubfolder: e.target.value 
                            }))}
                            placeholder="nifti"
                            className="mono text-sm"
                          />
                        </motion.div>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <FolderOpen size={16} className="text-accent" />
                        AI Models Folder
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="space-y-2">
                        <Label htmlFor="aiModelsPath" className="text-xs">
                          Local folder containing p-brain AI model files
                        </Label>
                        <Input
                          id="aiModelsPath"
                          value={config.aiModelsPath}
                          onChange={(e) => setConfig(prev => ({
                            ...prev,
                            aiModelsPath: e.target.value,
                          }))}
                          placeholder={DEFAULT_FOLDER_STRUCTURE.aiModelsPath}
                          className="mono text-sm"
                        />
                        <p className="text-[10px] text-muted-foreground">
                          If you don’t have the model files, download them from{' '}
                          <a
                            href="https://zenodo.org/records/15697443"
                            target="_blank"
                            rel="noreferrer"
                            className="underline"
                          >
                            https://zenodo.org/records/15697443
                          </a>
                          .
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <FileCode size={16} className="text-accent" />
                        File Matching Patterns
                        {validationState.totalWarnings > 0 && validationState.allValid && (
                          <Badge variant="outline" className="ml-auto border-warning text-warning text-[10px]">
                            {validationState.totalWarnings} warning{validationState.totalWarnings > 1 ? 's' : ''}
                          </Badge>
                        )}
                        {!validationState.allValid && (
                          <Badge variant="outline" className="ml-auto border-destructive text-destructive text-[10px]">
                            Invalid patterns
                          </Badge>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 sm:grid-cols-3">
                        <PatternInput
                          id="t1Pattern"
                          value={config.t1Pattern}
                          onChange={(value) => setConfig(prev => ({ ...prev, t1Pattern: value }))}
                          placeholder="*T1*.nii.gz"
                          patternType="file"
                          label="T1/Anatomical Pattern"
                          colorIndicator="bg-success"
                          optional
                        />

                        <PatternInput
                          id="t2Pattern"
                          value={(config as any).t2Pattern || ''}
                          onChange={(value) => setConfig(prev => ({ ...(prev as any), t2Pattern: value }))}
                          placeholder="*T2*.nii.gz"
                          patternType="file"
                          label="T2 Pattern"
                          optional
                        />

                        <PatternInput
                          id="dcePattern"
                          value={config.dcePattern}
                          onChange={(value) => setConfig(prev => ({ ...prev, dcePattern: value }))}
                          placeholder="*DCE*.nii.gz"
                          patternType="file"
                          label="DCE Pattern"
                          colorIndicator="bg-accent"
                          optional
                        />

                        <PatternInput
                          id="diffusionPattern"
                          value={config.diffusionPattern}
                          onChange={(value) => setConfig(prev => ({ ...prev, diffusionPattern: value }))}
                          placeholder="*DTI*.nii.gz"
                          patternType="file"
                          label="Diffusion Pattern"
                          colorIndicator="bg-warning"
                          optional
                        />
                      </div>

                      <div className="rounded-lg bg-muted/50 p-3">
                        <p className="text-xs text-muted-foreground">
                          <strong>Pattern syntax:</strong> Use <code className="mono bg-background px-1 rounded">*</code> as 
                          a wildcard. For example, <code className="mono bg-background px-1 rounded">*T1*.nii.gz</code> matches 
                          any file containing "T1" with a .nii.gz extension. Use <code className="mono bg-background px-1 rounded">**/</code> for 
                          recursive matching.
                        </p>
                      </div>

                      <div className="border-t border-border pt-4">
                        <div className="flex items-center gap-2 mb-3">
                          <FileCode size={14} className="text-accent" />
                          <span className="text-sm font-medium">T1/M0 Series Naming (Optional)</span>
                        </div>
                        <div className="grid gap-4 sm:grid-cols-3">
                          <div className="space-y-2">
                            <Label htmlFor="irPrefixes" className="text-xs">TI series prefixes (comma-separated)</Label>
                            <Input
                              id="irPrefixes"
                              value={pbrainIrPrefixes}
                              onChange={(e) => setPbrainIrPrefixes(e.target.value)}
                              placeholder="WIPTI_,WIPDelRec-TI_"
                              className="mono text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="irTi" className="text-xs">TI values (comma-separated, ms codes)</Label>
                            <Input
                              id="irTi"
                              value={pbrainIrTi}
                              onChange={(e) => setPbrainIrTi(e.target.value)}
                              placeholder="00120,00300,00600,01000,02000,04000,10000"
                              className="mono text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="vfaGlob" className="text-xs">VFA series glob(s) (comma-separated)</Label>
                            <Input
                              id="vfaGlob"
                              value={pbrainVfaGlob}
                              onChange={(e) => setPbrainVfaGlob(e.target.value)}
                              placeholder="*VFA*.nii*"
                              className="mono text-sm"
                            />
                          </div>
                        </div>
                        <div className="rounded-lg bg-muted/50 p-3 mt-4">
                          <p className="text-xs text-muted-foreground">
                            These map to p-brain discovery env vars: <code className="mono bg-background px-1 rounded">P_BRAIN_IR_PREFIXES</code>,{' '}
                            <code className="mono bg-background px-1 rounded">P_BRAIN_IR_TI</code>,{' '}
                            <code className="mono bg-background px-1 rounded">P_BRAIN_VFA_GLOB</code>.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="preview" className="mt-0 h-full min-h-0">
                <ScrollArea className="h-full pr-4">
                  <div className="grid gap-4 lg:grid-cols-2 pb-4">
                  <Card className="h-full">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium">
                        Expected Folder Structure
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ScrollArea className="h-[300px] rounded-lg border border-border bg-background p-3">
                        {renderFolderPreview()}
                      </ScrollArea>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium">
                        Configuration Summary
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-3">
                        <div className="flex items-center justify-between py-2 border-b border-border">
                          <span className="text-xs text-muted-foreground">Preset</span>
                          <Badge variant="secondary">
                            {PRESET_STRUCTURES.find(p => p.id === selectedPreset)?.name || 'Custom'}
                          </Badge>
                        </div>
                        
                        <div className="flex items-center justify-between py-2 border-b border-border">
                          <span className="text-xs text-muted-foreground">Subject folder</span>
                          <span className="mono text-xs">{config.subjectFolderPattern}</span>
                        </div>
                        
                        <div className="flex items-center justify-between py-2 border-b border-border">
                          <span className="text-xs text-muted-foreground">Structure type</span>
                          <span className="text-xs">
                            {config.useNestedStructure ? 'Nested' : 'Flat'}
                          </span>
                        </div>
                        
                        {config.useNestedStructure && config.niftiSubfolder && (
                          <div className="flex items-center justify-between py-2 border-b border-border">
                            <span className="text-xs text-muted-foreground">NIfTI subfolder</span>
                            <span className="mono text-xs">{config.niftiSubfolder}</span>
                          </div>
                        )}
                      </div>

                      <div className="space-y-2">
                        <p className="text-xs font-medium">File patterns:</p>
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1.5">
                            <span className="h-2 w-2 rounded-full bg-success" />
                            <span className="text-xs text-muted-foreground">T1:</span>
                            <span className="mono text-xs">{config.t1Pattern}</span>
                          </div>
                          {(config as any).t2Pattern ? (
                            <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1.5">
                              <span className="h-2 w-2 rounded-full bg-success" />
                              <span className="text-xs text-muted-foreground">T2:</span>
                              <span className="mono text-xs">{(config as any).t2Pattern}</span>
                            </div>
                          ) : null}
                          {(config as any).flairPattern ? (
                            <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1.5">
                              <span className="h-2 w-2 rounded-full bg-success" />
                              <span className="text-xs text-muted-foreground">FLAIR:</span>
                              <span className="mono text-xs">{(config as any).flairPattern}</span>
                            </div>
                          ) : null}
                          <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1.5">
                            <span className="h-2 w-2 rounded-full bg-accent" />
                            <span className="text-xs text-muted-foreground">DCE:</span>
                            <span className="mono text-xs">{config.dcePattern}</span>
                          </div>
                          <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2 py-1.5">
                            <span className="h-2 w-2 rounded-full bg-warning" />
                            <span className="text-xs text-muted-foreground">Diffusion:</span>
                            <span className="mono text-xs">{config.diffusionPattern}</span>
                          </div>
                        </div>
                      </div>

                      <div className="border-t border-border pt-4">
                        <div className="flex items-center justify-between">
                          <p className="text-xs font-medium">Per-subject matches</p>
                          <span className="text-[10px] text-muted-foreground">
                            {previewLoading ? 'Scanning…' : `${previewSubjects.length} subject${previewSubjects.length === 1 ? '' : 's'}`}
                          </span>
                        </div>

                        {previewError ? (
                          <p className="text-[10px] text-destructive mt-2">{previewError}</p>
                        ) : null}

                        <div className="mt-3 rounded-lg border border-border bg-background">
                          <ScrollArea className="h-[280px]">
                            <div className="min-w-[720px]">
                              <table className="w-full text-xs">
                                <thead className="sticky top-0 bg-background border-b border-border">
                                  <tr>
                                    <th className="text-left font-medium px-3 py-2">Subject</th>
                                    <th className="text-left font-medium px-3 py-2">T1</th>
                                    <th className="text-left font-medium px-3 py-2">DCE</th>
                                    <th className="text-left font-medium px-3 py-2">Diffusion</th>
                                    <th className="text-left font-medium px-3 py-2">T2</th>
                                    <th className="text-left font-medium px-3 py-2">FLAIR</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {previewSubjects.map((s) => {
                                    const m: any = (s as any).matches || {};
                                    const cell = (v: any) => {
                                      const txt = typeof v === 'string' && v.trim() ? v : '';
                                      return (
                                        <span className={txt ? 'mono' : 'text-muted-foreground'} title={txt || ''}>
                                          {txt || '—'}
                                        </span>
                                      );
                                    };

                                    return (
                                      <tr key={s.name} className="border-b border-border last:border-b-0">
                                        <td className="px-3 py-2 mono">{s.name}</td>
                                        <td className="px-3 py-2">{cell(m.t1)}</td>
                                        <td className="px-3 py-2">{cell(m.dce)}</td>
                                        <td className="px-3 py-2">{cell(m.diffusion)}</td>
                                        <td className="px-3 py-2">{cell(m.t2)}</td>
                                        <td className="px-3 py-2">{cell(m.flair)}</td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </ScrollArea>
                        </div>

                        <p className="text-[10px] text-muted-foreground mt-2">
                          Matches show the first file found per pattern (comma-separated fallbacks supported).
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                  </div>
                </ScrollArea>
              </TabsContent>
            </div>
          </Tabs>
        </div>

        <div className="flex justify-between pt-4 border-t border-border mt-4">
          <div className="flex gap-3">
            <Button variant="ghost" onClick={handleReset} className="gap-2">
              <ArrowCounterClockwise size={16} />
              Reset
            </Button>
          </div>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>
                    <Button 
                      onClick={handleSave} 
                      className="gap-2"
                      disabled={!validationState.allValid}
                    >
                      <FloppyDisk size={16} />
                      Save Configuration
                      {validationState.totalWarnings > 0 && validationState.allValid && (
                        <Warning size={14} className="text-warning" weight="fill" />
                      )}
                    </Button>
                  </span>
                </TooltipTrigger>
                {!validationState.allValid && (
                  <TooltipContent>
                    <p className="text-xs">Fix invalid patterns before saving</p>
                  </TooltipContent>
                )}
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
