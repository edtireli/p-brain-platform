import { useState, useEffect } from 'react';
import { Folder, FolderOpen, FileCode, TreeStructure, Info, FloppyDisk, ArrowCounterClockwise, CaretRight, CaretDown } from '@phosphor-icons/react';
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

interface FolderStructureConfigProps {
  project: Project;
  onSave: (config: FolderConfig) => void;
}

const PRESET_STRUCTURES = [
  {
    id: 'bids',
    name: 'BIDS Standard',
    description: 'Brain Imaging Data Structure format',
    config: {
      subjectFolderPattern: 'sub-{subject_id}',
      t1Pattern: 'anat/*T1w*.nii.gz',
      dcePattern: 'perf/*dce*.nii.gz',
      diffusionPattern: 'dwi/*dwi*.nii.gz',
      niftiSubfolder: '',
      useNestedStructure: true,
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
  const [config, setConfig] = useState<FolderConfig>(
    project.config.folderStructure || DEFAULT_FOLDER_STRUCTURE
  );
  const [selectedPreset, setSelectedPreset] = useState<string>('custom');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set(['root', 'subject', 'nifti', 'anat', 'perf', 'dwi'])
  );

  useEffect(() => {
    const matchingPreset = PRESET_STRUCTURES.find(
      p => p.id !== 'custom' && 
        JSON.stringify(p.config) === JSON.stringify(config)
    );
    setSelectedPreset(matchingPreset?.id || 'custom');
  }, [config]);

  const handlePresetChange = (presetId: string) => {
    const preset = PRESET_STRUCTURES.find(p => p.id === presetId);
    if (preset) {
      setConfig(preset.config);
      setSelectedPreset(presetId);
    }
  };

  const handleSave = () => {
    onSave(config);
    setIsOpen(false);
    toast.success('Folder structure configuration saved');
  };

  const handleReset = () => {
    setConfig(project.config.folderStructure || DEFAULT_FOLDER_STRUCTURE);
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
          Configure Folder Structure
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TreeStructure size={22} className="text-accent" />
            Folder Structure Configuration
          </DialogTitle>
          <DialogDescription>
            Define how subject folders are organized and where imaging files are located.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-hidden">
          <Tabs defaultValue="presets" className="h-full flex flex-col">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="presets">Presets</TabsTrigger>
              <TabsTrigger value="patterns">File Patterns</TabsTrigger>
              <TabsTrigger value="preview">Preview</TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-auto mt-4">
              <TabsContent value="presets" className="mt-0 h-full">
                <div className="grid gap-4 sm:grid-cols-2">
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
              </TabsContent>

              <TabsContent value="patterns" className="mt-0">
                <div className="space-y-6">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium flex items-center gap-2">
                        <Folder size={16} className="text-accent" />
                        Subject Folder Pattern
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="subjectPattern" className="text-xs">
                          Pattern for subject folder names
                        </Label>
                        <div className="flex gap-2">
                          <Input
                            id="subjectPattern"
                            value={config.subjectFolderPattern}
                            onChange={(e) => setConfig(prev => ({ 
                              ...prev, 
                              subjectFolderPattern: e.target.value 
                            }))}
                            placeholder="sub-{subject_id}"
                            className="mono text-sm"
                          />
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button variant="ghost" size="icon" className="shrink-0">
                                  <Info size={16} />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-xs">
                                <p className="text-xs">
                                  Use <code className="mono bg-muted px-1 rounded">{'{subject_id}'}</code> as a 
                                  placeholder for the subject identifier.
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>
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
                        <FileCode size={16} className="text-accent" />
                        File Matching Patterns
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 sm:grid-cols-3">
                        <div className="space-y-2">
                          <Label htmlFor="t1Pattern" className="text-xs flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full bg-success" />
                            T1/Anatomical Pattern
                          </Label>
                          <Input
                            id="t1Pattern"
                            value={config.t1Pattern}
                            onChange={(e) => setConfig(prev => ({ 
                              ...prev, 
                              t1Pattern: e.target.value 
                            }))}
                            placeholder="*T1*.nii.gz"
                            className="mono text-sm"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="dcePattern" className="text-xs flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full bg-accent" />
                            DCE Pattern
                          </Label>
                          <Input
                            id="dcePattern"
                            value={config.dcePattern}
                            onChange={(e) => setConfig(prev => ({ 
                              ...prev, 
                              dcePattern: e.target.value 
                            }))}
                            placeholder="*DCE*.nii.gz"
                            className="mono text-sm"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="diffusionPattern" className="text-xs flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full bg-warning" />
                            Diffusion Pattern
                          </Label>
                          <Input
                            id="diffusionPattern"
                            value={config.diffusionPattern}
                            onChange={(e) => setConfig(prev => ({ 
                              ...prev, 
                              diffusionPattern: e.target.value 
                            }))}
                            placeholder="*DTI*.nii.gz"
                            className="mono text-sm"
                          />
                        </div>
                      </div>

                      <div className="rounded-lg bg-muted/50 p-3">
                        <p className="text-xs text-muted-foreground">
                          <strong>Pattern syntax:</strong> Use <code className="mono bg-background px-1 rounded">*</code> as 
                          a wildcard. For example, <code className="mono bg-background px-1 rounded">*T1*.nii.gz</code> matches 
                          any file containing "T1" with a .nii.gz extension.
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="preview" className="mt-0 h-full">
                <div className="grid gap-4 lg:grid-cols-2 h-full">
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
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>

        <div className="flex justify-between pt-4 border-t border-border mt-4">
          <Button variant="ghost" onClick={handleReset} className="gap-2">
            <ArrowCounterClockwise size={16} />
            Reset
          </Button>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} className="gap-2">
              <FloppyDisk size={16} />
              Save Configuration
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
