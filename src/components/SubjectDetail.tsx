import { useState, useEffect } from 'react';
import { ArrowLeft, Eye, ChartLine, MapTrifold, Table as TableIcon, CheckCircle, XCircle, Clock } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { mockEngine } from '@/lib/mock-engine';
import type { Subject } from '@/types';
import { STAGE_NAMES } from '@/types';
import { VolumeViewer } from './VolumeViewer';
import { CurvesView } from './CurvesView';
import { MapsView } from './MapsView';
import { TablesView } from './TablesView';

interface SubjectDetailProps {
  subjectId: string;
  onBack: () => void;
}

export function SubjectDetail({ subjectId, onBack }: SubjectDetailProps) {
  const [subject, setSubject] = useState<Subject | null>(null);

  useEffect(() => {
    loadSubject();

    const unsubscribe = mockEngine.onStatusUpdate(update => {
      if (update.subjectId === subjectId) {
        setSubject(prev =>
          prev
            ? { ...prev, stageStatuses: { ...prev.stageStatuses, [update.stageId]: update.status } }
            : null
        );
      }
    });

    return () => {
      unsubscribe();
    };
  }, [subjectId]);

  const loadSubject = async () => {
    const data = await mockEngine.getSubject(subjectId);
    if (data) setSubject(data);
  };

  if (!subject) {
    return <div className="flex h-screen items-center justify-center">Loading...</div>;
  }

  const completedStages = Object.values(subject.stageStatuses).filter(s => s === 'done').length;
  const totalStages = Object.keys(subject.stageStatuses).length;
  const progressPercent = (completedStages / totalStages) * 100;

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
                <h1 className="text-2xl font-medium tracking-tight">{subject.name}</h1>
                <p className="mono text-xs text-muted-foreground mt-0.5">{subject.sourcePath}</p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                {subject.hasT1 && (
                  <Badge variant="secondary" className="text-xs font-normal px-2 py-1">
                    T1
                  </Badge>
                )}
                {subject.hasDCE && (
                  <Badge variant="secondary" className="text-xs font-normal px-2 py-1">
                    DCE
                  </Badge>
                )}
                {subject.hasDiffusion && (
                  <Badge variant="secondary" className="text-xs font-normal px-2 py-1">
                    DTI
                  </Badge>
                )}
              </div>
              
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-xs text-muted-foreground">Progress</div>
                  <div className="text-sm font-medium tabular-nums">
                    {completedStages} / {totalStages}
                  </div>
                </div>
                <div className="relative h-12 w-12">
                  <svg viewBox="0 0 100 100" className="rotate-[-90deg]">
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="6"
                      className="text-muted opacity-30"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="6"
                      strokeDasharray={`${progressPercent * 2.638} ${263.8 - progressPercent * 2.638}`}
                      className="text-accent transition-all duration-500"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium tabular-nums">
                    {Math.round(progressPercent)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-full p-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="inline-flex h-9 rounded-lg bg-muted p-1 text-muted-foreground">
            <TabsTrigger value="overview" className="gap-2 text-sm font-normal px-4">
              Overview
            </TabsTrigger>
            <TabsTrigger value="viewer" className="gap-2 text-sm font-normal px-4">
              <Eye size={16} />
              Viewer
            </TabsTrigger>
            <TabsTrigger value="curves" className="gap-2 text-sm font-normal px-4">
              <ChartLine size={16} />
              Curves
            </TabsTrigger>
            <TabsTrigger value="maps" className="gap-2 text-sm font-normal px-4">
              <MapTrifold size={16} />
              Maps
            </TabsTrigger>
            <TabsTrigger value="tables" className="gap-2 text-sm font-normal px-4">
              <TableIcon size={16} />
              Tables
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-5">
            <Card className="border-0 shadow-sm">
              <div className="p-5">
                <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground mb-4">
                  Pipeline Status
                </h2>
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {Object.entries(subject.stageStatuses).map(([stageId, status]) => (
                    <div
                      key={stageId}
                      className="flex items-center justify-between rounded-md bg-muted/40 px-3 py-2.5 transition-colors hover:bg-muted/60"
                    >
                      <span className="text-sm">
                        {STAGE_NAMES[stageId as keyof typeof STAGE_NAMES]}
                      </span>
                      <div className="flex items-center gap-1.5">
                        {status === 'done' ? (
                          <CheckCircle size={16} weight="fill" className="text-success" />
                        ) : status === 'failed' ? (
                          <XCircle size={16} weight="fill" className="text-destructive" />
                        ) : status === 'running' ? (
                          <Clock size={16} className="text-accent" />
                        ) : (
                          <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>

            <Card className="border-0 shadow-sm">
              <div className="p-5">
                <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground mb-4">
                  Data Availability
                </h2>
                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-md bg-muted/40 px-4 py-3">
                    <div className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">T1 / IR</div>
                    <div className="flex items-center gap-2">
                      {subject.hasT1 ? (
                        <>
                          <CheckCircle size={18} weight="fill" className="text-success" />
                          <span className="text-sm font-medium">Available</span>
                        </>
                      ) : (
                        <>
                          <XCircle size={18} weight="fill" className="text-muted-foreground" />
                          <span className="text-sm font-medium text-muted-foreground">Missing</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="rounded-md bg-muted/40 px-4 py-3">
                    <div className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">DCE Series</div>
                    <div className="flex items-center gap-2">
                      {subject.hasDCE ? (
                        <>
                          <CheckCircle size={18} weight="fill" className="text-success" />
                          <span className="text-sm font-medium">Available</span>
                        </>
                      ) : (
                        <>
                          <XCircle size={18} weight="fill" className="text-muted-foreground" />
                          <span className="text-sm font-medium text-muted-foreground">Missing</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="rounded-md bg-muted/40 px-4 py-3">
                    <div className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">Diffusion</div>
                    <div className="flex items-center gap-2">
                      {subject.hasDiffusion ? (
                        <>
                          <CheckCircle size={18} weight="fill" className="text-success" />
                          <span className="text-sm font-medium">Available</span>
                        </>
                      ) : (
                        <>
                          <XCircle size={18} weight="fill" className="text-muted-foreground" />
                          <span className="text-sm font-medium text-muted-foreground">Missing</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="viewer">
            <VolumeViewer subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="curves">
            <CurvesView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="maps">
            <MapsView subjectId={subjectId} />
          </TabsContent>

          <TabsContent value="tables">
            <TablesView subjectId={subjectId} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
