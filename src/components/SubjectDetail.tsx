import { useState, useEffect } from 'react';
import { ArrowLeft, Eye, ChartLine, MapTrifold, Table as TableIcon, FileText } from '@phosphor-icons/react';
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
        <div className="mx-auto max-w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="secondary" size="sm" onClick={onBack} className="gap-2">
                <ArrowLeft size={16} />
                Back
              </Button>
              <div>
                <h1 className="text-2xl font-semibold text-foreground">{subject.name}</h1>
                <p className="mono text-xs text-muted-foreground">{subject.sourcePath}</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-sm text-muted-foreground">Pipeline Progress</div>
                <div className="text-lg font-semibold">
                  {completedStages} / {totalStages} stages
                </div>
              </div>
              <div className="h-16 w-16">
                <svg viewBox="0 0 100 100" className="rotate-[-90deg]">
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    className="text-muted"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="10"
                    strokeDasharray={`${progressPercent * 2.827} ${282.7 - progressPercent * 2.827}`}
                    className="text-accent transition-all duration-500"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-full p-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid">
            <TabsTrigger value="overview" className="gap-2">
              <Eye size={18} />
              <span className="hidden sm:inline">Overview</span>
            </TabsTrigger>
            <TabsTrigger value="viewer" className="gap-2">
              <Eye size={18} />
              <span className="hidden sm:inline">Viewer</span>
            </TabsTrigger>
            <TabsTrigger value="curves" className="gap-2">
              <ChartLine size={18} />
              <span className="hidden sm:inline">Curves</span>
            </TabsTrigger>
            <TabsTrigger value="maps" className="gap-2">
              <MapTrifold size={18} />
              <span className="hidden sm:inline">Maps</span>
            </TabsTrigger>
            <TabsTrigger value="tables" className="gap-2">
              <TableIcon size={18} />
              <span className="hidden sm:inline">Tables</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <Card className="p-6">
              <h2 className="mb-4 text-lg font-semibold">Stage Status</h2>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(subject.stageStatuses).map(([stageId, status]) => (
                  <div
                    key={stageId}
                    className="flex items-center justify-between rounded-lg border border-border bg-card p-4"
                  >
                    <span className="text-sm font-medium">
                      {STAGE_NAMES[stageId as keyof typeof STAGE_NAMES]}
                    </span>
                    <Badge
                      variant={
                        status === 'done'
                          ? 'default'
                          : status === 'failed'
                          ? 'destructive'
                          : status === 'running'
                          ? 'secondary'
                          : 'outline'
                      }
                    >
                      {status.replace('_', ' ')}
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>

            <Card className="p-6">
              <h2 className="mb-4 text-lg font-semibold">Data Availability</h2>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-lg border border-border bg-card p-4">
                  <div className="mb-2 text-sm text-muted-foreground">T1 / IR</div>
                  <div className="text-2xl font-semibold">
                    {subject.hasT1 ? '✓ Available' : '✗ Missing'}
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <div className="mb-2 text-sm text-muted-foreground">DCE Series</div>
                  <div className="text-2xl font-semibold">
                    {subject.hasDCE ? '✓ Available' : '✗ Missing'}
                  </div>
                </div>
                <div className="rounded-lg border border-border bg-card p-4">
                  <div className="mb-2 text-sm text-muted-foreground">Diffusion</div>
                  <div className="text-2xl font-semibold">
                    {subject.hasDiffusion ? '✓ Available' : '✗ Missing'}
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
