import { useEffect, useMemo, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { VolumeViewer } from '@/components/VolumeViewer';
import { mockEngine, isBackendEngine } from '@/lib/mock-engine';
import { getBackendBaseUrl } from '@/lib/backend-engine';
import type { MapVolume, Subject } from '@/types';

interface MapsViewProps {
  subjectId: string;
}

export function MapsView({ subjectId }: MapsViewProps) {
  const [subject, setSubject] = useState<Subject | null>(null);
  const [maps, setMaps] = useState<MapVolume[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const selected = useMemo(() => maps.find(m => m.id === selectedId) ?? null, [maps, selectedId]);

  const [ensuringMaps, setEnsuringMaps] = useState(false);
  const [ensureMsg, setEnsureMsg] = useState<string>('');
  const [ensureOnce, setEnsureOnce] = useState(false);

  const [montages, setMontages] = useState<Array<{ id: string; name: string; path: string }>>([]);
  const [selectedMontageId, setSelectedMontageId] = useState<string>('');
  const selectedMontage = useMemo(
    () => montages.find(m => m.id === selectedMontageId) ?? null,
    [montages, selectedMontageId]
  );

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await mockEngine.getSubject(subjectId);
        if (cancelled) return;
        setSubject(s ?? null);

        const list = await mockEngine.getMapVolumes(subjectId);
        if (cancelled) return;
        setMaps(list);
        setSelectedId(prev => {
          if (prev && list.some(m => m.id === prev)) return prev;
          return list[0]?.id ?? '';
        });
      } catch (err) {
        console.error('Failed to load map volumes:', err);
        if (!cancelled) {
          setMaps([]);
          setSelectedId('');
        }
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [subjectId]);

  useEffect(() => {
    if (!isBackendEngine) return;
    if (ensureOnce) return;
    if (!subject) return;
    if (maps.length > 0) return;

    // If modelling hasn't been run yet (or failed), auto-trigger the pipeline once.
    const modelling = (subject.stageStatuses as any)?.modelling;
    if (modelling === 'done') return;

    setEnsureOnce(true);
    setEnsuringMaps(true);
    setEnsureMsg('Running p-brain to generate maps…');
    mockEngine
      .ensureSubjectArtifacts(subjectId, 'maps')
      .then((res: any) => {
        setEnsureMsg(res?.reason || 'Started');
      })
      .catch((e: any) => {
        setEnsureMsg(String(e?.message || e || 'Failed to start pipeline'));
      })
      .finally(() => setEnsuringMaps(false));
  }, [subjectId, subject, maps.length, ensureOnce]);

  useEffect(() => {
    if (!isBackendEngine) return;
    if (!subject) return;
    if (maps.length > 0) return;

    // While modelling is running, poll for new map outputs.
    const modelling = (subject.stageStatuses as any)?.modelling;
    if (modelling !== 'running') return;
    const t = window.setInterval(async () => {
      try {
        const list = await mockEngine.getMapVolumes(subjectId);
        setMaps(list);
        const s = await mockEngine.getSubject(subjectId);
        setSubject(s ?? null);
      } catch {
        // ignore
      }
    }, 2500);
    return () => window.clearInterval(t);
  }, [subjectId, subject, maps.length]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const list = await mockEngine.getMontageImages(subjectId);
        if (cancelled) return;
        setMontages(list);
        setSelectedMontageId(prev => {
          if (prev && list.some(m => m.id === prev)) return prev;
          return list[0]?.id ?? '';
        });
      } catch (err) {
        console.error('Failed to load montage images:', err);
        if (!cancelled) {
          setMontages([]);
          setSelectedMontageId('');
        }
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [subjectId]);

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="mb-4 text-lg font-semibold">Parameter Maps</h2>
        {maps.length === 0 ? (
          <div className="space-y-3 rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 p-6 text-sm text-muted-foreground">
            <div>No parameter-map volumes found for this subject.</div>
            {isBackendEngine ? (
              <div className="flex items-center justify-between gap-3">
                <div className="text-xs text-muted-foreground">{ensuringMaps ? ensureMsg : ensureMsg}</div>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      setEnsuringMaps(true);
                      const res = await mockEngine.ensureSubjectArtifacts(subjectId, 'maps');
                      setEnsureMsg(res?.reason || 'Started');
                    } catch (e: any) {
                      setEnsureMsg(String(e?.message || e || 'Failed to start pipeline'));
                    } finally {
                      setEnsuringMaps(false);
                    }
                  }}
                  disabled={ensuringMaps}
                  className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
                >
                  {ensuringMaps ? 'Running…' : 'Run p-brain'}
                </button>
              </div>
            ) : null}
          </div>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {maps.map(map => (
                <button
                  key={map.id}
                  type="button"
                  onClick={() => setSelectedId(map.id)}
                  className={`rounded-lg border p-6 text-left transition-colors ${
                    map.id === selectedId
                      ? 'border-primary/60 bg-primary/5'
                      : 'border-border bg-card hover:bg-muted/40'
                  }`}
                >
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="font-semibold">{map.name}</h3>
                    <Badge variant="secondary">Available</Badge>
                  </div>
                  <div className="mono text-xs text-muted-foreground">{map.unit}</div>
                  <div className="mt-3 text-xs text-muted-foreground">
                    {map.group === 'diffusion' ? 'Diffusion' : 'Modelling'}
                  </div>
                </button>
              ))}
            </div>

            <div className="mt-6 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-base font-semibold">Map Viewer</h3>
                <div className="w-[260px]">
                  <Select value={selectedId} onValueChange={setSelectedId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a map" />
                    </SelectTrigger>
                    <SelectContent>
                      {maps.map(m => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {selected ? <VolumeViewer subjectId={subjectId} path={selected.path} /> : null}
            </div>
          </>
        )}
      </Card>

      <Card className="p-6">
        <h2 className="mb-4 text-lg font-semibold">Generated Montages (p-brain output)</h2>

        {montages.length === 0 ? (
          <div className="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 p-6 text-sm text-muted-foreground">
            No montage PNGs found. Expected under <span className="mono">Images/AI/Montages</span>.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm text-muted-foreground">{montages.length} file(s)</div>
              <div className="w-[320px]">
                <Select value={selectedMontageId} onValueChange={setSelectedMontageId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a montage" />
                  </SelectTrigger>
                  <SelectContent>
                    {montages.map(m => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {selectedMontage ? (
              <div className="overflow-hidden rounded-lg border bg-muted/10">
                <div className="flex items-center justify-between border-b px-4 py-2">
                  <div className="mono text-xs text-muted-foreground">{selectedMontage.name}</div>
                  <Badge variant="secondary">PNG</Badge>
                </div>
                <div className="flex items-center justify-center p-4">
                  <img
                    alt={selectedMontage.name}
                    src={`${getBackendBaseUrl()}/subjects/${encodeURIComponent(subjectId)}/montages/image?path=${encodeURIComponent(
                      selectedMontage.path
                    )}`}
                    className="max-h-[720px] w-full max-w-full rounded-md border bg-background object-contain"
                  />
                </div>
              </div>
            ) : null}
          </div>
        )}
      </Card>
    </div>
  );
}
