import { useEffect, useMemo, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { VolumeViewer } from '@/components/VolumeViewer';
import { engine } from '@/lib/engine';
import type { MapVolume, Subject } from '@/types';

interface MapsViewProps {
  subjectId: string;
}

type ExpectedMap = {
  id: string;
  label: string;
  group: 'modelling' | 'diffusion';
  // candidate basenames (without extension) we accept as fulfilling this slot
  candidates: string[];
};

const EXPECTED_MAPS: ExpectedMap[] = [
  // Modelling
  { id: 'Ki_per_voxel', label: 'Ki (voxel)', group: 'modelling', candidates: ['Ki_per_voxel_tikhonov', 'Ki_per_voxel_patlak', 'Ki_per_voxel_two_compartment'] },
  { id: 'CBF_per_voxel', label: 'CBF (voxel)', group: 'modelling', candidates: ['CBF_per_voxel_tikhonov', 'CBF_per_voxel_patlak', 'CBF_per_voxel_two_compartment'] },
  { id: 'mtt_map', label: 'MTT', group: 'modelling', candidates: ['mtt_map'] },
  { id: 'cth_map', label: 'CTH', group: 'modelling', candidates: ['cth_map'] },
  { id: 'vp_per_voxel', label: 'vp (voxel)', group: 'modelling', candidates: ['vp_per_voxel_tikhonov', 'vp_per_voxel_patlak', 'vp_per_voxel_two_compartment'] },
  { id: 'Ki_map_atlas', label: 'Ki (atlas)', group: 'modelling', candidates: ['Ki_map_atlas_tikhonov', 'Ki_map_atlas_patlak', 'Ki_map_atlas_two_compartment'] },
  { id: 'CBF_atlas', label: 'CBF (atlas)', group: 'modelling', candidates: ['CBF_tikhonov_map_atlas_tikhonov', 'CBF_tikhonov_map_atlas_patlak'] },
  { id: 'MTT_atlas', label: 'MTT (atlas)', group: 'modelling', candidates: ['MTT_tikhonov_map_atlas_tikhonov', 'MTT_tikhonov_map_atlas_patlak'] },
  { id: 'CTH_atlas', label: 'CTH (atlas)', group: 'modelling', candidates: ['CTH_tikhonov_map_atlas_tikhonov', 'CTH_tikhonov_map_atlas_patlak'] },
  // Diffusion
  { id: 'FA', label: 'FA', group: 'diffusion', candidates: ['FA_map', 'fa_map'] },
  { id: 'MD', label: 'MD', group: 'diffusion', candidates: ['md_map', 'MD_map'] },
  { id: 'AD', label: 'AD', group: 'diffusion', candidates: ['ad_map', 'AD_map'] },
  { id: 'RD', label: 'RD', group: 'diffusion', candidates: ['rd_map', 'RD_map'] },
  { id: 'MO', label: 'MO', group: 'diffusion', candidates: ['mo_map', 'MO_map'] },
  { id: 'TensorResidual', label: 'Tensor residual', group: 'diffusion', candidates: ['tensor_residual_map', 'tensor_residual'] },
];

function baseNoExt(filename: string): string {
  return filename.replace(/\.(nii|nii\.gz|png)$/i, '');
}

export function MapsView({ subjectId }: MapsViewProps) {
  const [subject, setSubject] = useState<Subject | null>(null);
  const [maps, setMaps] = useState<MapVolume[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const selected = useMemo(() => maps.find(m => m.id === selectedId) ?? null, [maps, selectedId]);

  const availableByBase = useMemo(() => {
    const m = new Map<string, MapVolume>();
    for (const v of maps) m.set(baseNoExt(v.name).toLowerCase(), v);
    return m;
  }, [maps]);

  const expectedSlots = useMemo(() => {
    return EXPECTED_MAPS.map(slot => {
      const hit = slot.candidates
        .map(c => availableByBase.get(c.toLowerCase()))
        .find(Boolean) ?? null;
      return { slot, hit };
    });
  }, [availableByBase]);

  const [ensuringMaps, setEnsuringMaps] = useState(false);
  const [ensureMsg, setEnsureMsg] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await engine.getSubject(subjectId);
        if (cancelled) return;
        setSubject(s ?? null);

        const list = await engine.getMapVolumes(subjectId);
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

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="mb-4 text-lg font-semibold">Parameter Maps (NIfTI)</h2>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {expectedSlots.map(({ slot, hit }) => {
            const isSelected = hit?.id && hit.id === selectedId;
            const available = !!hit;
            return (
              <button
                key={slot.id}
                type="button"
                onClick={async () => {
                  if (available && hit) {
                    setSelectedId(hit.id);
                    return;
                  }

                  // Missing: schedule a full run for this subject.
                  if (!subject) return;
                  try {
                    setEnsuringMaps(true);
                    setEnsureMsg('Queued full pipeline (waiting for worker)…');
                    await engine.runFullPipeline(subject.projectId, [subjectId]);
                  } catch (e: any) {
                    setEnsureMsg(String(e?.message || e || 'Failed to queue pipeline'));
                  } finally {
                    setEnsuringMaps(false);
                  }
                }}
                className={`rounded-lg border p-6 text-left transition-colors ${
                  isSelected
                    ? 'border-primary/60 bg-primary/5'
                    : 'border-border bg-card hover:bg-muted/40'
                }`}
              >
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-semibold">{slot.label}</h3>
                  {available ? <Badge variant="secondary">Available</Badge> : <Badge variant="outline">Missing</Badge>}
                </div>
                <div className="mt-3 text-xs text-muted-foreground">
                  {slot.group === 'diffusion' ? 'Diffusion' : 'Modelling'}
                </div>
                <div className="mono mt-2 text-[11px] text-muted-foreground">
                  {available ? baseNoExt(hit!.name) : 'Click to run'}
                </div>
              </button>
            );
          })}
        </div>

        <div className="mt-6 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-base font-semibold">Map Viewer</h3>
            <div className="w-[260px]">
              <Select value={selectedId} onValueChange={setSelectedId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select an available map" />
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

          {ensureMsg ? (
            <div className="text-xs text-muted-foreground">{ensuringMaps ? ensureMsg : ensureMsg}</div>
          ) : null}
        </div>
      </Card>
    </div>
  );
}
