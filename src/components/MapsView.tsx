import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
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

type MapVariantKey = 'voxel' | 'segmentation' | 'tissue';

type TissueInfo = {
  tissueKey: string;
  tissueLabel: string;
  methodKey: string;
  methodLabel: string;
};

type MapVariantDef = {
  key: MapVariantKey;
  label: string;
  slot: ExpectedMap;
};

type MapGroupDef = {
  id: string;
  label: string;
  group: 'modelling' | 'diffusion';
  variants: MapVariantDef[];
};

// Candidate basenames are compared case-insensitively.
// NOTE: include both legacy names and current p-brain outputs.
const MAP_GROUPS: MapGroupDef[] = [
  {
    id: 'Ki',
    label: 'Ki',
    group: 'modelling',
    variants: [
      {
        key: 'voxel',
        label: 'Voxel',
        slot: {
          id: 'Ki_per_voxel',
          label: 'Ki (voxel)',
          group: 'modelling',
          candidates: ['Ki_per_voxel_tikhonov', 'Ki_per_voxel_patlak', 'Ki_per_voxel_two_compartment'],
        },
      },
      {
        key: 'segmentation',
        label: 'Segmentation',
        slot: {
          id: 'Ki_map_atlas',
          label: 'Ki (atlas)',
          group: 'modelling',
          candidates: ['Ki_map_atlas', 'Ki_map_atlas_tikhonov', 'Ki_map_atlas_patlak', 'Ki_map_atlas_two_compartment'],
        },
      },
      {
        key: 'tissue',
        label: 'Tissue',
        slot: {
          id: 'Ki_tissue',
          label: 'Ki (tissue)',
          group: 'modelling',
          candidates: [
            'Ki_wm_tikhonov',
            'Ki_wm_patlak',
            'Ki_wm_two_compartment',
            'Ki_cortical_gm_tikhonov',
            'Ki_cortical_gm_patlak',
            'Ki_cortical_gm_two_compartment',
            'Ki_subcortical_gm_tikhonov',
            'Ki_subcortical_gm_patlak',
            'Ki_subcortical_gm_two_compartment',
            'Ki_boundary_tikhonov',
            'Ki_boundary_patlak',
            'Ki_boundary_two_compartment',
          ],
        },
      },
    ],
  },
  {
    id: 'vp',
    label: 'vp',
    group: 'modelling',
    variants: [
      {
        key: 'voxel',
        label: 'Voxel',
        slot: {
          id: 'vp_per_voxel',
          label: 'vp (voxel)',
          group: 'modelling',
          candidates: ['vp_per_voxel_tikhonov', 'vp_per_voxel_patlak', 'vp_per_voxel_two_compartment'],
        },
      },
      {
        key: 'segmentation',
        label: 'Segmentation',
        slot: {
          id: 'vp_map_atlas',
          label: 'vp (atlas)',
          group: 'modelling',
          candidates: ['vp_map_atlas_tikhonov', 'vp_map_atlas_patlak', 'vp_map_atlas_two_compartment'],
        },
      },
      {
        key: 'tissue',
        label: 'Tissue',
        slot: { id: 'vp_tissue', label: 'vp (tissue)', group: 'modelling', candidates: [] },
      },
    ],
  },
  {
    id: 'CBF',
    label: 'CBF',
    group: 'modelling',
    variants: [
      {
        key: 'voxel',
        label: 'Voxel',
        slot: {
          id: 'CBF_per_voxel',
          label: 'CBF (voxel)',
          group: 'modelling',
          candidates: ['CBF_per_voxel_tikhonov', 'CBF_per_voxel_patlak', 'CBF_per_voxel_two_compartment'],
        },
      },
      {
        key: 'segmentation',
        label: 'Segmentation',
        slot: {
          id: 'CBF_atlas',
          label: 'CBF (atlas)',
          group: 'modelling',
          candidates: ['CBF_tikhonov_map_atlas', 'CBF_tikhonov_map_atlas_tikhonov', 'CBF_tikhonov_map_atlas_patlak'],
        },
      },
      {
        key: 'tissue',
        label: 'Tissue',
        slot: { id: 'CBF_tissue', label: 'CBF (tissue)', group: 'modelling', candidates: [] },
      },
    ],
  },
  {
    id: 'MTT',
    label: 'MTT',
    group: 'modelling',
    variants: [
      {
        key: 'voxel',
        label: 'Voxel',
        slot: { id: 'mtt_map', label: 'MTT (voxel)', group: 'modelling', candidates: ['mtt_map'] },
      },
      {
        key: 'segmentation',
        label: 'Segmentation',
        slot: {
          id: 'MTT_atlas',
          label: 'MTT (atlas)',
          group: 'modelling',
          candidates: ['MTT_tikhonov_map_atlas', 'MTT_tikhonov_map_atlas_tikhonov', 'MTT_tikhonov_map_atlas_patlak'],
        },
      },
      {
        key: 'tissue',
        label: 'Tissue',
        slot: { id: 'MTT_tissue', label: 'MTT (tissue)', group: 'modelling', candidates: [] },
      },
    ],
  },
  {
    id: 'CTH',
    label: 'CTH',
    group: 'modelling',
    variants: [
      {
        key: 'voxel',
        label: 'Voxel',
        slot: { id: 'cth_map', label: 'CTH (voxel)', group: 'modelling', candidates: ['cth_map'] },
      },
      {
        key: 'segmentation',
        label: 'Segmentation',
        slot: {
          id: 'CTH_atlas',
          label: 'CTH (atlas)',
          group: 'modelling',
          candidates: ['CTH_tikhonov_map_atlas', 'CTH_tikhonov_map_atlas_tikhonov', 'CTH_tikhonov_map_atlas_patlak'],
        },
      },
      {
        key: 'tissue',
        label: 'Tissue',
        slot: { id: 'CTH_tissue', label: 'CTH (tissue)', group: 'modelling', candidates: [] },
      },
    ],
  },
  {
    id: 'FA',
    label: 'FA',
    group: 'diffusion',
    variants: [
      { key: 'voxel', label: 'Voxel', slot: { id: 'FA', label: 'FA', group: 'diffusion', candidates: ['FA_map', 'fa_map'] } },
      { key: 'segmentation', label: 'Segmentation', slot: { id: 'FA_seg', label: 'FA (atlas)', group: 'diffusion', candidates: ['fa_map_atlas'] } },
      { key: 'tissue', label: 'Tissue', slot: { id: 'FA_tissue', label: 'FA (WM)', group: 'diffusion', candidates: ['FA_WM_map', 'fa_wm_map'] } },
    ],
  },
  {
    id: 'MD',
    label: 'MD',
    group: 'diffusion',
    variants: [
      { key: 'voxel', label: 'Voxel', slot: { id: 'MD', label: 'MD', group: 'diffusion', candidates: ['md_map', 'MD_map'] } },
      { key: 'segmentation', label: 'Segmentation', slot: { id: 'MD_seg', label: 'MD (seg)', group: 'diffusion', candidates: [] } },
      { key: 'tissue', label: 'Tissue', slot: { id: 'MD_tissue', label: 'MD (tissue)', group: 'diffusion', candidates: [] } },
    ],
  },
  {
    id: 'AD',
    label: 'AD',
    group: 'diffusion',
    variants: [
      { key: 'voxel', label: 'Voxel', slot: { id: 'AD', label: 'AD', group: 'diffusion', candidates: ['ad_map', 'AD_map'] } },
      { key: 'segmentation', label: 'Segmentation', slot: { id: 'AD_seg', label: 'AD (seg)', group: 'diffusion', candidates: [] } },
      { key: 'tissue', label: 'Tissue', slot: { id: 'AD_tissue', label: 'AD (tissue)', group: 'diffusion', candidates: [] } },
    ],
  },
  {
    id: 'RD',
    label: 'RD',
    group: 'diffusion',
    variants: [
      { key: 'voxel', label: 'Voxel', slot: { id: 'RD', label: 'RD', group: 'diffusion', candidates: ['rd_map', 'RD_map'] } },
      { key: 'segmentation', label: 'Segmentation', slot: { id: 'RD_seg', label: 'RD (seg)', group: 'diffusion', candidates: [] } },
      { key: 'tissue', label: 'Tissue', slot: { id: 'RD_tissue', label: 'RD (tissue)', group: 'diffusion', candidates: [] } },
    ],
  },
  {
    id: 'MO',
    label: 'MO',
    group: 'diffusion',
    variants: [
      { key: 'voxel', label: 'Voxel', slot: { id: 'MO', label: 'MO', group: 'diffusion', candidates: ['mo_map', 'MO_map'] } },
      { key: 'segmentation', label: 'Segmentation', slot: { id: 'MO_seg', label: 'MO (seg)', group: 'diffusion', candidates: [] } },
      { key: 'tissue', label: 'Tissue', slot: { id: 'MO_tissue', label: 'MO (tissue)', group: 'diffusion', candidates: [] } },
    ],
  },
  {
    id: 'TensorResidual',
    label: 'Tensor residual',
    group: 'diffusion',
    variants: [
      {
        key: 'voxel',
        label: 'Voxel',
        slot: { id: 'TensorResidual', label: 'Tensor residual', group: 'diffusion', candidates: ['tensor_residual_map', 'tensor_residual'] },
      },
      { key: 'segmentation', label: 'Segmentation', slot: { id: 'TensorResidual_seg', label: 'Tensor residual (seg)', group: 'diffusion', candidates: [] } },
      { key: 'tissue', label: 'Tissue', slot: { id: 'TensorResidual_tissue', label: 'Tensor residual (tissue)', group: 'diffusion', candidates: [] } },
    ],
  },
];

function baseNoExt(filename: string): string {
  return filename.replace(/\.(nii|nii\.gz|png)$/i, '');
}

function basenameFromPath(p: string): string {
  const s = String(p || '').replace(/\\/g, '/');
  const parts = s.split('/').filter(Boolean);
  return parts[parts.length - 1] || s;
}

function parseTissueInfoFromName(name: string): TissueInfo {
  const n = String(name || '').toLowerCase();

  const methodKey = /_patlak\b/.test(n) ? 'patlak' : /_two_compartment\b/.test(n) ? 'two_compartment' : /_tikhonov\b/.test(n) ? 'tikhonov' : 'default';
  const methodLabel = methodKey === 'patlak' ? 'Patlak' : methodKey === 'two_compartment' ? '2-comp' : methodKey === 'tikhonov' ? 'Tikhonov' : 'Default';

  // Tissue regions seen in real outputs (plus a few common ones).
  const regionMatchers: Array<{ key: string; label: string; re: RegExp }> = [
    { key: 'wm', label: 'WM', re: /(^|_)wm(_|\b)/ },
    { key: 'cortical_gm', label: 'Cortical GM', re: /(^|_)cortical_gm(_|\b)/ },
    { key: 'subcortical_gm', label: 'Subcortical GM', re: /(^|_)subcortical_gm(_|\b)/ },
    { key: 'gm_brainstem', label: 'Brainstem GM', re: /(^|_)gm_brainstem(_|\b)/ },
    { key: 'gm_cerebellum', label: 'Cerebellum GM', re: /(^|_)gm_cerebellum(_|\b)/ },
    { key: 'wm_cerebellum', label: 'Cerebellum WM', re: /(^|_)wm_cerebellum(_|\b)/ },
    { key: 'wm_cc', label: 'Corpus Callosum', re: /(^|_)wm_cc(_|\b)/ },
    { key: 'boundary', label: 'Boundary', re: /(^|_)boundary(_|\b)/ },
  ];

  for (const r of regionMatchers) {
    if (r.re.test(n)) return { tissueKey: r.key, tissueLabel: r.label, methodKey, methodLabel };
  }

  // Some diffusion WM files use fa_wm_map etc.
  if (/_wm_map\b/.test(n) || /_wm_/.test(n)) return { tissueKey: 'wm', tissueLabel: 'WM', methodKey, methodLabel };

  return { tissueKey: 'other', tissueLabel: 'Other', methodKey, methodLabel };
}

const TISSUE_ORDER = ['wm', 'cortical_gm', 'subcortical_gm', 'gm_brainstem', 'gm_cerebellum', 'wm_cerebellum', 'wm_cc', 'boundary', 'other'];

function labelForMontage(filename: string): string {
  const base = baseNoExt(filename).toLowerCase();
  if (base === 'cbf_montage') return 'CBF';
  if (base === 'mtt_montage') return 'MTT';
  if (base === 'cth_montage') return 'CTH';
  if (/^ki.*montage/.test(base)) return 'Ki';
  if (/^vp.*montage/.test(base)) return 'vp';
  if (/^fa.*montage/.test(base)) return 'FA';
  if (/^md.*montage/.test(base)) return 'MD';
  if (/^ad.*montage/.test(base)) return 'AD';
  if (/^rd.*montage/.test(base)) return 'RD';
  if (/^mo.*montage/.test(base)) return 'MO';
  return filename;
}

export function MapsView({ subjectId }: MapsViewProps) {
  const [subject, setSubject] = useState<Subject | null>(null);
  const [maps, setMaps] = useState<MapVolume[]>([]);

  const [openGroupId, setOpenGroupId] = useState<string>('');
  const [activeVariant, setActiveVariant] = useState<MapVariantKey>('voxel');

  const [montages, setMontages] = useState<Array<{ id: string; name: string; path: string }>>([]);
  const [selectedMontageId, setSelectedMontageId] = useState<string>('');
  const selectedMontage = useMemo(
    () => montages.find(m => m.id === selectedMontageId) ?? null,
    [montages, selectedMontageId]
  );

  const availableByBase = useMemo(() => {
    const m = new Map<string, MapVolume>();
    for (const v of maps) {
      const key = baseNoExt(basenameFromPath((v as any)?.path || v.name || v.id)).toLowerCase();
      m.set(key, v);
    }
    return m;
  }, [maps]);

  type MapGroupRuntime = {
    id: string;
    label: string;
    group: 'modelling' | 'diffusion';
    variants: Array<{ key: MapVariantKey; label: string; slot: ExpectedMap; hits: MapVolume[] }>;
    anyAvailable: boolean;
  };

  const grouped = useMemo((): MapGroupRuntime[] => {
    return MAP_GROUPS.map(g => {
      const variants = g.variants.map(v => {
        const hits = v.slot.candidates
          .map(c => availableByBase.get(String(c).toLowerCase()))
          .filter(Boolean) as MapVolume[];
        return { ...v, hits };
      });
      return {
        id: g.id,
        label: g.label,
        group: g.group,
        variants,
        anyAvailable: variants.some(v => (v.hits?.length ?? 0) > 0),
      };
    });
  }, [availableByBase]);

  const openGroup = useMemo(() => grouped.find(g => g.id === openGroupId) ?? null, [grouped, openGroupId]);

  const [activeFileId, setActiveFileId] = useState<string>('');
  const [activeTissueKey, setActiveTissueKey] = useState<string>('wm');
  const [activeTissueMethod, setActiveTissueMethod] = useState<string>('tikhonov');

  useEffect(() => {
    if (!openGroup) return;
    const preferred = openGroup.variants.find(v => v.key === activeVariant && (v.hits?.length ?? 0) > 0);
    if (preferred) return;
    const firstAvail = openGroup.variants.find(v => (v.hits?.length ?? 0) > 0);
    setActiveVariant(firstAvail?.key ?? 'voxel');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openGroupId]);

  useEffect(() => {
    if (!openGroup) {
      setActiveFileId('');
      return;
    }
    const v = openGroup.variants.find(x => x.key === activeVariant) ?? openGroup.variants[0];
    const first = v?.hits?.[0];
    setActiveFileId(first?.id ?? '');
		if (activeVariant === 'tissue' && first) {
			const info = parseTissueInfoFromName(first.name);
			setActiveTissueKey(info.tissueKey);
			setActiveTissueMethod(info.methodKey);
		}
  }, [openGroupId, activeVariant, openGroup]);

  const [ensuringMaps, setEnsuringMaps] = useState(false);
  const [ensureMsg, setEnsureMsg] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const s = await engine.getSubject(subjectId);
        if (cancelled) return;
        setSubject(s ?? null);

        const montageList = await engine.getMontageImages(subjectId);
        if (cancelled) return;
        setMontages(montageList);
        setSelectedMontageId(prev => {
          if (prev && montageList.some(m => m.id === prev)) return prev;
          return montageList[0]?.id ?? '';
        });

        const list = await engine.getMapVolumes(subjectId);
        if (cancelled) return;
        setMaps(list);
      } catch (err) {
        console.error('Failed to load map volumes:', err);
        if (!cancelled) {
          setMontages([]);
          setSelectedMontageId('');
          setMaps([]);
          setOpenGroupId('');
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

        <div className="space-y-4">
          {montages.length > 0 ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary">Montages</Badge>
                  <span className="text-xs text-muted-foreground">{montages.length} images</span>
                </div>

                <div className="w-[320px]">
                  <Select value={selectedMontageId} onValueChange={setSelectedMontageId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a montage" />
                    </SelectTrigger>
                    <SelectContent>
                      {montages.map(m => (
                        <SelectItem key={m.id} value={m.id}>
                          {labelForMontage(m.name)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {selectedMontage ? (
                <div className="overflow-hidden rounded-lg border bg-card">
                  <PanZoomImage src={selectedMontage.path} alt={selectedMontage.name} />
                </div>
              ) : null}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No montage images found yet.</div>
          )}

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {grouped.map(g => {
              const available = g.anyAvailable;
              return (
                <button
                  key={g.id}
                  type="button"
                  onClick={() => setOpenGroupId(g.id)}
                  className="rounded-lg border border-border bg-card p-6 text-left transition-colors hover:bg-muted/40"
                >
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="font-semibold">{g.label}</h3>
                    {available ? <Badge variant="secondary">Available</Badge> : <Badge variant="outline">Missing</Badge>}
                  </div>
                  <div className="mt-3 text-xs text-muted-foreground">{g.group === 'diffusion' ? 'Diffusion' : 'Modelling'}</div>
                  <div className="mono mt-2 text-[11px] text-muted-foreground">
                    {available ? 'Click to view' : 'Click to view / run'}
                  </div>
                </button>
              );
            })}
          </div>

          {ensureMsg ? <div className="text-xs text-muted-foreground">{ensureMsg}</div> : null}
        </div>
      </Card>

      <Dialog
        open={!!openGroupId}
        onOpenChange={open => {
          if (!open) setOpenGroupId('');
        }}
      >
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>{openGroup?.label || 'Map'}</DialogTitle>
            <DialogDescription>
              {openGroup?.group === 'diffusion' ? 'Diffusion parameter maps.' : 'Modelling parameter maps.'}
            </DialogDescription>
          </DialogHeader>

          {openGroup ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap gap-2">
                  {openGroup.variants.map(v => {
                    const isActive = v.key === activeVariant;
                    const has = (v.hits?.length ?? 0) > 0;
                    return (
                      <Button
                        key={v.key}
                        type="button"
                        size="sm"
                        variant={isActive ? 'default' : 'outline'}
                        onClick={() => setActiveVariant(v.key)}
                      >
                        {v.label}
                        {!has ? ' (missing)' : ''}
                      </Button>
                    );
                  })}
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    disabled={!subject || ensuringMaps}
                    onClick={async () => {
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
                  >
                    Run pipeline
                  </Button>
                </div>
              </div>

              {(() => {
                const v = openGroup.variants.find(x => x.key === activeVariant) ?? openGroup.variants[0];
                if (!v) return null;
                if (!v.hits || v.hits.length === 0) {
                  return (
                    <div className="rounded-lg border border-border bg-muted/20 p-6 text-sm text-muted-foreground">
                      {v.slot.label} is missing for this subject.
                    </div>
                  );
                }

                const active = v.hits.find(h => h.id === activeFileId) ?? v.hits[0];
                const options = v.hits;

                return (
                  <div className="space-y-3">
                    {activeVariant === 'tissue' ? (
                      (() => {
                        const enriched = options
                          .map(o => ({ o, info: parseTissueInfoFromName(o.name) }))
                          .sort((a, b) => {
                            const ai = TISSUE_ORDER.indexOf(a.info.tissueKey);
                            const bi = TISSUE_ORDER.indexOf(b.info.tissueKey);
                            if (ai !== bi) return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
                            return a.o.name.localeCompare(b.o.name);
                          });

                        const tissueKeys = Array.from(new Set(enriched.map(x => x.info.tissueKey)));
                        tissueKeys.sort((a, b) => {
                          const ai = TISSUE_ORDER.indexOf(a);
                          const bi = TISSUE_ORDER.indexOf(b);
                          return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
                        });

                        const key = tissueKeys.includes(activeTissueKey) ? activeTissueKey : tissueKeys[0] || 'other';
                        const tissueOptions = enriched.filter(x => x.info.tissueKey === key);
                        const methods = Array.from(new Set(tissueOptions.map(x => x.info.methodKey)));
                        const methodKey = methods.includes(activeTissueMethod) ? activeTissueMethod : methods[0] || 'default';
                        const selected = tissueOptions.find(x => x.info.methodKey === methodKey)?.o ?? tissueOptions[0]?.o;
                        const methodLabelFor = (mk: string) => (mk === 'patlak' ? 'Patlak' : mk === 'two_compartment' ? '2-comp' : mk === 'tikhonov' ? 'Tikhonov' : 'Default');

                        return (
                          <div className="space-y-3">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <div className="flex flex-wrap gap-2">
                                {tissueKeys.map(tk => {
                                  const label = enriched.find(x => x.info.tissueKey === tk)?.info.tissueLabel || tk;
                                  return (
                                    <Button
                                      key={tk}
                                      type="button"
                                      size="sm"
                                      variant={tk === key ? 'default' : 'outline'}
                                      onClick={() => {
                                        setActiveTissueKey(tk);
                                        // reset method if current not present
                                        const ms = Array.from(new Set(enriched.filter(x => x.info.tissueKey === tk).map(x => x.info.methodKey)));
                                        setActiveTissueMethod(ms.includes(activeTissueMethod) ? activeTissueMethod : (ms[0] || 'default'));
                                      }}
                                    >
                                      {label}
                                    </Button>
                                  );
                                })}
                              </div>

                              {methods.length > 1 ? (
                                <div className="flex flex-wrap gap-2">
                                  {methods.map(mk => (
                                    <Button
                                      key={mk}
                                      type="button"
                                      size="sm"
                                      variant={mk === methodKey ? 'secondary' : 'outline'}
                                      onClick={() => setActiveTissueMethod(mk)}
                                    >
                                      {methodLabelFor(mk)}
                                    </Button>
                                  ))}
                                </div>
                              ) : null}
                            </div>

                            {selected ? (
                              <>
                                <div className="text-xs text-muted-foreground">{selected.name}</div>
                                <VolumeViewer subjectId={subjectId} path={selected.path} kind="map" />
                              </>
                            ) : (
                              <div className="rounded-lg border border-border bg-muted/20 p-6 text-sm text-muted-foreground">
                                No tissue maps found.
                              </div>
                            )}
                          </div>
                        );
                      })()
                    ) : options.length > 1 ? (
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-xs text-muted-foreground">Showing:</div>
                        <div className="w-[360px]">
                          <Select value={active?.id ?? ''} onValueChange={setActiveFileId}>
                            <SelectTrigger>
                              <SelectValue placeholder="Select a variant file" />
                            </SelectTrigger>
                            <SelectContent>
                              {options.map(o => (
                                <SelectItem key={o.id} value={o.id}>
                                  {o.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    ) : null}

                    {activeVariant !== 'tissue' ? <VolumeViewer subjectId={subjectId} path={active.path} kind="map" /> : null}
                  </div>
                );
              })()}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

function PanZoomImage({ src, alt }: { src: string; alt: string }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scale, setScale] = useState(1);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [drag, setDrag] = useState<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);

  const reset = () => {
    setScale(1);
    setTx(0);
    setTy(0);
  };

  return (
    <div
      ref={containerRef}
      className="relative h-[520px] w-full select-none overflow-hidden bg-black/5"
      onDoubleClick={reset}
      onMouseLeave={() => setHover(null)}
      onMouseMove={(e) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const ix = (cx - tx) / scale;
        const iy = (cy - ty) / scale;
        setHover({ x: ix, y: iy });

        if (drag) {
          setTx(drag.tx + (e.clientX - drag.x));
          setTy(drag.ty + (e.clientY - drag.y));
        }
      }}
      onMouseDown={(e) => {
        e.preventDefault();
        setDrag({ x: e.clientX, y: e.clientY, tx, ty });
      }}
      onMouseUp={() => setDrag(null)}
      onWheel={(e) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        e.preventDefault();

        const zoom = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const nextScale = clamp(scale * zoom, 0.5, 10);
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        // Keep the point under the cursor stable during zoom.
        const ix = (cx - tx) / scale;
        const iy = (cy - ty) / scale;
        const nextTx = cx - ix * nextScale;
        const nextTy = cy - iy * nextScale;

        setScale(nextScale);
        setTx(nextTx);
        setTy(nextTy);
      }}
    >
      <img
        src={src}
        alt={alt}
        loading="lazy"
        draggable={false}
        className="absolute left-0 top-0 max-w-none"
        style={{ transform: `translate(${tx}px, ${ty}px) scale(${scale})`, transformOrigin: '0 0' }}
      />

      <div className="pointer-events-none absolute bottom-2 left-2 rounded-md bg-background/80 px-2 py-1 text-[11px] text-muted-foreground shadow-sm">
        <div className="mono">wheel: zoom · drag: pan · dblclick: reset</div>
        {hover ? <div className="mono">x={Math.round(hover.x)} y={Math.round(hover.y)}</div> : null}
      </div>
    </div>
  );
}
