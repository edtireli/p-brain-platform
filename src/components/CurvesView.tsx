import { useState, useEffect } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { engine, isBackendEngine } from '@/lib/engine';
import type { Curve, PatlakData, DeconvolutionData } from '@/types';

const Plot = createPlotlyComponent(Plotly);

interface CurvesViewProps {
  subjectId: string;
}

export function CurvesView({ subjectId }: CurvesViewProps) {
  const [curves, setCurves] = useState<Curve[]>([]);
  const [patlakData, setPatlakData] = useState<PatlakData | null>(null);
  const [deconvData, setDeconvData] = useState<DeconvolutionData | null>(null);
  const fmt = (value: unknown, digits: number): string => {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  };
  const [availableRegions, setAvailableRegions] = useState<Array<{ key: string; label: string }>>([]);
  const [activeRegion, setActiveRegion] = useState<string>('gm');
  const [ensuring, setEnsuring] = useState(false);
  const [ensureMsg, setEnsureMsg] = useState('');
  const [ensureOnce, setEnsureOnce] = useState(false);

  useEffect(() => {
    loadData();
  }, [subjectId]);

  useEffect(() => {
    // Reload modelling outputs when region changes.
    loadModels(activeRegion);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subjectId, activeRegion]);

  const prettyRegionLabel = (key: string): string => {
    const k = (key || '').toLowerCase();
    const mapping: Record<string, string> = {
      gm: 'Gray Matter',
      cortical_gm: 'Cortical GM',
      subcortical_gm: 'Subcortical GM',
      wm: 'White Matter',
      boundary: 'Boundary',
      gm_brainstem: 'Brainstem (GM)',
      gm_cerebellum: 'Cerebellum (GM)',
      wm_cerebellum: 'Cerebellum (WM)',
      wm_cc: 'Corpus callosum (WM)',
    };
    if (mapping[k]) return mapping[k];
    return k.replace(/_/g, ' ').replace(/\b\w/g, m => m.toUpperCase());
  };

  const loadData = async () => {
    try {
      const curvesData = await engine.getCurves(subjectId);
      setCurves(curvesData);

      // Discover regions from AI tissue curves (preferred for modelling endpoints).
      const seen = new Set<string>();
      const regs = curvesData
        .filter(c => /^tissue_ai_/i.test(c.id))
        .map(c => c.id.replace(/^tissue_ai_/i, ''))
        .filter(k => {
          const kk = (k || '').trim();
          if (!kk) return false;
          if (seen.has(kk)) return false;
          seen.add(kk);
          return true;
        })
        .map(k => ({ key: k, label: prettyRegionLabel(k) }));

      regs.sort((a, b) => a.label.localeCompare(b.label));
      setAvailableRegions(regs);

      setActiveRegion(prev => {
        if (regs.length === 0) return prev || 'gm';
        return regs.some(r => r.key === prev) ? prev : regs[0]!.key;
      });
    } catch (error) {
      console.error('Failed to load curves:', error);
    }
  };

  const loadModels = async (regionKey: string) => {
    try {
      const patlak = await engine.getPatlakData(subjectId, regionKey || 'gm');
      setPatlakData(patlak);
    } catch {
      setPatlakData(null);
    }

    try {
      const deconv = await engine.getDeconvolutionData(subjectId, regionKey || 'gm');
      setDeconvData(deconv);
    } catch {
      setDeconvData(null);
    }
  };

  useEffect(() => {
    if (!isBackendEngine) return;
    if (ensureOnce) return;
    if (curves.length > 0) return;

    setEnsureOnce(true);
    setEnsuring(true);
    setEnsureMsg('Running p-brain to generate curves…');
    engine
      .ensureSubjectArtifacts(subjectId, 'curves')
      .then((res: any) => setEnsureMsg(res?.reason || 'Started'))
      .catch((e: any) => setEnsureMsg(String(e?.message || e || 'Failed to start pipeline')))
      .finally(() => setEnsuring(false));
  }, [subjectId, curves.length, ensureOnce]);

  useEffect(() => {
    if (!isBackendEngine) return;
    if (curves.length > 0) return;
    if (!ensureOnce) return;

    let disposed = false;
    let attempts = 0;
    let t: number | null = null;

    const tick = async () => {
      if (disposed) return;
      attempts += 1;
      try {
        const curvesData = await engine.getCurves(subjectId);
        if (curvesData.length > 0) {
          setCurves(curvesData);
          return;
        }
      } catch {
        // ignore
      }
      if (disposed) return;
      if (attempts >= 24) return;
      t = window.setTimeout(() => void tick(), 2500);
    };

    t = window.setTimeout(() => void tick(), 2500);
    return () => {
      disposed = true;
      if (t != null) window.clearTimeout(t);
    };
  }, [subjectId, curves.length, ensureOnce]);

  const hasPatlak = !!patlakData && (patlakData.x?.length ?? 0) > 0 && (patlakData.y?.length ?? 0) > 0;
  const hasDeconv = !!deconvData && (deconvData.timePoints?.length ?? 0) > 0 && (deconvData.residue?.length ?? 0) > 0;

  const hasInputCurves = curves.some(c => /^aif_/i.test(c.id) || /^vif_/i.test(c.id));
  const hasTissueCurves = curves.some(c => /^tissue_/i.test(c.id));
  const useSplitAxis = hasInputCurves && hasTissueCurves;

  const regionLabel = prettyRegionLabel(activeRegion);

  return (
    <div className="space-y-6">
      <Tabs defaultValue="concentration" className="w-full">
        <TabsList>
          <TabsTrigger value="concentration">Concentration Curves</TabsTrigger>
          <TabsTrigger value="patlak">Patlak Analysis</TabsTrigger>
          <TabsTrigger value="deconvolution">Tikhonov</TabsTrigger>
        </TabsList>

        <TabsContent value="concentration">
          <Card className="p-6">
            <h2 className="mb-4 text-lg font-semibold">Concentration Time Curves</h2>
            {curves.length > 0 ? (
              <Plot
                data={curves.map(curve => ({
                  x: curve.timePoints,
                  y: curve.values,
                  name: curve.name,
                  type: 'scatter',
                  mode: 'lines',
                  line: { width: 2 },
                  yaxis: useSplitAxis && /^tissue_/i.test(curve.id) ? 'y2' : 'y',
                }))}
                layout={{
                  autosize: true,
                  margin: { l: 60, r: 40, t: 40, b: 60 },
                  xaxis: {
                    title: 'Time (s)',
                    gridcolor: '#e0e0e0',
                  },
                  yaxis: {
                    title: useSplitAxis ? 'Input (mM)' : 'Concentration (mM)',
                    gridcolor: '#e0e0e0',
                  },
                  yaxis2: useSplitAxis
                    ? {
                        title: 'Tissue (mM)',
                        overlaying: 'y',
                        side: 'right',
                        showgrid: false,
                      }
                    : undefined,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { family: 'IBM Plex Sans, sans-serif' },
                  legend: { orientation: 'h', y: -0.2 },
                }}
                config={{ responsive: true, displayModeBar: true }}
                style={{ width: '100%', height: '500px' }}
              />
            ) : (
              <div className="flex h-[400px] flex-col items-center justify-center gap-3 text-muted-foreground">
                <div>No curve data available</div>
                {isBackendEngine ? (
                  <div className="flex items-center gap-3">
                    <div className="text-xs">{ensureMsg}</div>
                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          setEnsuring(true);
                          const res = await engine.ensureSubjectArtifacts(subjectId, 'curves');
                          setEnsureMsg(res?.reason || 'Started');
                        } catch (e: any) {
                          setEnsureMsg(String(e?.message || e || 'Failed to start pipeline'));
                        } finally {
                          setEnsuring(false);
                        }
                      }}
                      disabled={ensuring}
                      className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
                    >
                      {ensuring ? 'Running…' : 'Run p-brain'}
                    </button>
                  </div>
                ) : null}
              </div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="patlak">
          <Card className="p-6">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Patlak Analysis ({regionLabel})</h2>
              {availableRegions.length > 1 ? (
                <div className="flex flex-wrap gap-2">
                  {availableRegions.map(r => (
                    <Button
                      key={r.key}
                      type="button"
                      size="sm"
                      variant={r.key === activeRegion ? 'default' : 'outline'}
                      onClick={() => setActiveRegion(r.key)}
                    >
                      {r.label}
                    </Button>
                  ))}
                </div>
              ) : null}
            </div>
            {hasPatlak ? (
              <div className="space-y-6">
                <Plot
                  data={[
                    {
                      x: patlakData!.x,
                      y: patlakData!.y,
                      name: 'Measured',
                      type: 'scatter',
                      mode: 'markers',
                      marker: { size: 6, color: '#4A90E2' },
                    },
                    {
                      x: patlakData!.fitLineX,
                      y: patlakData!.fitLineY,
                      name: 'Linear Fit',
                      type: 'scatter',
                      mode: 'lines',
                      line: { width: 3, color: '#E94B3C', dash: 'dash' },
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 60, r: 40, t: 40, b: 60 },
                    xaxis: {
                      title: 'x(t) = ∫Cp(τ)dτ / Cp(t)',
                      gridcolor: '#e0e0e0',
                    },
                    yaxis: {
                      title: 'y(t) = Ct(t) / Cp(t)',
                      gridcolor: '#e0e0e0',
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { family: 'IBM Plex Sans, sans-serif' },
                    legend: { orientation: 'h', y: -0.2 },
                  }}
                  config={{ responsive: true, displayModeBar: true }}
                  style={{ width: '100%', height: '500px' }}
                />

                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">Ki</div>
                    <div className="mono text-2xl font-semibold">
                      {fmt(patlakData?.Ki, 2)}
                      <span className="ml-2 text-sm font-normal">ml/100g/min</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">vp</div>
                    <div className="mono text-2xl font-semibold">
                      {fmt(patlakData?.vp, 3)}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">R²</div>
                    <div className="mono text-2xl font-semibold">
                      {fmt(patlakData?.r2, 3)}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                No Patlak data available
              </div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="deconvolution">
          <Card className="p-6">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Deconvolution & Perfusion Metrics ({regionLabel})</h2>
              {availableRegions.length > 1 ? (
                <div className="flex flex-wrap gap-2">
                  {availableRegions.map(r => (
                    <Button
                      key={r.key}
                      type="button"
                      size="sm"
                      variant={r.key === activeRegion ? 'default' : 'outline'}
                      onClick={() => setActiveRegion(r.key)}
                    >
                      {r.label}
                    </Button>
                  ))}
                </div>
              ) : null}
            </div>
            {hasDeconv ? (
              <div className="space-y-6">
                <div className="grid gap-6 lg:grid-cols-2">
                  <div>
                    <h3 className="mb-3 text-sm font-semibold">Residue Function R(t)</h3>
                    <Plot
                      data={[
                        {
                          x: deconvData!.timePoints,
                          y: deconvData!.residue,
                          name: 'R(t)',
                          type: 'scatter',
                          mode: 'lines',
                          line: { width: 2, color: '#4A90E2' },
                        },
                      ]}
                      layout={{
                        autosize: true,
                        margin: { l: 50, r: 20, t: 20, b: 50 },
                        xaxis: { title: 'Time (s)', gridcolor: '#e0e0e0' },
                        yaxis: { title: 'R(t)', gridcolor: '#e0e0e0' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { family: 'IBM Plex Sans, sans-serif', size: 12 },
                        showlegend: false,
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%', height: '300px' }}
                    />
                  </div>

                  <div>
                    <h3 className="mb-3 text-sm font-semibold">Transit Time Distribution h(t)</h3>
                    <Plot
                      data={[
                        {
                          x: deconvData!.timePoints,
                          y: deconvData!.h_t,
                          name: 'h(t)',
                          type: 'scatter',
                          mode: 'lines',
                          line: { width: 2, color: '#E94B3C' },
                          fill: 'tozeroy',
                          fillcolor: 'rgba(233, 75, 60, 0.2)',
                        },
                      ]}
                      layout={{
                        autosize: true,
                        margin: { l: 50, r: 20, t: 20, b: 50 },
                        xaxis: { title: 'Time (s)', gridcolor: '#e0e0e0' },
                        yaxis: { title: 'h(t)', gridcolor: '#e0e0e0' },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { family: 'IBM Plex Sans, sans-serif', size: 12 },
                        showlegend: false,
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%', height: '300px' }}
                    />
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">CBF</div>
                    <div className="mono text-2xl font-semibold">
                      {fmt(deconvData?.CBF, 1)}
                      <span className="ml-2 text-sm font-normal">ml/100g/min</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">MTT</div>
                    <div className="mono text-2xl font-semibold">
                      {fmt(deconvData?.MTT, 2)}
                      <span className="ml-2 text-sm font-normal">s</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">CTH</div>
                    <div className="mono text-2xl font-semibold">
                      {fmt(deconvData?.CTH, 2)}
                      <span className="ml-2 text-sm font-normal">s</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                No deconvolution data available
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
