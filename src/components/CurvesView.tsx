import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { engine } from '@/lib/engine';
import type { Curve, PatlakData, ToftsData, DeconvolutionData } from '@/types';

interface CurvesViewProps {
  subjectId: string;
}

export function CurvesView({ subjectId }: CurvesViewProps) {
  const [curves, setCurves] = useState<Curve[]>([]);
  const [patlakData, setPatlakData] = useState<PatlakData | null>(null);
  const [toftsData, setToftsData] = useState<ToftsData | null>(null);
  const [deconvData, setDeconvData] = useState<DeconvolutionData | null>(null);
  const [ensuring, setEnsuring] = useState(false);
  const [ensureMsg, setEnsureMsg] = useState('');
  const [ensureOnce, setEnsureOnce] = useState(false);

  useEffect(() => {
    loadData();
  }, [subjectId]);

  const loadData = async () => {
    try {
      const curvesData = await engine.getCurves(subjectId);
      setCurves(curvesData);

      const patlak = await mockEngine.getPatlakData(subjectId, 'gm');
      setPatlakData(patlak);

      const tofts = await mockEngine.getToftsData(subjectId, 'gm');
      setToftsData(tofts);

      const deconv = await mockEngine.getDeconvolutionData(subjectId, 'gm');
      setDeconvData(deconv);
    } catch (error) {
      console.error('Failed to load curves:', error);
    }
  };

  useEffect(() => {
    if (!isBackendEngine) return;
    if (ensureOnce) return;
    if (curves.length > 0) return;

    setEnsureOnce(true);
    setEnsuring(true);
    setEnsureMsg('Running p-brain to generate curves…');
    mockEngine
      .ensureSubjectArtifacts(subjectId, 'curves')
      .then((res: any) => setEnsureMsg(res?.reason || 'Started'))
      .catch((e: any) => setEnsureMsg(String(e?.message || e || 'Failed to start pipeline')))
      .finally(() => setEnsuring(false));
  }, [subjectId, curves.length, ensureOnce]);

  useEffect(() => {
    if (!isBackendEngine) return;
    if (curves.length > 0) return;
    if (!ensureOnce) return;

    const t = window.setInterval(async () => {
      try {
        const curvesData = await mockEngine.getCurves(subjectId);
        if (curvesData.length > 0) {
          setCurves(curvesData);
          window.clearInterval(t);
        }
      } catch {
        // ignore
      }
    }, 2500);
    return () => window.clearInterval(t);
  }, [subjectId, curves.length, ensureOnce]);

  return (
    <div className="space-y-6">
      <Tabs defaultValue="concentration" className="w-full">
        <TabsList>
          <TabsTrigger value="concentration">Concentration Curves</TabsTrigger>
          <TabsTrigger value="patlak">Patlak Analysis</TabsTrigger>
          <TabsTrigger value="tofts">Extended Tofts</TabsTrigger>
          <TabsTrigger value="deconvolution">Deconvolution</TabsTrigger>
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
                }))}
                layout={{
                  autosize: true,
                  margin: { l: 60, r: 40, t: 40, b: 60 },
                  xaxis: {
                    title: 'Time (s)',
                    gridcolor: '#e0e0e0',
                  },
                  yaxis: {
                    title: 'Concentration (mM)',
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
                          const res = await mockEngine.ensureSubjectArtifacts(subjectId, 'curves');
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
            <h2 className="mb-4 text-lg font-semibold">Patlak Analysis (Gray Matter)</h2>
            {patlakData ? (
              <div className="space-y-6">
                <Plot
                  data={[
                    {
                      x: patlakData.x,
                      y: patlakData.y,
                      name: 'Measured',
                      type: 'scatter',
                      mode: 'markers',
                      marker: { size: 6, color: '#4A90E2' },
                    },
                    {
                      x: patlakData.fitLineX,
                      y: patlakData.fitLineY,
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
                      {patlakData.Ki.toFixed(2)}
                      <span className="ml-2 text-sm font-normal">ml/100g/min</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">vp</div>
                    <div className="mono text-2xl font-semibold">
                      {patlakData.vp.toFixed(3)}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">R²</div>
                    <div className="mono text-2xl font-semibold">
                      {patlakData.r2.toFixed(3)}
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

        <TabsContent value="tofts">
          <Card className="p-6">
            <h2 className="mb-4 text-lg font-semibold">Extended Tofts Model (Gray Matter)</h2>
            {toftsData ? (
              <div className="space-y-6">
                <Plot
                  data={[
                    {
                      x: toftsData.timePoints,
                      y: toftsData.measured,
                      name: 'Measured Ct(t)',
                      type: 'scatter',
                      mode: 'markers',
                      marker: { size: 5, color: '#4A90E2' },
                    },
                    {
                      x: toftsData.timePoints,
                      y: toftsData.fitted,
                      name: 'Fitted Ct(t)',
                      type: 'scatter',
                      mode: 'lines',
                      line: { width: 3, color: '#E94B3C' },
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 60, r: 40, t: 40, b: 60 },
                    xaxis: {
                      title: 'Time (s)',
                      gridcolor: '#e0e0e0',
                    },
                    yaxis: {
                      title: 'Tissue Concentration (mM)',
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
                    <div className="mb-1 text-sm text-muted-foreground">Ktrans</div>
                    <div className="mono text-2xl font-semibold">
                      {toftsData.Ktrans.toFixed(3)}
                      <span className="ml-2 text-sm font-normal">min⁻¹</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">ve</div>
                    <div className="mono text-2xl font-semibold">{toftsData.ve.toFixed(3)}</div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">vp</div>
                    <div className="mono text-2xl font-semibold">{toftsData.vp.toFixed(3)}</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-[400px] items-center justify-center text-muted-foreground">
                No Tofts data available
              </div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="deconvolution">
          <Card className="p-6">
            <h2 className="mb-4 text-lg font-semibold">Deconvolution & Perfusion Metrics (Gray Matter)</h2>
            {deconvData ? (
              <div className="space-y-6">
                <div className="grid gap-6 lg:grid-cols-2">
                  <div>
                    <h3 className="mb-3 text-sm font-semibold">Residue Function R(t)</h3>
                    <Plot
                      data={[
                        {
                          x: deconvData.timePoints,
                          y: deconvData.residue,
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
                          x: deconvData.timePoints,
                          y: deconvData.h_t,
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
                      {deconvData.CBF.toFixed(1)}
                      <span className="ml-2 text-sm font-normal">ml/100g/min</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">MTT</div>
                    <div className="mono text-2xl font-semibold">
                      {deconvData.MTT.toFixed(2)}
                      <span className="ml-2 text-sm font-normal">s</span>
                    </div>
                  </div>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <div className="mb-1 text-sm text-muted-foreground">CTH</div>
                    <div className="mono text-2xl font-semibold">
                      {deconvData.CTH.toFixed(2)}
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
