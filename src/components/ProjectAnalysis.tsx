import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { engine } from '@/lib/engine';
import type {
  Project,
  ProjectAnalysisDataset,
  ProjectAnalysisGroupCompareResponse,
  ProjectAnalysisOlsResponse,
  ProjectAnalysisPearsonResponse,
} from '@/types';
import { toast } from 'sonner';

interface ProjectAnalysisProps {
  projectId: string;
  onBack: () => void;
}

type CovRow = Record<string, string>;

function parseDelimited(text: string): { rows: CovRow[]; headers: string[] } {
  const raw = (text || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = raw
    .split('\n')
    .map(l => l.trimEnd())
    .filter(l => l.trim().length > 0);

  if (lines.length === 0) return { rows: [], headers: [] };

  const headerLine = lines[0];
  const delim = headerLine.includes('\t') ? '\t' : ',';

  const headers = headerLine
    .split(delim)
    .map(h => (h || '').trim())
    .filter(Boolean);

  const rows: CovRow[] = [];
  for (const line of lines.slice(1)) {
    const parts = line.split(delim);
    const row: CovRow = {};
    for (let i = 0; i < headers.length; i++) {
      row[headers[i]] = (parts[i] ?? '').trim();
    }
    rows.push(row);
  }

  return { rows, headers };
}

function toNumber(v: unknown): number | null {
  if (v == null) return null;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

export function ProjectAnalysis({ projectId, onBack }: ProjectAnalysisProps) {
  const NONE_SELECT_VALUE = '__none__';

  const [project, setProject] = useState<Project | null>(null);
  const [dataset, setDataset] = useState<ProjectAnalysisDataset | null>(null);
  const [view, setView] = useState<'total' | 'atlas'>('total');

  const [covHeaders, setCovHeaders] = useState<string[]>([]);
  const [covRows, setCovRows] = useState<CovRow[]>([]);

  const [region, setRegion] = useState<string>('');

  const [xField, setXField] = useState<string>('');
  const [yField, setYField] = useState<string>('');
  const [pearson, setPearson] = useState<ProjectAnalysisPearsonResponse | null>(null);

  const [groupField, setGroupField] = useState<string>('');
  const [groupA, setGroupA] = useState<string>('');
  const [groupB, setGroupB] = useState<string>('');
  const [groupMetric, setGroupMetric] = useState<string>('');
  const [groupCompare, setGroupCompare] = useState<ProjectAnalysisGroupCompareResponse | null>(null);

  const [regY, setRegY] = useState<string>('');
  const [regX, setRegX] = useState<string>('');
  const [regCov1, setRegCov1] = useState<string>('');
  const [regCov2, setRegCov2] = useState<string>('');
  const [ols, setOls] = useState<ProjectAnalysisOlsResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const p = await engine.getProject(projectId);
        if (!cancelled) setProject(p || null);
      } catch {
        if (!cancelled) setProject(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const ds = await engine.getProjectAnalysisDataset(projectId, view);
        if (cancelled) return;
        setDataset(ds);
        const defaultRegion = ds.regions?.[0] || '';
        setRegion(prev => (prev && ds.regions?.includes(prev) ? prev : defaultRegion));
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setDataset(null);
          toast.error('Failed to load analysis dataset');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, view]);

  const covByKey = useMemo(() => {
    const m = new Map<string, CovRow>();
    for (const r of covRows) {
      const k =
        r.subject_id ||
        r.subjectId ||
        r.subject ||
        r.name ||
        r.id ||
        '';
      const key = String(k || '').trim();
      if (!key) continue;
      m.set(key, r);
    }
    return m;
  }, [covRows]);

  const perSubject = useMemo(() => {
    if (!dataset) return [] as Array<Record<string, any>>;

    const bySubject = new Map<string, Array<any>>();
    for (const row of dataset.rows || []) {
      const sid = String(row.subjectId || '');
      if (!sid) continue;
      const arr = bySubject.get(sid) || [];
      arr.push(row);
      bySubject.set(sid, arr);
    }

    const out: Array<Record<string, any>> = [];
    for (const [sid, rows] of bySubject.entries()) {
      const hit = rows.find(r => String(r.region || '') === String(region || '')) || rows[0];
      if (!hit) continue;
      const subjectName = String(hit.subjectName || '');

      const cov =
        covByKey.get(subjectName) ||
        covByKey.get(sid) ||
        null;

      const merged: Record<string, any> = {
        subjectId: sid,
        subjectName,
        region: String(hit.region || ''),
      };

      for (const k of Object.keys(hit)) {
        if (k === 'subjectId' || k === 'subjectName' || k === 'region') continue;
        merged[k] = hit[k];
      }

      if (cov) {
        for (const [k, v] of Object.entries(cov)) {
          if (k in merged) continue;
          merged[k] = v;
        }
      }

      out.push(merged);
    }

    out.sort((a, b) => String(a.subjectName).localeCompare(String(b.subjectName)));
    return out;
  }, [dataset, region, covByKey]);

  const metricFields = useMemo(() => {
    const set = new Set<string>();
    for (const r of perSubject) {
      for (const [k, v] of Object.entries(r)) {
        if (k === 'subjectId' || k === 'subjectName' || k === 'region') continue;
        const n = toNumber(v);
        if (n == null) continue;
        set.add(k);
      }
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [perSubject]);

  useEffect(() => {
    if (!metricFields.length) return;
    setXField(prev => (prev && metricFields.includes(prev) ? prev : metricFields[0]));
    setYField(prev => (prev && metricFields.includes(prev) ? prev : metricFields[Math.min(1, metricFields.length - 1)]));
    setGroupMetric(prev => (prev && metricFields.includes(prev) ? prev : metricFields[0]));
    setRegX(prev => (prev && metricFields.includes(prev) ? prev : metricFields[0]));
    setRegY(prev => (prev && metricFields.includes(prev) ? prev : metricFields[Math.min(1, metricFields.length - 1)]));
  }, [metricFields]);

  const groupValues = useMemo(() => {
    if (!groupField) return [] as string[];
    const set = new Set<string>();
    for (const r of perSubject) {
      const v = r[groupField];
      if (v == null) continue;
      const s = String(v).trim();
      if (!s) continue;
      set.add(s);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [perSubject, groupField]);

  const groupFields = useMemo(() => {
    const set = new Set<string>();
    for (const r of perSubject) {
      for (const [k, v] of Object.entries(r)) {
        if (k === 'subjectId' || k === 'subjectName' || k === 'region') continue;
        if (metricFields.includes(k)) continue;
        if (v == null) continue;
        const s = String(v).trim();
        if (!s) continue;
        set.add(k);
      }
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [perSubject, metricFields]);

  const runPearson = async () => {
    setPearson(null);
    setOls(null);
    try {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const r of perSubject) {
        const xv = toNumber(r[xField]);
        const yv = toNumber(r[yField]);
        if (xv == null || yv == null) continue;
        xs.push(xv);
        ys.push(yv);
      }
      if (xs.length < 3) {
        toast.error('Need at least 3 paired samples');
        return;
      }
      const res = await engine.analysisPearson(xs, ys);
      setPearson(res);
    } catch (err) {
      console.error(err);
      toast.error('Failed to compute correlation');
    }
  };

  const runGroupCompare = async () => {
    setGroupCompare(null);
    try {
      const a: number[] = [];
      const b: number[] = [];
      for (const r of perSubject) {
        const g = String(r[groupField] ?? '').trim();
        const yv = toNumber(r[groupMetric]);
        if (!g || yv == null) continue;
        if (g === groupA) a.push(yv);
        if (g === groupB) b.push(yv);
      }
      if (a.length < 2 || b.length < 2) {
        toast.error('Need at least 2 samples per group');
        return;
      }
      const res = await engine.analysisGroupCompare(a, b);
      setGroupCompare(res);
    } catch (err) {
      console.error(err);
      toast.error('Failed to compare groups');
    }
  };

  const runOls = async () => {
    setOls(null);
    setPearson(null);
    try {
      const cols: string[] = ['(Intercept)', regX];
      const covs = [regCov1, regCov2].filter(Boolean);
      for (const c of covs) cols.push(c);

      const y: number[] = [];
      const X: number[][] = [];

      for (const r of perSubject) {
        const yv = toNumber(r[regY]);
        const xv = toNumber(r[regX]);
        if (yv == null || xv == null) continue;

        const row: number[] = [1, xv];
        let ok = true;
        for (const c of covs) {
          const cv = toNumber(r[c]);
          if (cv == null) {
            ok = false;
            break;
          }
          row.push(cv);
        }
        if (!ok) continue;

        y.push(yv);
        X.push(row);
      }

      if (y.length < Math.max(6, X[0]?.length + 2)) {
        toast.error('Not enough samples for regression');
        return;
      }

      const res = await engine.analysisOls(y, X, cols);
      setOls(res);
    } catch (err) {
      console.error(err);
      toast.error('Failed to fit regression');
    }
  };

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
                <h1 className="text-2xl font-medium tracking-tight">Analysis</h1>
                <p className="mono text-xs text-muted-foreground mt-0.5">
                  {project?.name || projectId}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-full p-6 space-y-6">
        <Card className="shadow-sm">
          <CardContent className="p-6 space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label>View</Label>
                <Select value={view} onValueChange={(v) => setView(v as any)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select view" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="total">Total (whole brain / tissue)</SelectItem>
                    <SelectItem value="atlas">Atlas (parcels)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Region</Label>
                <Select value={region} onValueChange={setRegion}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select region" />
                  </SelectTrigger>
                  <SelectContent>
                    {(dataset?.regions || []).map((r) => (
                      <SelectItem key={r} value={r}>{r}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Covariates (CSV/TSV)</Label>
                <Input
                  type="file"
                  accept=".csv,.tsv,.txt,.json"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    try {
                      const text = await f.text();
                      const { rows, headers } = parseDelimited(text);
                      setCovRows(rows);
                      setCovHeaders(headers);
                      toast.success(`Loaded covariates: ${rows.length} rows`);
                    } catch (err) {
                      console.error(err);
                      toast.error('Failed to read covariates file');
                    }
                  }}
                />
                {covHeaders.length ? (
                  <div className="text-xs text-muted-foreground">Columns: {covHeaders.join(', ')}</div>
                ) : null}
              </div>
            </div>

            <div className="text-xs text-muted-foreground">
              Subjects: {new Set((dataset?.rows || []).map(r => r.subjectId)).size} · Records: {(dataset?.rows || []).length}
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-3">
          <Card className="shadow-sm">
            <CardContent className="p-6 space-y-4">
              <div className="text-sm font-medium">Correlation</div>
              <div className="grid gap-3">
                <div className="space-y-2">
                  <Label>X</Label>
                  <Select value={xField} onValueChange={setXField}>
                    <SelectTrigger className="w-full"><SelectValue placeholder="Select X" /></SelectTrigger>
                    <SelectContent>
                      {metricFields.map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Y</Label>
                  <Select value={yField} onValueChange={setYField}>
                    <SelectTrigger className="w-full"><SelectValue placeholder="Select Y" /></SelectTrigger>
                    <SelectContent>
                      {metricFields.map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={runPearson} disabled={!xField || !yField}>Compute</Button>
                {pearson ? (
                  <div className="text-sm">
                    <div>n = {pearson.n}</div>
                    <div>r = {pearson.r.toFixed(4)}</div>
                    <div>p = {pearson.p.toExponential(3)}</div>
                  </div>
                ) : null}
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardContent className="p-6 space-y-4">
              <div className="text-sm font-medium">Group comparison</div>
              <div className="grid gap-3">
                <div className="space-y-2">
                  <Label>Group column</Label>
                  <Select value={groupField} onValueChange={(v) => {
                    setGroupField(v);
                    setGroupA('');
                    setGroupB('');
                    setGroupCompare(null);
                  }}>
                    <SelectTrigger className="w-full"><SelectValue placeholder="Select group" /></SelectTrigger>
                    <SelectContent>
                      {groupFields.map((g) => (
                        <SelectItem key={g} value={g}>{g}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Metric</Label>
                  <Select value={groupMetric} onValueChange={setGroupMetric}>
                    <SelectTrigger className="w-full"><SelectValue placeholder="Select metric" /></SelectTrigger>
                    <SelectContent>
                      {metricFields.map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Group A</Label>
                    <Select value={groupA} onValueChange={setGroupA}>
                      <SelectTrigger className="w-full"><SelectValue placeholder="A" /></SelectTrigger>
                      <SelectContent>
                        {groupValues.map((v) => (
                          <SelectItem key={v} value={v}>{v}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Group B</Label>
                    <Select value={groupB} onValueChange={setGroupB}>
                      <SelectTrigger className="w-full"><SelectValue placeholder="B" /></SelectTrigger>
                      <SelectContent>
                        {groupValues.map((v) => (
                          <SelectItem key={v} value={v}>{v}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Button onClick={runGroupCompare} disabled={!groupField || !groupA || !groupB || !groupMetric}>Compare</Button>

                {groupCompare ? (
                  <div className="text-sm space-y-1">
                    <div>nA = {groupCompare.na} · meanA = {groupCompare.meanA.toFixed(4)}</div>
                    <div>nB = {groupCompare.nb} · meanB = {groupCompare.meanB.toFixed(4)}</div>
                    <div>t-test p = {groupCompare.t_p.toExponential(3)}</div>
                    <div>Mann–Whitney p = {groupCompare.mw_p.toExponential(3)}</div>
                    <div>Cohen's d = {groupCompare.cohen_d.toFixed(3)}</div>
                    {groupCompare.shapiroA_p != null || groupCompare.shapiroB_p != null ? (
                      <div className="text-xs text-muted-foreground">
                        Shapiro p: A {groupCompare.shapiroA_p != null ? groupCompare.shapiroA_p.toExponential(3) : 'n/a'} · B {groupCompare.shapiroB_p != null ? groupCompare.shapiroB_p.toExponential(3) : 'n/a'}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-sm">
            <CardContent className="p-6 space-y-4">
              <div className="text-sm font-medium">Regression (OLS)</div>
              <div className="grid gap-3">
                <div className="space-y-2">
                  <Label>Y</Label>
                  <Select value={regY} onValueChange={setRegY}>
                    <SelectTrigger className="w-full"><SelectValue placeholder="Select Y" /></SelectTrigger>
                    <SelectContent>
                      {metricFields.map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>X</Label>
                  <Select value={regX} onValueChange={setRegX}>
                    <SelectTrigger className="w-full"><SelectValue placeholder="Select X" /></SelectTrigger>
                    <SelectContent>
                      {metricFields.map((m) => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Covariate 1 (optional)</Label>
                    <Select
                      value={regCov1}
                      onValueChange={(v) => setRegCov1(v === NONE_SELECT_VALUE ? '' : v)}
                    >
                      <SelectTrigger className="w-full"><SelectValue placeholder="None" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value={NONE_SELECT_VALUE}>None</SelectItem>
                        {metricFields.map((m) => (
                          <SelectItem key={m} value={m}>{m}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Covariate 2 (optional)</Label>
                    <Select
                      value={regCov2}
                      onValueChange={(v) => setRegCov2(v === NONE_SELECT_VALUE ? '' : v)}
                    >
                      <SelectTrigger className="w-full"><SelectValue placeholder="None" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value={NONE_SELECT_VALUE}>None</SelectItem>
                        {metricFields.map((m) => (
                          <SelectItem key={m} value={m}>{m}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Button onClick={runOls} disabled={!regX || !regY}>Fit</Button>

                {ols ? (
                  <div className="text-sm space-y-1">
                    <div>n = {ols.n} · df = {ols.df_resid} · R² = {ols.r2.toFixed(4)}</div>
                    {ols.residual_shapiro_p != null ? (
                      <div className="text-xs text-muted-foreground">Residual Shapiro p = {ols.residual_shapiro_p.toExponential(3)}</div>
                    ) : null}
                    <div className="mt-2 space-y-1">
                      {ols.coefficients.map((c) => (
                        <div key={c.name} className="flex justify-between gap-3">
                          <span className="mono text-xs">{c.name}</span>
                          <span className="mono text-xs">β={c.beta.toFixed(4)} p={c.p.toExponential(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
