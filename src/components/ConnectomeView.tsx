import { useEffect, useMemo, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table';
import { engine } from '@/lib/engine';
import { getBackendBaseUrl } from '@/lib/backend-engine';
import type { ConnectomeData } from '@/types';

interface ConnectomeViewProps {
  subjectId: string;
}

function fmt(v: any): string {
  if (v == null) return '—';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '—';
    return v.toFixed(4);
  }
  return String(v);
}

type ConnectomeNode = {
  id: number;
  label: number;
  name: string;
};

type ConnectomeEdge = {
  i: number;
  j: number;
  w: number;
};

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;

  const pushField = () => {
    row.push(field);
    field = '';
  };
  const pushRow = () => {
    // Skip empty trailing row.
    if (row.length === 1 && row[0] === '') {
      row = [];
      return;
    }
    rows.push(row);
    row = [];
  };

  const s = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  for (let idx = 0; idx < s.length; idx += 1) {
    const ch = s[idx];
    if (inQuotes) {
      if (ch === '"') {
        const next = s[idx + 1];
        if (next === '"') {
          field += '"';
          idx += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      continue;
    }
    if (ch === ',') {
      pushField();
      continue;
    }
    if (ch === '\n') {
      pushField();
      pushRow();
      continue;
    }
    field += ch;
  }
  pushField();
  pushRow();
  return rows;
}

function buildConnectomeGraph(matrixCsv: string, labelsCsv: string): { nodes: ConnectomeNode[]; edges: ConnectomeEdge[] } {
  const labelsRows = parseCsv(labelsCsv);
  const matrixRows = parseCsv(matrixCsv);

  if (labelsRows.length < 2 || matrixRows.length < 2) {
    return { nodes: [], edges: [] };
  }

  const labelToName = new Map<number, string>();
  const namesByIndex: string[] = [];

  const looksLikeHeader = (cols: string[]): boolean => {
    // If the first row doesn't contain a numeric label anywhere, treat it as a header.
    const candidates = [cols[0], cols[1], cols[2]];
    return !candidates.some(v => Number.isFinite(Number.parseInt(String(v ?? '').trim(), 10)));
  };

  const startRow = labelsRows.length > 0 && looksLikeHeader(labelsRows[0] ?? []) ? 1 : 0;
  for (let r = startRow; r < labelsRows.length; r += 1) {
    const cols = labelsRows[r] ?? [];
    if (cols.length === 0) continue;

    // Common formats:
    // - label,name
    // - name,label
    // - index,label,name (or similar)
    // - name-only (aligned by row order)
    const trimmed = cols.map(c => String(c ?? '').trim());
    const ints = trimmed.map(v => Number.parseInt(v, 10));
    const labelIdx = ints.findIndex(v => Number.isFinite(v));

    if (labelIdx >= 0) {
      const label = ints[labelIdx];
      const nameCandidate = trimmed.find((_, i) => i !== labelIdx && trimmed[i] !== '');
      labelToName.set(label, nameCandidate || String(label));
    } else {
      const name = trimmed.find(v => v !== '') || '';
      if (name) namesByIndex.push(name);
    }
  }

  // Header: label,<label1>,<label2>...
  const header = matrixRows[0] ?? [];
  const labels: number[] = [];
  for (let c = 1; c < header.length; c += 1) {
    const v = Number.parseInt(String(header[c] ?? '').trim(), 10);
    if (Number.isFinite(v)) labels.push(v);
  }

  const nodes: ConnectomeNode[] = labels.map((lbl, idx) => {
    const nameFromLabel = labelToName.get(lbl);
    const nameFromIndex = namesByIndex[idx];
    const name = (nameFromLabel && nameFromLabel.trim()) || (nameFromIndex && nameFromIndex.trim()) || String(lbl);
    return {
      id: idx,
      label: lbl,
      name,
    };
  });

  const n = nodes.length;
  if (!n) return { nodes: [], edges: [] };

  // Parse matrix values.
  const edges: ConnectomeEdge[] = [];
  for (let r = 1; r < matrixRows.length && r <= n; r += 1) {
    const cols = matrixRows[r] ?? [];
    for (let c = r + 1; c <= n; c += 1) {
      const raw = cols[c] ?? '';
      const w = Number.parseFloat(String(raw).trim());
      if (!Number.isFinite(w) || w <= 0) continue;
      edges.push({ i: r - 1, j: c - 1, w });
    }
  }

  edges.sort((a, b) => b.w - a.w);
  // Cap edges for legibility & performance.
  const maxEdges = Math.min(350, edges.length);
  return { nodes, edges: edges.slice(0, maxEdges) };
}

function InteractiveConnectome({ matrixUrl, labelsUrl }: { matrixUrl: string; labelsUrl: string }) {
  const [nodes, setNodes] = useState<ConnectomeNode[]>([]);
  const [edges, setEdges] = useState<ConnectomeEdge[]>([]);
  const [vizError, setVizError] = useState<string | null>(null);
  const [hoverNode, setHoverNode] = useState<number | null>(null);
  const [hoverEdge, setHoverEdge] = useState<number | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setVizError(null);
    setNodes([]);
    setEdges([]);

    (async () => {
      try {
        const [mRes, lRes] = await Promise.all([
          fetch(matrixUrl, { signal: ac.signal }),
          fetch(labelsUrl, { signal: ac.signal }),
        ]);
        if (!mRes.ok) throw new Error(`Failed to fetch matrix (${mRes.status})`);
        if (!lRes.ok) throw new Error(`Failed to fetch labels (${lRes.status})`);
        const [mText, lText] = await Promise.all([mRes.text(), lRes.text()]);
        const g = buildConnectomeGraph(mText, lText);
        setNodes(g.nodes);
        setEdges(g.edges);
      } catch (e: any) {
        if (ac.signal.aborted) return;
        setVizError(typeof e?.message === 'string' ? e.message : 'Failed to load connectome graph.');
      }
    })();

    return () => {
      ac.abort();
    };
  }, [matrixUrl, labelsUrl]);

  const strength = useMemo(() => {
    const out = new Array(nodes.length).fill(0) as number[];
    for (const e of edges) {
      out[e.i] += e.w;
      out[e.j] += e.w;
    }
    return out;
  }, [nodes.length, edges]);

  const activeNode = selectedNode != null ? selectedNode : hoverNode;

  // Make the visualization larger for readability.
  const viewSize = 840;
  const r = 310;
  const labelR = r + 34;

  const positions = useMemo(() => {
    const n = nodes.length;
    if (!n) return [] as Array<{ x: number; y: number; a: number }>;
    const out: Array<{ x: number; y: number; a: number }> = [];
    for (let i = 0; i < n; i += 1) {
      const a = (Math.PI / 2) - (2 * Math.PI * i) / n;
      out.push({ x: r * Math.cos(a), y: r * Math.sin(a), a });
    }
    return out;
  }, [nodes.length]);

  const maxW = useMemo(() => {
    let m = 0;
    for (const e of edges) m = Math.max(m, e.w);
    return m > 0 ? m : 1;
  }, [edges]);

  const maxStrength = useMemo(() => {
    let m = 0;
    for (const s of strength) m = Math.max(m, s);
    return m > 0 ? m : 1;
  }, [strength]);

  const edgeColorFor = (wn: number): string => {
    // Uses existing theme variables (see `src/index.css`) to avoid introducing new colors.
    if (wn >= 0.85) return 'var(--accent)';
    if (wn >= 0.6) return 'var(--primary)';
    if (wn >= 0.35) return 'var(--success)';
    if (wn >= 0.15) return 'var(--warning)';
    return 'var(--muted-foreground)';
  };

  const nodeColorFor = (sn: number): string => {
    if (sn >= 0.85) return 'var(--accent)';
    if (sn >= 0.6) return 'var(--primary)';
    if (sn >= 0.35) return 'var(--success)';
    if (sn >= 0.15) return 'var(--warning)';
    return 'var(--muted-foreground)';
  };

  const labelStep = useMemo(() => {
    // Keep labels readable for large atlases.
    const n = nodes.length;
    if (n <= 80) return 1;
    if (n <= 140) return 2;
    return Math.ceil(n / 70);
  }, [nodes.length]);

  const labelText = (name: string): string => {
    const s = String(name ?? '').trim();
    if (!s) return '';
    return s.length > 16 ? `${s.slice(0, 14)}…` : s;
  };

  return (
    <div className="mb-4">
      <div className="text-xs text-muted-foreground mb-2">
        Interactive connectome (edge/node color indicates strength; hover/click nodes)
      </div>
      <div className="rounded-md border bg-background overflow-auto">
        {vizError ? (
          <div className="p-3 text-sm text-destructive">{vizError}</div>
        ) : nodes.length === 0 ? (
          <div className="p-3 text-sm text-muted-foreground">Loading connectome graph…</div>
        ) : (
          <svg
            viewBox={`${-viewSize / 2} ${-viewSize / 2} ${viewSize} ${viewSize}`}
            className="w-full h-[720px]"
            role="img"
            aria-label="Interactive connectome"
            preserveAspectRatio="xMidYMid meet"
            onMouseLeave={() => {
              setHoverNode(null);
              setHoverEdge(null);
            }}
          >
            {/* Edges */}
            {edges.map((e, idx) => {
              const pi = positions[e.i];
              const pj = positions[e.j];
              if (!pi || !pj) return null;
              const wn = Math.max(0, Math.min(1, e.w / maxW));

              const incident = activeNode != null && (e.i === activeNode || e.j === activeNode);
              const isHoverEdge = hoverEdge === idx;

              const opacity = activeNode != null ? (incident ? 0.65 : 0.08) : 0.12 + 0.38 * wn;
              const strokeWidth = (activeNode != null ? (incident ? 2.4 : 1.0) : 1.2 + 3.6 * wn) + (isHoverEdge ? 1.0 : 0);
              const d = `M ${pi.x.toFixed(2)} ${pi.y.toFixed(2)} Q 0 0 ${pj.x.toFixed(2)} ${pj.y.toFixed(2)}`;
              const stroke = edgeColorFor(wn);

              return (
                <path
                  key={`${e.i}-${e.j}-${idx}`}
                  d={d}
                  fill="none"
                  stroke={stroke}
                  strokeOpacity={opacity}
                  strokeWidth={strokeWidth}
                  onMouseEnter={() => {
                    setHoverEdge(idx);
                    setHoverNode(null);
                  }}
                >
                  <title>
                    {nodes[e.i]?.name} ↔ {nodes[e.j]?.name} (w={e.w.toFixed(0)})
                  </title>
                </path>
              );
            })}

            {/* Labels */}
            {nodes.map((n, idx) => {
              const p = positions[idx];
              if (!p) return null;
              const isActive = activeNode === idx;
              const show = isActive || idx % labelStep === 0;
              if (!show) return null;

              const x = labelR * Math.cos(p.a);
              const y = labelR * Math.sin(p.a);
              const rightSide = Math.cos(p.a) >= 0;
              const rotate = (p.a * 180) / Math.PI;
              const textRotate = rightSide ? rotate : rotate + 180;
              const anchor: 'start' | 'end' = rightSide ? 'start' : 'end';
              const fill = isActive ? 'var(--foreground)' : 'var(--muted-foreground)';
              const fillOpacity = isActive ? 0.95 : 0.65;
              const label = labelText(n.name);
              if (!label) return null;

              return (
                <text
                  key={`${n.id}-label`}
                  x={x}
                  y={y}
                  textAnchor={anchor}
                  dominantBaseline="middle"
                  fontSize={isActive ? 12 : 10}
                  fill={fill}
                  fillOpacity={fillOpacity}
                  transform={`rotate(${textRotate.toFixed(2)} ${x.toFixed(2)} ${y.toFixed(2)})`}
                  style={{ userSelect: 'none', pointerEvents: 'none' }}
                >
                  <title>{n.name}</title>
                  {label}
                </text>
              );
            })}

            {/* Nodes */}
            {nodes.map((n, idx) => {
              const p = positions[idx];
              if (!p) return null;
              const isActive = activeNode === idx;
              const isDim = activeNode != null && !isActive;
              const sn = Math.max(0, Math.min(1, (strength[idx] ?? 0) / maxStrength));
              const radius = isActive ? 8.2 : 5.6;
              const fillOpacity = isDim ? 0.35 : 0.9;
              const fill = isActive ? 'var(--foreground)' : nodeColorFor(sn);
              return (
                <circle
                  key={n.id}
                  cx={p.x}
                  cy={p.y}
                  r={radius}
                  fill={fill}
                  fillOpacity={fillOpacity}
                  onMouseEnter={() => {
                    setHoverNode(idx);
                    setHoverEdge(null);
                  }}
                  onClick={() => {
                    setSelectedNode(prev => (prev === idx ? null : idx));
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <title>
                    {n.name} (strength={strength[idx]?.toFixed(0) ?? '0'})
                  </title>
                </circle>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}

export function ConnectomeView({ subjectId }: ConnectomeViewProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ConnectomeData | null>(null);
  const [running, setRunning] = useState(false);

  const pollTimerRef = useRef<number | null>(null);
  const unmountedRef = useRef(false);

  const clearPoll = () => {
    if (pollTimerRef.current != null) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const load = async () => {
    const payload = await engine.getSubjectConnectome(subjectId);
    if (unmountedRef.current) return;
    setData(payload);
    return payload;
  };

  useEffect(() => {
    unmountedRef.current = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        await load();
      } catch (e: any) {
        if (unmountedRef.current) return;
        setData(null);
        setError(typeof e?.message === 'string' ? e.message : 'Failed to load connectome.');
      } finally {
        if (!unmountedRef.current) setLoading(false);
      }
    })();

    return () => {
      unmountedRef.current = true;
      clearPoll();
    };
  }, [subjectId]);

  const metrics = data?.metrics ?? null;

  const rows = useMemo(() => {
    const smallWorld = metrics?.small_worldness;
    const sigma = smallWorld && typeof smallWorld === 'object' ? smallWorld.sigma : null;

    return [
      { label: 'Density', value: metrics?.density },
      { label: 'Clustering Coefficient', value: metrics?.clustering_coefficient },
      { label: 'Transitivity', value: metrics?.transitivity },
      { label: 'Characteristic Path Length', value: metrics?.characteristic_path_length },
      { label: 'Small-worldness (σ)', value: sigma },
      { label: 'Global Efficiency', value: metrics?.global_efficiency },
      { label: 'Local Efficiency', value: metrics?.local_efficiency },
      { label: 'Assortativity Coefficient', value: metrics?.assortativity_coefficient },
    ];
  }, [metrics]);

  const baseUrl = useMemo(() => {
    try {
      return getBackendBaseUrl();
    } catch {
      return null;
    }
  }, []);

  const downloadUrl = (kind: 'matrix' | 'labels' | 'metrics' | 'image'): string | null => {
    if (!baseUrl) return null;
    return `${baseUrl}/subjects/${encodeURIComponent(subjectId)}/connectome/file?kind=${encodeURIComponent(kind)}`;
  };

  const runConnectome = async () => {
    if (running) return;
    clearPoll();
    setRunning(true);
    setError(null);

    try {
      await engine.runSubjectStage(subjectId, 'connectome', { runDependencies: false });
      

      // Poll for artifacts to appear.
      let attempts = 0;
      const tick = async () => {
        if (unmountedRef.current) return;
        attempts += 1;
        try {
          const payload = await load();
          if (payload?.available) {
            setRunning(false);
            clearPoll();
            return;
          }
        } catch {
          // ignore; next poll attempt may succeed
        }
        if (unmountedRef.current) return;
        if (attempts >= 90) {
          setRunning(false);
          clearPoll();
          return;
        }
        pollTimerRef.current = window.setTimeout(() => void tick(), 2000);
      };

      pollTimerRef.current = window.setTimeout(() => void tick(), 1200);
    } catch (e: any) {
      setError(typeof e?.message === 'string' ? e.message : 'Failed to run connectome.');
      setRunning(false);
    }
  };

  return (
    <div className="space-y-5">
      <Card className="border-0 shadow-sm">
        <div className="p-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
                Network Connectome
              </h2>
              <p className="text-xs text-muted-foreground mt-1">
                Structural connectivity derived from tractography streamlines between atlas parcels.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Button type="button" size="sm" onClick={runConnectome} disabled={running}>
                {running ? 'Running…' : 'Run connectome'}
              </Button>
              {downloadUrl('metrics') ? (
                <Button asChild variant="outline" size="sm">
                  <a href={downloadUrl('metrics') as string} target="_blank" rel="noreferrer">
                    Download metrics
                  </a>
                </Button>
              ) : null}
              {downloadUrl('matrix') ? (
                <Button asChild variant="outline" size="sm">
                  <a href={downloadUrl('matrix') as string} target="_blank" rel="noreferrer">
                    Download matrix
                  </a>
                </Button>
              ) : null}
              {downloadUrl('labels') ? (
                <Button asChild variant="outline" size="sm">
                  <a href={downloadUrl('labels') as string} target="_blank" rel="noreferrer">
                    Download labels
                  </a>
                </Button>
              ) : null}
            </div>
          </div>

          <div className="mt-4">
            {loading ? (
              <div className="text-sm text-muted-foreground">Loading…</div>
            ) : error ? (
              <div className="text-sm text-destructive">{error}</div>
            ) : data && !data.available ? (
              <div className="text-sm text-muted-foreground">
                No connectome outputs found. Run connectome (requires Tractography outputs).
              </div>
            ) : null}

            {data?.available && metrics ? (
              <>
                {downloadUrl('matrix') && downloadUrl('labels') ? (
                  <InteractiveConnectome
                    matrixUrl={downloadUrl('matrix') as string}
                    labelsUrl={downloadUrl('labels') as string}
                  />
                ) : null}

                <Table>
                  <TableBody>
                    {rows.map(r => (
                      <TableRow key={r.label}>
                        <TableCell className="font-medium">{r.label}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.value)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            ) : null}

            {data?.error ? (
              <div className="text-xs text-muted-foreground mt-3">{data.error}</div>
            ) : null}
          </div>
        </div>
      </Card>
    </div>
  );
}
