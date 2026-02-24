import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Download } from '@phosphor-icons/react';
import { engine } from '@/lib/engine';
import { motion } from 'framer-motion';
import type { MetricsTable } from '@/types';

interface TablesViewProps {
  subjectId: string;
}

const metricDescriptions: Record<string, string> = {
  Ki: 'Transfer constant from Patlak analysis — unidirectional transport rate across blood-brain barrier',
  vp: 'Plasma volume fraction from Patlak analysis',
  CBF: 'Cerebral blood flow from deconvolution analysis',
  MTT: 'Mean transit time — average time for blood to traverse capillary network',
  CTH: 'Capillary transit-time heterogeneity — standard deviation of transit-time distribution',
};

function MetricHeader({ label, unit }: { label: string; unit: string }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="cursor-help">
            <div>{label}</div>
            <div className="mono text-xs font-normal text-muted-foreground">{unit}</div>
          </div>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-[200px]">
          <p className="text-xs">{metricDescriptions[label]}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function TablesView({ subjectId }: TablesViewProps) {
  const [metricsTable, setMetricsTable] = useState<MetricsTable | null>(null);
  const [view, setView] = useState<'atlas' | 'tissue'>('atlas');

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await engine.getMetricsTable(subjectId, view);
        setMetricsTable(data);
      } catch (error) {
        console.error('Failed to load metrics table:', error);
      }
    };
    loadData();
  }, [subjectId, view]);

  const handleExport = () => {
    if (!metricsTable) return;

    const headers = ['Region', 'Ki (ml/100g/min)', 'vp', 'CBF (ml/100g/min)', 'MTT (s)', 'CTH (s)'];
    const rows = metricsTable.rows.map(row => [
      row.region,
      row.Ki?.toFixed(2) || 'N/A',
      row.vp?.toFixed(3) || 'N/A',
      row.CBF?.toFixed(1) || 'N/A',
      row.MTT?.toFixed(2) || 'N/A',
      row.CTH?.toFixed(2) || 'N/A',
    ]);

    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metrics_${view}_${subjectId}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <Card className="rounded-xl border border-border/60 bg-card/50 p-6">
        <div className="mb-4 flex items-center justify-between">
          <div className="space-y-1">
            <h3 className="text-[13px] font-medium">Quantitative Metrics by Region</h3>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant={view === 'atlas' ? 'secondary' : 'ghost'}
                className="h-7 px-2 text-xs"
                onClick={() => setView('atlas')}
              >
                Atlas
              </Button>
              <Button
                type="button"
                size="sm"
                variant={view === 'tissue' ? 'secondary' : 'ghost'}
                className="h-7 px-2 text-xs"
                onClick={() => setView('tissue')}
              >
                Tissue
              </Button>
            </div>
          </div>

          <Button
            onClick={handleExport}
            variant="ghost"
            size="sm"
            className="gap-2 text-xs"
            disabled={!metricsTable}
          >
            <Download size={14} weight="bold" />
            Export CSV
          </Button>
        </div>

        {metricsTable ? (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="text-left">Region</TableHead>
                  <TableHead className="text-right">
                    <MetricHeader label="Ki" unit="ml/100g/min" />
                  </TableHead>
                  <TableHead className="text-right">
                    <MetricHeader label="vp" unit="fraction" />
                  </TableHead>
                  <TableHead className="text-right">
                    <MetricHeader label="CBF" unit="ml/100g/min" />
                  </TableHead>
                  <TableHead className="text-right">
                    <MetricHeader label="MTT" unit="s" />
                  </TableHead>
                  <TableHead className="text-right">
                    <MetricHeader label="CTH" unit="s" />
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metricsTable.rows.map((row, idx) => (
                  <motion.tr
                    key={row.region}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.03 }}
                    className="border-b border-border/40 hover:bg-muted/30"
                  >
                    <TableCell className="font-medium text-[12px]">{row.region}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.Ki?.toFixed(2) || '—'}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.vp?.toFixed(3) || '—'}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.CBF?.toFixed(1) || '—'}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.MTT?.toFixed(2) || '—'}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.CTH?.toFixed(2) || '—'}</TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <div className="flex h-[300px] items-center justify-center text-muted-foreground text-sm">
            No metrics data available
          </div>
        )}
      </Card>

      <Card className="rounded-xl border border-border/60 bg-card/50 p-6">
        <h3 className="mb-4 text-[13px] font-medium">Metric Definitions</h3>
        <div className="grid gap-3 text-xs text-muted-foreground">
          <div>
            <span className="mono font-semibold text-foreground">Ki</span> — Transfer constant from Patlak analysis,
            quantifying unidirectional transport across the blood-brain barrier
          </div>
          <div>
            <span className="mono font-semibold text-foreground">vp</span> — Plasma volume fraction representing the
            vascular space within tissue
          </div>
          <div>
            <span className="mono font-semibold text-foreground">CBF</span> — Cerebral blood flow from deconvolution
            analysis
          </div>
          <div>
            <span className="mono font-semibold text-foreground">MTT</span> — Mean transit time for blood to traverse
            the capillary network
          </div>
          <div>
            <span className="mono font-semibold text-foreground">CTH</span> — Capillary transit-time heterogeneity,
            standard deviation of the transit-time distribution
          </div>
        </div>
      </Card>
    </div>
  );
}
