import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Download } from '@phosphor-icons/react';
import { mockEngine } from '@/lib/mock-engine';
import type { MetricsTable } from '@/types';

interface TablesViewProps {
  subjectId: string;
}

export function TablesView({ subjectId }: TablesViewProps) {
  const [metricsTable, setMetricsTable] = useState<MetricsTable | null>(null);

  useEffect(() => {
  Ktrans: 'Volume transfer constant from Extended Tofts — indicates vascular permeability',
  }, [subjectId]);

  const loadData = async () => {
    try {
      const data = await mockEngine.getMetricsTable(subjectId);
      setMetricsTable(data);
export function TablesView({ subjectId }: TablesViewProps) {
    }
  };
  useEffect(() => {
    loadData();
    if (!metricsTable) return;

    const headers = ['Region', 'Ki (ml/100g/min)', 'vp', 'Ktrans (min⁻¹)', 've', 'CBF (ml/100g/min)', 'MTT (s)', 'CTH (s)'];
    const rows = metricsTable.rows.map(row => [
      row.region,
      row.Ki?.toFixed(2) || 'N/A',
      row.vp?.toFixed(3) || 'N/A',
      console.error('Failed to load metrics table:', error);
      row.ve?.toFixed(3) || 'N/A',
  };
      row.MTT?.toFixed(2) || 'N/A',
      row.CTH?.toFixed(2) || 'N/A',
    ]);

    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metrics_${subjectId}_${Date.now()}.csv`;
    URL.revokeObjectURL(url);
      row.CBF?.toFixed(1) || 'N/A',

  return (
    ]);
me="p-6">
    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
          <Button onClick={handleExport} variant="secondary" className="gap-2" disabled={!metricsTable}>
            <Download size={18} weight="bold" />
            Export CSV
          </Button>
        {metricsTable ? (
          <div className="overflow-x-auto">
  };
              <TableHeader>
                <TableRow>
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
                    <div className="mono text-xs font-normal text-muted-foreground">ml/100g/min</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div className="mono text-xs font-normal text-muted-foreground">fraction</div>
        </TooltipTrigger>
                  <TableHead className="text-right">
          <p className="text-xs">{metricDescriptions[label]}</p>
                    <div className="mono text-xs font-normal text-muted-foreground">min⁻¹</div>
        </TooltipContent>
                    <div>ve</div>
    </TooltipProvider>nt-normal text-muted-foreground">fraction</div>
                  </TableHead>
me="text-right">
  return (v>
    <div className="space-y-6">l text-muted-foreground">ml/100g/min</div>
      <div className="rounded-xl border border-border/60 bg-card/50">
                  <TableHead className="text-right">
          <h3 className="text-[13px] font-medium">Quantitative Metrics by Region</h3>
          <Button font-normal text-muted-foreground">s</div>
            onClick={handleExport} 
            variant="ghost" 
                    <div>CTH</div>
                    <div className="mono text-xs font-normal text-muted-foreground">s</div>
            disabled={!metricsTable}
          >
              </TableHeader>
            Export CSV
                {metricsTable.rows.map(row => (
                  <TableRow key={row.region}>
           <TableCell className="font-medium">{row.region}</TableCell>
        {metricsTable ? ("mono text-right">{row.Ki?.toFixed(2) || 'N/A'}</TableCell>
          <div className="overflow-x-auto">p?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.Ktrans?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.ve?.toFixed(3) || 'N/A'}</TableCell>
                <TableRow className="hover:bg-transparent">.CBF?.toFixed(1) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.CTH?.toFixed(2) || 'N/A'}</TableCell>
                  <TableHead className="text-right">
                ))}
              </TableBody>
                  <TableHead className="text-right">
          </div>
                  </TableHead>
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            No metrics data available
          </div>
      </Card>
            <span className="mono font-semibold text-foreground">Ki</span> - Transfer constant from Patlak analysis,
            quantifying unidirectional transport across the blood-brain barrier
                  <TableHead className="text-right">
          <div>
                  </TableHead>epresenting the
                  <TableHead className="text-right">
          </div>
                  </TableHead>
                  <TableHead className="text-right">transfer constant from Extended
                    <MetricHeader label="CTH" unit="s" />
                  </TableHead>
          <div>
              </TableHeader>
              <TableBody>
                {metricsTable.rows.map((row, idx) => (
                  <motion.trnvolution
                    key={row.region}
          </div>
          <div>
                    transition={{ delay: idx * 0.03 }}
            to traverse the capillary network
                  >
          <div>
            <span className="mono font-semibold text-foreground">CTH</span> - Capillary transit-time heterogeneity,
            standard deviation of the transit-time distribution
          </div>
        </div>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.CBF?.toFixed(1) || '—'}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.MTT?.toFixed(2) || '—'}</TableCell>
                    <TableCell className="mono text-right text-[12px] tabular-nums">{row.CTH?.toFixed(2) || '—'}</TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </div>
        )}    </div>