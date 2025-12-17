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
    loadData();
  }, [subjectId]);

  const loadData = async () => {
    try {
      const data = await mockEngine.getMetricsTable(subjectId);
      setMetricsTable(data);
    } catch (error) {
      console.error('Failed to load metrics table:', error);
    }
  };

  const handleExport = () => {
    if (!metricsTable) return;

    const headers = ['Region', 'Ki (ml/100g/min)', 'vp', 'Ktrans (min⁻¹)', 've', 'CBF (ml/100g/min)', 'MTT (s)', 'CTH (s)'];
    const rows = metricsTable.rows.map(row => [
      row.region,
      row.Ki?.toFixed(2) || 'N/A',
      row.vp?.toFixed(3) || 'N/A',
      row.Ktrans?.toFixed(3) || 'N/A',
      row.ve?.toFixed(3) || 'N/A',
      row.CBF?.toFixed(1) || 'N/A',
      row.MTT?.toFixed(2) || 'N/A',
      row.CTH?.toFixed(2) || 'N/A',
    ]);

    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metrics_${subjectId}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Quantitative Metrics by Region</h2>
          <Button onClick={handleExport} variant="secondary" className="gap-2" disabled={!metricsTable}>
            <Download size={18} weight="bold" />
            Export CSV
          </Button>
        </div>

        {metricsTable ? (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="font-semibold">Region</TableHead>
                  <TableHead className="text-right">
                    <div>Ki</div>
                    <div className="mono text-xs font-normal text-muted-foreground">ml/100g/min</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div>vp</div>
                    <div className="mono text-xs font-normal text-muted-foreground">fraction</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div>Ktrans</div>
                    <div className="mono text-xs font-normal text-muted-foreground">min⁻¹</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div>ve</div>
                    <div className="mono text-xs font-normal text-muted-foreground">fraction</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div>CBF</div>
                    <div className="mono text-xs font-normal text-muted-foreground">ml/100g/min</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div>MTT</div>
                    <div className="mono text-xs font-normal text-muted-foreground">s</div>
                  </TableHead>
                  <TableHead className="text-right">
                    <div>CTH</div>
                    <div className="mono text-xs font-normal text-muted-foreground">s</div>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metricsTable.rows.map(row => (
                  <TableRow key={row.region}>
                    <TableCell className="font-medium">{row.region}</TableCell>
                    <TableCell className="mono text-right">{row.Ki?.toFixed(2) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.vp?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.Ktrans?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.ve?.toFixed(3) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.CBF?.toFixed(1) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.MTT?.toFixed(2) || 'N/A'}</TableCell>
                    <TableCell className="mono text-right">{row.CTH?.toFixed(2) || 'N/A'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            No metrics data available
          </div>
        )}
      </Card>

      <Card className="p-6">
        <h2 className="mb-4 text-lg font-semibold">About These Metrics</h2>
        <div className="space-y-4 text-sm text-muted-foreground">
          <div>
            <span className="mono font-semibold text-foreground">Ki</span> - Transfer constant from Patlak analysis,
            quantifying unidirectional transport across the blood-brain barrier
          </div>
          <div>
            <span className="mono font-semibold text-foreground">vp</span> - Plasma volume fraction, representing the
            fractional blood volume in tissue
          </div>
          <div>
            <span className="mono font-semibold text-foreground">Ktrans</span> - Volume transfer constant from Extended
            Tofts model, indicating vascular permeability
          </div>
          <div>
            <span className="mono font-semibold text-foreground">ve</span> - Extravascular extracellular volume fraction
          </div>
          <div>
            <span className="mono font-semibold text-foreground">CBF</span> - Cerebral blood flow from deconvolution
            analysis
          </div>
          <div>
            <span className="mono font-semibold text-foreground">MTT</span> - Mean transit time, average time for blood
            to traverse the capillary network
          </div>
          <div>
            <span className="mono font-semibold text-foreground">CTH</span> - Capillary transit-time heterogeneity,
            standard deviation of the transit-time distribution
          </div>
        </div>
      </Card>
    </div>
  );
}
