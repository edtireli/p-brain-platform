import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface MapsViewProps {
  subjectId: string;
}

export function MapsView({ subjectId }: MapsViewProps) {
  const maps = [
    { name: 'Ki Map', unit: 'ml/100g/min', available: true },
    { name: 'vp Map', unit: 'fraction', available: true },
    { name: 'Ktrans Map', unit: 'min⁻¹', available: true },
    { name: 've Map', unit: 'fraction', available: true },
    { name: 'CBF Map', unit: 'ml/100g/min', available: true },
    { name: 'MTT Map', unit: 's', available: true },
    { name: 'CTH Map', unit: 's', available: true },
    { name: 'FA Map', unit: 'fraction', available: false },
    { name: 'MD Map', unit: 'mm²/s', available: false },
  ];

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h2 className="mb-4 text-lg font-semibold">Parameter Maps</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {maps.map(map => (
            <div
              key={map.name}
              className={`rounded-lg border ${
                map.available ? 'border-border bg-card' : 'border-dashed border-muted-foreground/30 bg-muted/20'
              } p-6`}
            >
              <div className="mb-2 flex items-center justify-between">
                <h3 className="font-semibold">{map.name}</h3>
                {map.available ? (
                  <Badge variant="default" className="bg-success text-success-foreground">
                    Ready
                  </Badge>
                ) : (
                  <Badge variant="secondary">Not Available</Badge>
                )}
              </div>
              <div className="mono text-xs text-muted-foreground">{map.unit}</div>
              {map.available && (
                <div className="mt-4 flex h-32 items-center justify-center rounded bg-muted/50 text-sm text-muted-foreground">
                  [Interactive map viewer would render here]
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-6">
        <h2 className="mb-4 text-lg font-semibold">Map Controls</h2>
        <div className="space-y-4 text-sm text-muted-foreground">
          <p>
            In a full implementation, this view would provide:
          </p>
          <ul className="ml-6 list-disc space-y-2">
            <li>Interactive 3D volume rendering of parameter maps</li>
            <li>Slice-by-slice navigation with colormap customization</li>
            <li>Overlay capabilities with anatomical segmentation</li>
            <li>Histogram visualization for distribution analysis</li>
            <li>Region-of-interest statistics extraction</li>
            <li>Export functionality for individual maps</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}
