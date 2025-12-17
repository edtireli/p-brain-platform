import { useState, useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { mockEngine } from '@/lib/mock-engine';

interface VolumeViewerProps {
  subjectId: string;
}

export function VolumeViewer({ subjectId }: VolumeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sliceZ, setSliceZ] = useState(32);
  const [timeFrame, setTimeFrame] = useState(0);
  const [windowMin, setWindowMin] = useState(0);
  const [windowMax, setWindowMax] = useState(2000);
  const [colormap, setColormap] = useState('grayscale');
  const [overlayAlpha, setOverlayAlpha] = useState(0.5);
  const [showOverlay, setShowOverlay] = useState(false);
  const [sliceData, setSliceData] = useState<number[][] | null>(null);

  useEffect(() => {
    loadSlice();
  }, [subjectId, sliceZ, timeFrame]);

  useEffect(() => {
    if (sliceData) {
      renderSlice();
    }
  }, [sliceData, windowMin, windowMax, colormap, overlayAlpha, showOverlay]);

  const loadSlice = async () => {
    try {
      const data = await mockEngine.getSliceData('/mock/dce.nii.gz', sliceZ, timeFrame);
      setSliceData(data.data);
    } catch (error) {
      console.error('Failed to load slice:', error);
    }
  };

  const renderSlice = () => {
    if (!canvasRef.current || !sliceData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = sliceData.length;
    canvas.width = size;
    canvas.height = size;

    const imageData = ctx.createImageData(size, size);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const value = sliceData[y][x];
        const normalized = Math.max(0, Math.min(1, (value - windowMin) / (windowMax - windowMin)));
        
        const idx = (y * size + x) * 4;
        
        if (colormap === 'grayscale') {
          const gray = Math.floor(normalized * 255);
          imageData.data[idx] = gray;
          imageData.data[idx + 1] = gray;
          imageData.data[idx + 2] = gray;
          imageData.data[idx + 3] = 255;
        } else if (colormap === 'viridis') {
          const viridisColor = getViridisColor(normalized);
          imageData.data[idx] = viridisColor[0];
          imageData.data[idx + 1] = viridisColor[1];
          imageData.data[idx + 2] = viridisColor[2];
          imageData.data[idx + 3] = 255;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);

    if (showOverlay) {
      ctx.fillStyle = `rgba(255, 100, 100, ${overlayAlpha})`;
      const centerX = size / 2;
      const centerY = size / 2;
      const radius = size * 0.2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  const getViridisColor = (t: number): [number, number, number] => {
    const viridis = [
      [68, 1, 84],
      [72, 40, 120],
      [62, 73, 137],
      [49, 104, 142],
      [38, 130, 142],
      [31, 158, 137],
      [53, 183, 121],
      [109, 205, 89],
      [180, 222, 44],
      [253, 231, 37],
    ];

    const idx = Math.floor(t * (viridis.length - 1));
    const nextIdx = Math.min(idx + 1, viridis.length - 1);
    const localT = (t * (viridis.length - 1)) - idx;

    const r = viridis[idx][0] + (viridis[nextIdx][0] - viridis[idx][0]) * localT;
    const g = viridis[idx][1] + (viridis[nextIdx][1] - viridis[idx][1]) * localT;
    const b = viridis[idx][2] + (viridis[nextIdx][2] - viridis[idx][2]) * localT;

    return [Math.floor(r), Math.floor(g), Math.floor(b)];
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
      <Card className="flex items-center justify-center bg-muted/30 p-8">
        <canvas
          ref={canvasRef}
          className="max-h-[600px] max-w-full rounded-lg border border-border shadow-lg"
          style={{ imageRendering: 'pixelated' }}
        />
      </Card>

      <Card className="p-6">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label>Slice (Z): {sliceZ}</Label>
            <Slider
              value={[sliceZ]}
              onValueChange={([value]) => setSliceZ(value)}
              min={0}
              max={63}
              step={1}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label>Time Frame (t): {timeFrame}</Label>
            <Slider
              value={[timeFrame]}
              onValueChange={([value]) => setTimeFrame(value)}
              min={0}
              max={79}
              step={1}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label>Window Min: {windowMin}</Label>
            <Slider
              value={[windowMin]}
              onValueChange={([value]) => setWindowMin(value)}
              min={0}
              max={2000}
              step={10}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label>Window Max: {windowMax}</Label>
            <Slider
              value={[windowMax]}
              onValueChange={([value]) => setWindowMax(value)}
              min={0}
              max={4000}
              step={10}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label>Colormap</Label>
            <Select value={colormap} onValueChange={setColormap}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="grayscale">Grayscale</SelectItem>
                <SelectItem value="viridis">Viridis</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Overlay</Label>
              <button
                onClick={() => setShowOverlay(!showOverlay)}
                className={`rounded px-3 py-1 text-sm font-medium transition-colors ${
                  showOverlay ? 'bg-accent text-accent-foreground' : 'bg-muted text-muted-foreground'
                }`}
              >
                {showOverlay ? 'ON' : 'OFF'}
              </button>
            </div>
            {showOverlay && (
              <div className="space-y-2">
                <Label>Overlay Alpha: {overlayAlpha.toFixed(2)}</Label>
                <Slider
                  value={[overlayAlpha]}
                  onValueChange={([value]) => setOverlayAlpha(value)}
                  min={0}
                  max={1}
                  step={0.05}
                  className="w-full"
                />
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
