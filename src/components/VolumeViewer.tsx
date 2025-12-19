import { useState, useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { mockEngine } from '@/lib/mock-engine';
import type { VolumeFile } from '@/types';

interface VolumeViewerProps {
  subjectId: string;
  path?: string;
  kind?: 'dce' | 't1' | 'diffusion';
  allowSelect?: boolean;
}

export function VolumeViewer({ subjectId, path, kind = 'dce', allowSelect = true }: VolumeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [volumePath, setVolumePath] = useState<string | null>(null);
  const [volumes, setVolumes] = useState<VolumeFile[]>([]);
  const [selectedVolumeId, setSelectedVolumeId] = useState<string>('');
  const [maxZ, setMaxZ] = useState(63);
  const [maxT, setMaxT] = useState(79);
  const [sliceZ, setSliceZ] = useState(32);
  const [timeFrame, setTimeFrame] = useState(0);
  const [windowMin, setWindowMin] = useState(0);
  const [windowMax, setWindowMax] = useState(2000);
  const [colormap, setColormap] = useState('grayscale');
  const [viewMode, setViewMode] = useState<'single' | 'grid'>('single');
  const [slices, setSlices] = useState<number[][][] | null>(null);

  const isImagePath = (p: string | null): boolean => {
    if (!p) return false;
    const base = p.split('?')[0];
    return /\.(png|jpe?g|webp|gif)$/i.test(base);
  };

  const isImage = isImagePath(volumePath);

  useEffect(() => {
    const load = async () => {
      try {
        if (path) {
          setVolumes([]);
          setSelectedVolumeId(path);
          setVolumePath(path);
          return;
        }

        if (allowSelect) {
          const list = await mockEngine.getVolumes(subjectId);
          setVolumes(list);

          const preferred =
            list.find(v => (v.kind || '').toString().toLowerCase() === kind)?.path ||
            list[0]?.path ||
            null;

          if (preferred) {
            setSelectedVolumeId(preferred);
            setVolumePath(preferred);
          } else {
            const resolved = await mockEngine.resolveDefaultVolume(subjectId, kind);
            setSelectedVolumeId(resolved);
            setVolumePath(resolved);
          }
        } else {
          const resolvedPath = await mockEngine.resolveDefaultVolume(subjectId, kind);
          setSelectedVolumeId(resolvedPath);
          setVolumePath(resolvedPath);
        }

      } catch (error) {
        console.error('Failed to resolve volume:', error);
        setVolumePath(null);
      }
    };
    load();
  }, [subjectId, path, kind, allowSelect]);

  useEffect(() => {
    const loadInfo = async () => {
      try {
        if (!volumePath) return;
        if (isImage) return;
        const info = await mockEngine.getVolumeInfo(volumePath, subjectId);
        const zMax = Math.max(0, (info.dimensions[2] ?? 1) - 1);
        const tMax = Math.max(0, (info.dimensions[3] ?? 1) - 1);
        setMaxZ(zMax);
        setMaxT(tMax);

        setSliceZ(prev => Math.min(prev, zMax));
        setTimeFrame(prev => Math.min(prev, tMax));
        setWindowMin(Math.floor(info.min));
        setWindowMax(Math.ceil(info.max) || 1);
      } catch (error) {
        console.error('Failed to load volume info:', error);
      }
    };
    loadInfo();
  }, [subjectId, volumePath, isImage]);

  useEffect(() => {
    if (!isImage) loadSlice();
  }, [subjectId, volumePath, sliceZ, timeFrame, viewMode]);

  useEffect(() => {
    if (slices) {
      renderSlice();
    }
  }, [slices, windowMin, windowMax, colormap, viewMode]);

  const loadSlice = async () => {
    try {
      if (!volumePath) return;
      if (isImage) return;

      if (viewMode === 'single') {
        const data = await mockEngine.getSliceData(volumePath, sliceZ, timeFrame, subjectId);
        setSlices([data.data]);
        return;
      }

      const start = Math.max(0, sliceZ - 4);
      const sliceIndices = Array.from({ length: 9 }, (_, i) => Math.min(maxZ, start + i));
      const frames = await Promise.all(
        sliceIndices.map(z => mockEngine.getSliceData(volumePath, z, timeFrame, subjectId).then(r => r.data))
      );
      setSlices(frames);
    } catch (error) {
      console.error('Failed to load slice:', error);
    }
  };

  const renderSlice = () => {
    if (!canvasRef.current || !slices || slices.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = slices[0].length;
    const width = slices[0][0]?.length ?? 0;
    if (width <= 0 || height <= 0) return;
    const tilesX = viewMode === 'grid' ? 3 : 1;
    const tilesY = viewMode === 'grid' ? 3 : 1;
    canvas.width = width * tilesX;
    canvas.height = height * tilesY;

    const imageData = ctx.createImageData(canvas.width, canvas.height);

    const denom = windowMax - windowMin;
    const safeDenom = denom === 0 ? 1 : denom;

    for (let tile = 0; tile < tilesX * tilesY; tile++) {
      const slice = slices[Math.min(tile, slices.length - 1)];
      const tileX = tile % tilesX;
      const tileY = Math.floor(tile / tilesX);
      const ox = tileX * width;
      const oy = tileY * height;

      for (let y = 0; y < height; y++) {
        const row = slice[y];
        for (let x = 0; x < width; x++) {
          const value = row?.[x] ?? 0;
          const normalized = Math.max(0, Math.min(1, (value - windowMin) / safeDenom));
          const px = ox + x;
          const py = oy + y;
          const idx = (py * canvas.width + px) * 4;

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
    }

    ctx.putImageData(imageData, 0, 0);
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
        {isImage && volumePath ? (
          <img
            alt="Volume preview"
            src={volumePath}
            className="max-h-[600px] w-full max-w-full rounded-lg border border-border bg-background object-contain"
          />
        ) : (
          <canvas
            ref={canvasRef}
            className="max-h-[600px] max-w-full rounded-lg border border-border shadow-lg"
            style={{ imageRendering: 'pixelated' }}
          />
        )}
      </Card>

      <Card className="p-6">
        <div className="space-y-6">
          {allowSelect && !path ? (
            <div className="space-y-2">
              <Label>Image</Label>
              <Select
                value={selectedVolumeId}
                onValueChange={(value) => {
                  setSelectedVolumeId(value);
                  setVolumePath(value);
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a volume" />
                </SelectTrigger>
                <SelectContent>
                  {volumes.map(v => (
                    <SelectItem key={v.id} value={v.path}>
                      {v.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}

          {isImage ? null : (
            <>
              <div className="space-y-2">
                <Label>View</Label>
                <Select value={viewMode} onValueChange={(v) => setViewMode(v as 'single' | 'grid')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="single">Single slice</SelectItem>
                    <SelectItem value="grid">Multi-slice (3×3)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Slice (Z): {sliceZ}</Label>
                <Slider
                  value={[sliceZ]}
                  onValueChange={([value]) => setSliceZ(value)}
                  min={0}
                  max={maxZ}
                  step={1}
                  className="w-full"
                />
              </div>

              {maxT > 0 ? (
                <div className="space-y-2">
                  <Label>Time Frame (t): {timeFrame}</Label>
                  <Slider
                    value={[timeFrame]}
                    onValueChange={([value]) => setTimeFrame(value)}
                    min={0}
                    max={maxT}
                    step={1}
                    className="w-full"
                  />
                </div>
              ) : null}

              <div className="space-y-2">
                <Label>Window Min: {windowMin}</Label>
                <Slider
                  value={[windowMin]}
                  onValueChange={([value]) => setWindowMin(value)}
                  min={Math.min(windowMin, windowMax)}
                  max={windowMax}
                  step={10}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                <Label>Window Max: {windowMax}</Label>
                <Slider
                  value={[windowMax]}
                  onValueChange={([value]) => setWindowMax(value)}
                  min={Math.min(windowMin, windowMax)}
                  max={Math.max(windowMax, windowMin + 1)}
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
            </>
          )}
        </div>
      </Card>
    </div>
  );
}
