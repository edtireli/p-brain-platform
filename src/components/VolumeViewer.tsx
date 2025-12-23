import { useState, useEffect, useMemo, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { engine } from '@/lib/engine';
import type { VolumeFile } from '@/types';

interface VolumeViewerProps {
  subjectId: string;
  path?: string;
  kind?: 'dce' | 't1' | 't2' | 'flair' | 'diffusion' | 'map';
  allowSelect?: boolean;
}

export function VolumeViewer({ subjectId, path, kind = 'dce', allowSelect = true }: VolumeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hotkeyRef = useRef<HTMLDivElement>(null);
  const [volumePath, setVolumePath] = useState<string | null>(null);
  const [volumes, setVolumes] = useState<VolumeFile[]>([]);
  const [selectedVolumeId, setSelectedVolumeId] = useState<string>('');
  const [activeKind, setActiveKind] = useState<NonNullable<VolumeFile['kind']>>(() => kind);
  const [maxZ, setMaxZ] = useState(63);
  const [maxT, setMaxT] = useState(79);
  const [sliceZ, setSliceZ] = useState(32);
  const [timeFrame, setTimeFrame] = useState(0);
  const [windowMin, setWindowMin] = useState(0);
  const [windowMax, setWindowMax] = useState(2000);
	const [colormap, setColormap] = useState<'grayscale' | 'viridis' | 'hlcolour'>(() => (kind === 'map' ? 'hlcolour' : 'grayscale'));
	const [viewMode, setViewMode] = useState<'single' | 'grid'>(() => (kind === 'map' ? 'grid' : 'single'));
  const [slices, setSlices] = useState<number[][][] | null>(null);
  const [hotkeysActive, setHotkeysActive] = useState(false);

  useEffect(() => {
    setActiveKind(kind);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind]);

  const kindButtons = useMemo(
    () =>
      [
        { key: 'dce', label: 'DCE' },
        { key: 't1', label: 'T1' },
        { key: 't2', label: 'T2' },
        { key: 'flair', label: 'FLAIR' },
        { key: 'diffusion', label: 'Diffusion' },
      ] as const,
    []
  );

  const availableKinds = useMemo(() => {
    const s = new Set<string>();
    for (const v of volumes) {
      const k = String((v as any)?.kind || '').toLowerCase();
      if (k) s.add(k);
    }
    return s;
  }, [volumes]);

  const isImagePath = (p: string | null): boolean => {
    if (!p) return false;
    const base = p.split('?')[0];
    return /\.(png|jpe?g|webp|gif)$/i.test(base);
  };

  const isImage = isImagePath(volumePath);

  useEffect(() => {
    if (!hotkeysActive) return;
    if (!volumePath || isImage) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSliceZ(prev => Math.max(0, Math.min(maxZ, prev - 1)));
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSliceZ(prev => Math.max(0, Math.min(maxZ, prev + 1)));
        return;
      }

      const isDce = String(activeKind).toLowerCase() === 'dce';
      if (!isDce || maxT <= 0) return;

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setTimeFrame(prev => Math.max(0, Math.min(maxT, prev - 1)));
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        setTimeFrame(prev => Math.max(0, Math.min(maxT, prev + 1)));
      }
    };

    window.addEventListener('keydown', onKeyDown, { passive: false });
    return () => window.removeEventListener('keydown', onKeyDown as any);
  }, [hotkeysActive, volumePath, isImage, maxZ, maxT, activeKind]);

  useEffect(() => {
    const load = async () => {
      try {
        if (path) {
          setVolumes([]);
          setSelectedVolumeId(path);
          setVolumePath(path);
			setColormap(activeKind === 'map' ? 'hlcolour' : 'grayscale');
			setViewMode(activeKind === 'map' ? 'grid' : 'single');
          return;
        }

        if (allowSelect) {
          const list = await engine.getVolumes(subjectId);
          setVolumes(list);

          const preferred =
            list.find(v => (v.kind || '').toString().toLowerCase() === String(activeKind).toLowerCase())?.path ||
            list[0]?.path ||
            null;

          if (preferred) {
            setSelectedVolumeId(preferred);
            setVolumePath(preferred);
          } else {
            const resolved = await engine.resolveDefaultVolume(subjectId, activeKind as any);
            setSelectedVolumeId(resolved);
            setVolumePath(resolved);
          }
        } else {
          const resolvedPath = await engine.resolveDefaultVolume(subjectId, activeKind as any);
          setSelectedVolumeId(resolvedPath);
          setVolumePath(resolvedPath);
        }

      } catch (error) {
        console.error('Failed to resolve volume:', error);
        setVolumePath(null);
      }
    };
    load();
  }, [subjectId, path, activeKind, allowSelect]);

  useEffect(() => {
    const loadInfo = async () => {
      try {
        if (!volumePath) return;
        if (isImage) return;
        const info = await engine.getVolumeInfo(volumePath, subjectId);
        const zMax = Math.max(0, (info.dimensions[2] ?? 1) - 1);
        const tMax = Math.max(0, (info.dimensions[3] ?? 1) - 1);
        setMaxZ(zMax);
        setMaxT(tMax);

        setSliceZ(prev => Math.min(prev, zMax));
        setTimeFrame(prev => Math.min(prev, tMax));
			const min = Number.isFinite(Number(info.min)) ? Number(info.min) : 0;
			const max = Number.isFinite(Number(info.max)) ? Number(info.max) : 1;
			setWindowMin(Math.floor(min));
			setWindowMax(Math.max(1, Math.ceil(max)));
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
        const data = await engine.getSliceData(volumePath, sliceZ, timeFrame, subjectId);
        setSlices([data.data]);
        return;
      }

      const start = Math.max(0, sliceZ - 4);
      const sliceIndices = Array.from({ length: 9 }, (_, i) => Math.min(maxZ, start + i));
      const frames = await Promise.all(
        sliceIndices.map(z => engine.getSliceData(volumePath, z, timeFrame, subjectId).then(r => r.data))
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
          // NIfTI slices often come out inverted in our canvas mapping; flip vertically.
          const py = oy + (height - 1 - y);
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
			} else if (colormap === 'hlcolour') {
				const c = getHLColour(normalized);
				imageData.data[idx] = c[0];
				imageData.data[idx + 1] = c[1];
				imageData.data[idx + 2] = c[2];
				imageData.data[idx + 3] = 255;
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const getViridisColor = (t: number): [number, number, number] => {
    if (!Number.isFinite(t)) return [0, 0, 0];
    const tt = Math.max(0, Math.min(1, t));
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

		const scaled = tt * (viridis.length - 1);
    const idx = Math.max(0, Math.min(viridis.length - 1, Math.floor(scaled)));
    const nextIdx = Math.min(idx + 1, viridis.length - 1);
    const localT = scaled - idx;

    const r = viridis[idx]![0] + (viridis[nextIdx]![0] - viridis[idx]![0]) * localT;
    const g = viridis[idx]![1] + (viridis[nextIdx]![1] - viridis[idx]![1]) * localT;
    const b = viridis[idx]![2] + (viridis[nextIdx]![2] - viridis[idx]![2]) * localT;

    return [Math.floor(r), Math.floor(g), Math.floor(b)];
  };

	const getHLColour = (t: number): [number, number, number] => {
		if (!Number.isFinite(t)) return [0, 0, 0];
		const tt = Math.max(0, Math.min(1, t));
		// Match p-brain's "specthl" anchors (utils/montage.py).
		const anchors: Array<[number, [number, number, number]]> = [
			[0.0, [0, 0, 0]],
			[0.10, [0, 0, 40]],
			[0.22, [0, 0, 120]],
			[0.35, [60, 0, 170]],
			[0.50, [130, 0, 180]],
			[0.62, [200, 0, 120]],
			[0.73, [230, 30, 60]],
			[0.83, [255, 120, 0]],
			[0.92, [255, 200, 0]],
			[1.0, [255, 255, 255]],
		];

		let i = 0;
		while (i < anchors.length - 2 && tt > anchors[i + 1]![0]) i++;
		const [t0, c0] = anchors[i]!;
		const [t1, c1] = anchors[i + 1]!;
		const span = t1 - t0 || 1;
		const u = (tt - t0) / span;
		const r = c0[0] + (c1[0] - c0[0]) * u;
		const g = c0[1] + (c1[1] - c0[1]) * u;
		const b = c0[2] + (c1[2] - c0[2]) * u;
		return [Math.floor(r), Math.floor(g), Math.floor(b)];
	};

  return (
    <div className="flex flex-col gap-4">
      <Card
        ref={hotkeyRef as any}
        className="flex min-h-[65vh] flex-1 items-center justify-center bg-muted/30 p-3"
        tabIndex={0}
        onFocus={() => setHotkeysActive(true)}
        onBlur={() => setHotkeysActive(false)}
        onMouseEnter={() => setHotkeysActive(true)}
        onMouseLeave={() => setHotkeysActive(false)}
      >
        {!volumePath ? (
          <div className="text-center text-sm text-muted-foreground">
            No volumes available yet
          </div>
        ) : isImage && volumePath ? (
          <img
            alt="Volume preview"
            src={volumePath}
            className="h-full w-full rounded-lg border border-border bg-background object-contain"
          />
        ) : (
          <canvas
            ref={canvasRef}
            className="h-full w-full rounded-lg border border-border shadow-lg"
            style={{ imageRendering: 'pixelated' }}
          />
        )}
      </Card>

      <Card className="p-4">
        <div className="flex flex-col gap-4">
          {allowSelect && !path && activeKind !== 'map' ? (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="space-y-1">
                <Label>Modality</Label>
                <div className="mt-2 flex flex-wrap gap-2">
                  {kindButtons.map(kb => {
                    const k = kb.key;
                    const isActive = String(activeKind).toLowerCase() === k;
                    const has = volumes.length > 0 ? availableKinds.has(k) : true;
                    return (
                      <Button
                        key={k}
                        type="button"
                        size="sm"
                        variant={isActive ? 'default' : 'outline'}
                        disabled={!has}
                        onClick={async () => {
                          setActiveKind(k);
                          try {
                            const preferred = volumes.find(v => (v.kind || '').toString().toLowerCase() === k)?.path;
                            const resolved = preferred || (await engine.resolveDefaultVolume(subjectId, k as any));
                            setSelectedVolumeId(resolved);
                            setVolumePath(resolved);
                          } catch (e) {
                            console.error('Failed to switch modality', e);
                          }
                        }}
                      >
                        {kb.label}
                      </Button>
                    );
                  })}
                </div>
              </div>

              {/* Keep the full list for power users (lots of volumes) */}
              <div className="min-w-[260px]">
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
            </div>
          ) : allowSelect && !path ? (
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
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="space-y-1">
                  <Label>View</Label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Button size="sm" type="button" variant={viewMode === 'single' ? 'default' : 'outline'} onClick={() => setViewMode('single')}>
                      Single
                    </Button>
                    <Button size="sm" type="button" variant={viewMode === 'grid' ? 'default' : 'outline'} onClick={() => setViewMode('grid')}>
                      3×3
                    </Button>
                  </div>
                </div>

                <div className="space-y-1">
                  <Label>Colormap</Label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      type="button"
                      variant={colormap === 'grayscale' ? 'default' : 'outline'}
                      onClick={() => setColormap('grayscale')}
                    >
                      Gray
                    </Button>
                    <Button
                      size="sm"
                      type="button"
                      variant={colormap === 'hlcolour' ? 'default' : 'outline'}
                      onClick={() => setColormap('hlcolour')}
                    >
                      HLColour
                    </Button>
                    <Button
                      size="sm"
                      type="button"
                      variant={colormap === 'viridis' ? 'default' : 'outline'}
                      onClick={() => setColormap('viridis')}
                    >
                      Viridis
                    </Button>
                  </div>
                </div>
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

            </>
          )}
        </div>
      </Card>
    </div>
  );
}
