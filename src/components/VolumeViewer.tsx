import { useState, useEffect, useMemo, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { engine } from '@/lib/engine';
import type { RoiMaskVolume, RoiOverlay, VolumeFile } from '@/types';
import { ArrowsInSimple, ArrowsOutSimple } from '@phosphor-icons/react';

function computedCssColor(expr: string, fallback: string): string {
  try {
    const el = document.createElement('div');
    el.style.color = expr;
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    el.style.top = '-9999px';
    document.body.appendChild(el);
    const color = getComputedStyle(el).color;
    document.body.removeChild(el);
    return color || fallback;
  } catch {
    return fallback;
  }
}

interface VolumeViewerProps {
  subjectId: string;
  path?: string;
  kind?: 'dce' | 't1' | 't2' | 'flair' | 'diffusion' | 'map';
  allowSelect?: boolean;
  overlayPaths?: string[];

  showRoiOverlays?: boolean;
  roiOverlays?: RoiOverlay[];
  roiMasks?: RoiMaskVolume[];
}

export function VolumeViewer({
  subjectId,
  path,
  kind = 'dce',
  allowSelect = true,
  overlayPaths,
  showRoiOverlays = false,
  roiOverlays = [],
  roiMasks = [],
}: VolumeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hotkeyRef = useRef<HTMLDivElement>(null);
  const viewWrapRef = useRef<HTMLDivElement>(null);
  const renderMetaRef = useRef<{ width: number; height: number; tilesX: number; tilesY: number } | null>(null);
  const [volumePath, setVolumePath] = useState<string | null>(null);
  const [volumes, setVolumes] = useState<VolumeFile[]>([]);
  const [selectedVolumeId, setSelectedVolumeId] = useState<string>('');
  const [activeKind, setActiveKind] = useState<NonNullable<VolumeFile['kind']>>(() => kind);
  const [maxZ, setMaxZ] = useState(63);
  const [maxT, setMaxT] = useState(79);
  const [sliceZ, setSliceZ] = useState(32);
  const [timeFrame, setTimeFrame] = useState(0);
  const [dataMin, setDataMin] = useState(0);
  const [dataMax, setDataMax] = useState(1);
  const [windowMin, setWindowMin] = useState(0);
  const [windowMax, setWindowMax] = useState(2000);
  const [colormap, setColormap] = useState<'grayscale' | 'viridis' | 'hlcolour'>(() => (kind === 'map' ? 'hlcolour' : 'grayscale'));
  const [viewMode, setViewMode] = useState<'single' | 'grid'>(() => 'single');
  const [slices, setSlices] = useState<number[][][] | null>(null);
  const [roiMaskTiles, setRoiMaskTiles] = useState<number[][][][] | null>(null);
  const [hotkeysActive, setHotkeysActive] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showKeybindings, setShowKeybindings] = useState(false);
  const [roiHover, setRoiHover] = useState<{ x: number; y: number; labels: string[] } | null>(null);

  const VIEW_ROTATION_DEG = -90;
  const VIEW_SCALE = 1.06;

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

  useEffect(() => {
    if (!showRoiOverlays) return;
    if (!Array.isArray(roiOverlays) || roiOverlays.length === 0) return;
    if (!Number.isFinite(maxZ) || maxZ <= 0) return;

    const overlayZs = Array.from(
      new Set(
        roiOverlays
          .map(o => Number((o as any)?.sliceIndex))
          .filter(z => Number.isFinite(z))
          .map(z => Math.max(0, Math.min(maxZ, z)))
      )
    ).sort((a, b) => a - b);

    if (overlayZs.length === 0) return;

    const visibleZs =
      viewMode === 'grid'
        ? new Set(Array.from({ length: 9 }, (_, i) => Math.min(maxZ, Math.max(0, sliceZ - 4 + i))))
        : new Set([sliceZ]);

    // If no overlay would be visible at the current slice selection, jump to the first overlay.
    const anyVisible = overlayZs.some(z => visibleZs.has(z));
    if (!anyVisible) {
      setSliceZ(overlayZs[0]!);
    }
  }, [showRoiOverlays, roiOverlays, maxZ, viewMode, sliceZ]);

  const isImagePath = (p: string | null): boolean => {
    if (!p) return false;
    const base = p.split('?')[0];
    return /\.(png|jpe?g|webp|gif)$/i.test(base);
  };

  const hasOverlay = Array.isArray(overlayPaths) && overlayPaths.length > 0;
  const isImage = !hasOverlay && isImagePath(volumePath);

  const format3Sig = useMemo(() => {
    const fmt = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 3 });
    return (x: number) => (Number.isFinite(x) ? fmt.format(x) : '—');
  }, []);

  useEffect(() => {
    const onFs = () => {
      const el = hotkeyRef.current as any;
      const fsEl = (document as any).fullscreenElement as Element | null;
      setIsFullscreen(!!fsEl && !!el && (fsEl === el || (el instanceof Element && el.contains(fsEl))));
    };
    document.addEventListener('fullscreenchange', onFs);
    return () => document.removeEventListener('fullscreenchange', onFs);
  }, []);

  const getTauriWindow = () => {
    const tauri = (window as any)?.__TAURI__;
    if (!tauri) return null;
    if (typeof tauri?.window?.getCurrentWindow === 'function') return tauri.window.getCurrentWindow();
    if (tauri?.window?.appWindow) return tauri.window.appWindow;
    return null;
  };

  const toggleFullscreen = async () => {
    const tauriWin: any = getTauriWindow();
    if (tauriWin?.setFullscreen && tauriWin?.isFullscreen) {
      try {
        const current = await tauriWin.isFullscreen();
        await tauriWin.setFullscreen(!current);
        setIsFullscreen(!current);
        return;
      } catch {
        // fall through to DOM fullscreen
      }
    }

    const el = hotkeyRef.current as any;
    if (!el) return;

    try {
      if ((document as any).fullscreenElement) {
        await (document as any).exitFullscreen?.();
      } else {
        await el.requestFullscreen?.();
      }
    } catch {
      // ignore (fullscreen can be blocked by browser policies)
    }
  };

  const handleHotkey = (key: string, preventDefault: () => void) => {
    if (!volumePath || isImage) return;

    if (key === 'ArrowUp') {
      preventDefault();
      setSliceZ(prev => Math.max(0, Math.min(maxZ, prev + 1)));
      return;
    }
    if (key === 'ArrowDown') {
      preventDefault();
      setSliceZ(prev => Math.max(0, Math.min(maxZ, prev - 1)));
      return;
    }

    const isDce = String(activeKind).toLowerCase() === 'dce';
    if (isDce && maxT > 0) {
      if (key === 'ArrowLeft') {
        preventDefault();
        setTimeFrame(prev => Math.max(0, Math.min(maxT, prev - 1)));
        return;
      }
      if (key === 'ArrowRight') {
        preventDefault();
        setTimeFrame(prev => Math.max(0, Math.min(maxT, prev + 1)));
        return;
      }
    }

    if (key === 'f' || key === 'F') {
      preventDefault();
      void toggleFullscreen();
    }
  };

  useEffect(() => {
    if (!hotkeysActive) return;
    if (!volumePath || isImage) return;

    const onKeyDown = (e: KeyboardEvent) => {
      handleHotkey(e.key, () => e.preventDefault());
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
          setViewMode('single');
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
        const primaryInfo = await engine.getVolumeInfo(volumePath, subjectId, String(activeKind));
        const zMax = Math.max(0, (primaryInfo.dimensions[2] ?? 1) - 1);
        const tMax = Math.max(0, (primaryInfo.dimensions[3] ?? 1) - 1);
        setMaxZ(zMax);
        setMaxT(tMax);

        setSliceZ(prev => Math.min(prev, zMax));
        setTimeFrame(prev => Math.min(prev, tMax));
      const rawMin = Number.isFinite(Number(primaryInfo.min)) ? Number(primaryInfo.min) : 0;
      const rawMax = Number.isFinite(Number(primaryInfo.max)) ? Number(primaryInfo.max) : 1;
			const dMin = Math.min(rawMin, rawMax);
			const dMax = Math.max(rawMin, rawMax);

      let mergedDataMin = dMin;
      let mergedDataMax = dMax;
      let mergedDispMin = Number.isFinite(Number((primaryInfo as any).displayMin)) ? Number((primaryInfo as any).displayMin) : dMin;
      let mergedDispMax = Number.isFinite(Number((primaryInfo as any).displayMax)) ? Number((primaryInfo as any).displayMax) : dMax;

      if (hasOverlay) {
        for (const p of overlayPaths || []) {
          try {
            const info = await engine.getVolumeInfo(p, subjectId, String(activeKind));
            const m0 = Number.isFinite(Number(info.min)) ? Number(info.min) : undefined;
            const m1 = Number.isFinite(Number(info.max)) ? Number(info.max) : undefined;
            if (m0 !== undefined && m1 !== undefined) {
              mergedDataMin = Math.min(mergedDataMin, m0, m1);
              mergedDataMax = Math.max(mergedDataMax, m0, m1);
            }
            const dm = Number.isFinite(Number((info as any).displayMin)) ? Number((info as any).displayMin) : undefined;
            const dx = Number.isFinite(Number((info as any).displayMax)) ? Number((info as any).displayMax) : undefined;
            if (dm !== undefined && dx !== undefined) {
              mergedDispMin = Math.min(mergedDispMin, dm, dx);
              mergedDispMax = Math.max(mergedDispMax, dm, dx);
            }
          } catch {
            // ignore overlay range failures
          }
        }
      }

      setDataMin(mergedDataMin);
      setDataMax(mergedDataMax);
      setWindowMin(Math.min(mergedDispMin, mergedDispMax));
      setWindowMax(Math.max(mergedDispMin, mergedDispMax));
      } catch (error) {
        console.error('Failed to load volume info:', error);
      }
    };
    loadInfo();
  }, [subjectId, volumePath, isImage, activeKind, hasOverlay, overlayPaths]);

  useEffect(() => {
    if (!isImage) loadSlice();
  }, [subjectId, volumePath, sliceZ, timeFrame, viewMode, showRoiOverlays, roiMasks]);

  useEffect(() => {
    if (slices) {
      renderSlice();
    }
  }, [
    slices,
    roiMaskTiles,
    windowMin,
    windowMax,
    colormap,
    viewMode,
    showRoiOverlays,
    roiOverlays,
    roiMasks,
    sliceZ,
    maxZ,
  ]);

  const loadSlice = async () => {
    try {
      if (!volumePath) return;
      if (isImage) return;

      const wantRoiMasks = showRoiOverlays && Array.isArray(roiMasks) && roiMasks.length > 0;

      const loadMasksForZ = async (z: number): Promise<number[][][]> => {
        if (!wantRoiMasks) return [];
        const layers = await Promise.all(
          (roiMasks || []).map(m => engine.getSliceData(m.path, z, 0, subjectId).then(r => (Array.isArray(r.data) ? (r.data as number[][]) : [])))
        );
        return layers;
      };

      const composeOverlay = (layers: number[][][]): number[][] => {
        const h = layers[0]?.length ?? 0;
        const w = layers[0]?.[0]?.length ?? 0;
        const out: number[][] = Array.from({ length: h }, () => Array.from({ length: w }, () => 0));
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            let best = 0;
            let bestAbs = 0;
            for (const layer of layers) {
              const v = layer?.[y]?.[x] ?? 0;
              const av = Math.abs(v);
              if (av > bestAbs && av > 1e-8) {
                best = v;
                bestAbs = av;
              }
            }
            out[y]![x] = best;
          }
        }
        return out;
      };

      if (viewMode === 'single') {
        if (hasOverlay) {
          const layers = await Promise.all(
            (overlayPaths || []).map(p => engine.getSliceData(p, sliceZ, timeFrame, subjectId).then(r => r.data))
          );
          setSlices([composeOverlay(layers)]);
          setRoiMaskTiles(null);
          return;
        }
        const data = await engine.getSliceData(volumePath, sliceZ, timeFrame, subjectId);
        setSlices([Array.isArray(data.data) ? (data.data as number[][]) : []]);
        setRoiMaskTiles(wantRoiMasks ? [await loadMasksForZ(sliceZ)] : null);
        return;
      }

      const start = Math.max(0, sliceZ - 4);
      const sliceIndices = Array.from({ length: 9 }, (_, i) => Math.min(maxZ, start + i));
      const frames = await Promise.all(
        sliceIndices.map(async z => {
          if (hasOverlay) {
            const layers = await Promise.all(
              (overlayPaths || []).map(p => engine.getSliceData(p, z, timeFrame, subjectId).then(r => r.data))
            );
            return composeOverlay(layers);
          }
          return engine.getSliceData(volumePath, z, timeFrame, subjectId).then(r => (Array.isArray(r.data) ? (r.data as number[][]) : []));
        })
      );
      setSlices(frames);

      if (wantRoiMasks) {
        const maskTiles = await Promise.all(sliceIndices.map(z => loadMasksForZ(z)));
        setRoiMaskTiles(maskTiles);
      } else {
        setRoiMaskTiles(null);
      }
    } catch (error) {
      console.error('Failed to load slice:', error);
    }
  };

  const renderSlice = () => {
    if (!canvasRef.current || !Array.isArray(slices) || slices.length === 0) return;

    const firstValid = slices.find(s => Array.isArray(s) && (s.length === 0 || Array.isArray((s as any)[0])));
    if (!firstValid || !Array.isArray(firstValid)) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = firstValid.length;
    const width = (firstValid as any)[0]?.length ?? 0;
    if (width <= 0 || height <= 0) return;
    const tilesX = viewMode === 'grid' ? 3 : 1;
    const tilesY = viewMode === 'grid' ? 3 : 1;
    renderMetaRef.current = { width, height, tilesX, tilesY };
    canvas.width = width * tilesX;
    canvas.height = height * tilesY;

    const imageData = ctx.createImageData(canvas.width, canvas.height);

    const parseRgb = (raw: string): [number, number, number] | null => {
      const s = (raw || '').trim().toLowerCase();
      // rgb(1, 2, 3) / rgba(1, 2, 3, 0.5)
      let m = s.match(/rgba?\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
      if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
      // rgb(1 2 3 / 0.5)
      m = s.match(/rgba?\((\d+)\s+(\d+)\s+(\d+)/);
      if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
      return null;
    };

    const wantRoiMasks = showRoiOverlays && Array.isArray(roiMasks) && roiMasks.length > 0 && Array.isArray(roiMaskTiles);
    const aifColor = parseRgb(computedCssColor('var(--destructive)', 'rgb(255, 0, 0)')) || [255, 0, 0];
    const vifColor = parseRgb(computedCssColor('var(--primary)', 'rgb(0, 160, 255)')) || [0, 160, 255];
    const maskAlpha = 0.38;

    const maskColors: Array<[number, number, number]> = (roiMasks || []).map(m => {
      const t = String((m as any)?.roiType || '').toLowerCase();
      if (t.includes('vein')) return vifColor;
      if (t.includes('artery')) return aifColor;
      // Fallback: subtype heuristics.
      const st = String((m as any)?.roiSubType || '').toLowerCase();
      if (st.includes('sinus') || st.includes('vein') || st.includes('vif')) return vifColor;
      return aifColor;
    });

    const denom = windowMax - windowMin;
    const safeDenom = denom === 0 ? 1 : denom;

    for (let tile = 0; tile < tilesX * tilesY; tile++) {
      const sliceRaw = slices[Math.min(tile, slices.length - 1)];
      const slice = Array.isArray(sliceRaw) ? sliceRaw : firstValid;
      const tileX = tile % tilesX;
      const tileY = Math.floor(tile / tilesX);
      const ox = tileX * width;
      const oy = tileY * height;

      const masksForTile = wantRoiMasks ? (roiMaskTiles as any)[Math.min(tile, (roiMaskTiles as any).length - 1)] : null;

      for (let y = 0; y < height; y++) {
        const row = (slice as any)[y];
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

          // Pixel-level ROI mask overlays (AIF/VIF).
          if (wantRoiMasks && Array.isArray(masksForTile) && masksForTile.length > 0) {
            for (let mi = 0; mi < masksForTile.length; mi++) {
              const mrow = (masksForTile as any)[mi]?.[y];
              const mv = mrow?.[x] ?? 0;
              if (Number(mv) > 0) {
                const c = maskColors[mi] || aifColor;
                const inv = 1 - maskAlpha;
                imageData.data[idx] = Math.round(imageData.data[idx] * inv + c[0] * maskAlpha);
                imageData.data[idx + 1] = Math.round(imageData.data[idx + 1] * inv + c[1] * maskAlpha);
                imageData.data[idx + 2] = Math.round(imageData.data[idx + 2] * inv + c[2] * maskAlpha);
                imageData.data[idx + 3] = 255;
              }
            }
          }
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);

    if (showRoiOverlays && Array.isArray(roiOverlays) && roiOverlays.length > 0) {
      try {
        const stroke = computedCssColor('var(--destructive)', 'rgba(255, 0, 0, 0.95)');

        const start = Math.max(0, sliceZ - 4);
        const sliceIndices =
          viewMode === 'grid'
            ? Array.from({ length: 9 }, (_, i) => Math.min(maxZ, start + i))
            : [sliceZ];

        ctx.save();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.95;

        for (let tile = 0; tile < tilesX * tilesY; tile++) {
          const tileX = tile % tilesX;
          const tileY = Math.floor(tile / tilesX);
          const ox = tileX * width;
          const oy = tileY * height;
          const zForTile = sliceIndices[Math.min(tile, sliceIndices.length - 1)] ?? sliceZ;

          const overlaysForSlice = roiOverlays.filter(o => Number(o.sliceIndex) === Number(zForTile));
          for (const o of overlaysForSlice) {
            const row0 = Math.max(0, Math.min(height - 1, Number(o.row0)));
            const row1 = Math.max(0, Math.min(height - 1, Number(o.row1)));
            const col0 = Math.max(0, Math.min(width - 1, Number(o.col0)));
            const col1 = Math.max(0, Math.min(width - 1, Number(o.col1)));

            const r0 = Math.min(row0, row1);
            const r1 = Math.max(row0, row1);
            const c0 = Math.min(col0, col1);
            const c1 = Math.max(col0, col1);

            // Canvas y is vertically flipped relative to voxel row index.
            const x0 = ox + c0;
            const y0 = oy + (height - 1 - r1);
            const w = c1 - c0 + 1;
            const h = r1 - r0 + 1;

            const cx = x0 + w / 2;
            const cy = y0 + h / 2;
            const rx = Math.max(2, w / 2);
            const ry = Math.max(2, h / 2);
            ctx.beginPath();
            ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
        ctx.restore();
      } catch {
        // ignore overlay rendering errors
      }
    }
  };

  const updateRoiHover = (clientX: number, clientY: number) => {
    if (!showRoiOverlays) {
      setRoiHover(null);
      return;
    }
    if (!Array.isArray(roiMasks) || roiMasks.length === 0) {
      setRoiHover(null);
      return;
    }
    if (!Array.isArray(roiMaskTiles) || roiMaskTiles.length === 0) {
      setRoiHover(null);
      return;
    }
    const wrap = viewWrapRef.current;
    const card = hotkeyRef.current;
    const meta = renderMetaRef.current;
    if (!wrap || !card || !meta) {
      setRoiHover(null);
      return;
    }

    const rect = wrap.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    // Invert the view transform: rotate(-90) scale(1.06) about center.
    const dx = (clientX - cx) / VIEW_SCALE;
    const dy = (clientY - cy) / VIEW_SCALE;
    // Original = R(+90) * transformed
    const x0 = -dy;
    const y0 = dx;

    const w0 = Math.max(1, wrap.clientWidth);
    const h0 = Math.max(1, wrap.clientHeight);
    const lx = x0 + w0 / 2;
    const ly = y0 + h0 / 2;
    if (lx < 0 || lx >= w0 || ly < 0 || ly >= h0) {
      setRoiHover(null);
      return;
    }

    const canvasW = meta.width * meta.tilesX;
    const canvasH = meta.height * meta.tilesY;
    const cX = Math.max(0, Math.min(canvasW - 1, Math.floor((lx / w0) * canvasW)));
    const cY = Math.max(0, Math.min(canvasH - 1, Math.floor((ly / h0) * canvasH)));

    const tileX = Math.max(0, Math.min(meta.tilesX - 1, Math.floor(cX / meta.width)));
    const tileY = Math.max(0, Math.min(meta.tilesY - 1, Math.floor(cY / meta.height)));
    const tileIndex = tileY * meta.tilesX + tileX;

    const xIn = cX - tileX * meta.width;
    const yCanvas = cY - tileY * meta.height;
    const yIn = meta.height - 1 - yCanvas;

    const masksForTile: any = (roiMaskTiles as any)[Math.min(tileIndex, (roiMaskTiles as any).length - 1)];
    if (!Array.isArray(masksForTile) || masksForTile.length === 0) {
      setRoiHover(null);
      return;
    }

    const labels: string[] = [];
    for (let mi = 0; mi < masksForTile.length; mi++) {
      const mv = masksForTile?.[mi]?.[yIn]?.[xIn] ?? 0;
      if (Number(mv) > 0) {
        const m = roiMasks[mi];
        const type = String((m as any)?.roiType || '').trim();
        const sub = String((m as any)?.roiSubType || '').trim();
        labels.push(type && sub ? `${type}: ${sub}` : String((m as any)?.name || m?.id || 'ROI'));
      }
    }

    if (labels.length === 0) {
      setRoiHover(null);
      return;
    }

    const cardRect = card.getBoundingClientRect();
    setRoiHover({
      x: Math.max(8, Math.min(cardRect.width - 8, clientX - cardRect.left + 12)),
      y: Math.max(8, Math.min(cardRect.height - 8, clientY - cardRect.top + 12)),
      labels,
    });
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
      {allowSelect && !path && activeKind !== 'map' ? (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
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
      ) : null}

      <Card
        ref={hotkeyRef as any}
        className="relative mx-auto aspect-square w-full max-w-[70vh] overflow-hidden bg-muted/30 p-0"
        tabIndex={0}
        onFocus={() => setHotkeysActive(true)}
        onBlur={() => setHotkeysActive(false)}
        onMouseEnter={() => setHotkeysActive(true)}
        onMouseLeave={() => setHotkeysActive(false)}
        onPointerDown={() => {
          setHotkeysActive(true);
          (hotkeyRef.current as any)?.focus?.();
        }}
        onKeyDown={(e: any) => {
          setHotkeysActive(true);
          handleHotkey(String(e?.key || ''), () => e.preventDefault());
        }}
      >
        <Button
          type="button"
          variant="secondary"
          size="icon"
          className="absolute right-2 top-2 z-10 h-9 w-9 rounded-full bg-background/70 backdrop-blur"
          onClick={() => void toggleFullscreen()}
          aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        >
          {isFullscreen ? <ArrowsInSimple size={18} /> : <ArrowsOutSimple size={18} />}
        </Button>


      <div
        className="absolute right-0 top-1/2 z-10 -translate-y-1/2 rounded-l-md border border-border bg-background/70 px-2 py-1 text-[11px] text-muted-foreground backdrop-blur"
        onMouseEnter={() => setShowKeybindings(true)}
      >
        Keys
      </div>

      <div
        className={`absolute right-0 top-0 z-10 h-full w-56 border-l border-border bg-background/90 p-3 text-xs text-muted-foreground backdrop-blur transition-transform ${
          showKeybindings ? 'translate-x-0 pointer-events-auto' : 'translate-x-full pointer-events-none'
        }`}
        onMouseEnter={() => setShowKeybindings(true)}
        onMouseLeave={() => setShowKeybindings(false)}
      >
        <div className="mb-2 font-medium text-foreground">Keybindings</div>
        <div className="mono">↑ / ↓: slice</div>
        <div className="mono">← / →: time (DCE only)</div>
        <div className="mono">F: fullscreen</div>
        <div className="mt-3 text-[11px]">{hotkeysActive ? 'Hotkeys active' : 'Click viewer to activate'}</div>
      </div>

        {!volumePath ? (
          <div className="text-center text-sm text-muted-foreground">
            No volumes available yet
          </div>
        ) : isImage && volumePath ? (
          <div
            ref={viewWrapRef}
            className="flex h-full w-full items-center justify-center"
            style={{ transform: `rotate(${VIEW_ROTATION_DEG}deg) scale(${VIEW_SCALE})`, transformOrigin: 'center' }}
            onPointerMove={(e: any) => updateRoiHover(e.clientX, e.clientY)}
            onPointerLeave={() => setRoiHover(null)}
          >
            <img
              alt="Volume preview"
              src={volumePath}
              className="h-full w-full bg-background object-contain"
            />
          </div>
        ) : (
          <div
            ref={viewWrapRef}
            className="flex h-full w-full items-center justify-center"
            style={{ transform: `rotate(${VIEW_ROTATION_DEG}deg) scale(${VIEW_SCALE})`, transformOrigin: 'center' }}
            onPointerMove={(e: any) => updateRoiHover(e.clientX, e.clientY)}
            onPointerLeave={() => setRoiHover(null)}
          >
            <canvas
              ref={canvasRef}
              className="h-full w-full"
              style={{ imageRendering: 'pixelated' }}
            />
          </div>
        )}

        {roiHover ? (
          <div
            className="pointer-events-none absolute z-20 max-w-[260px] rounded-md border border-border bg-background/90 px-2 py-1 text-xs text-foreground shadow-sm backdrop-blur"
            style={{ left: roiHover.x, top: roiHover.y }}
          >
            {roiHover.labels.map((l, i) => (
              <div key={`${l}-${i}`}>{l}</div>
            ))}
          </div>
        ) : null}
      </Card>

      <Card className="p-4">
        <div className="flex flex-col gap-4">
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

          {String(activeKind).toLowerCase() === 'map' ? (
            <div className="space-y-2">
              <Label>
                Colourbar range: {format3Sig(windowMin)} → {format3Sig(windowMax)}
              </Label>
              <Slider
                value={[windowMin, windowMax]}
                onValueChange={([a, b]) => {
                  const lo = Math.min(a, b);
                  const hi = Math.max(a, b);
                  setWindowMin(lo);
                  setWindowMax(hi);
                }}
                min={Math.min(dataMin, dataMax)}
                max={Math.max(dataMin, dataMax)}
                step={Math.max(1e-6, (Math.max(dataMin, dataMax) - Math.min(dataMin, dataMax)) / 500)}
                className="w-full"
              />
            </div>
          ) : null}

              <div className="space-y-2">
                <Label>Slice (Z)</Label>
                <div className="flex flex-wrap items-center gap-2">
                  <Button type="button" size="sm" variant="outline" onClick={() => setSliceZ(v => Math.max(0, v - 1))} disabled={sliceZ <= 0}>
                    −
                  </Button>
                  <Input
                    type="number"
                    value={sliceZ}
                    min={0}
                    max={maxZ}
                    step={1}
                    className="h-9 w-[110px]"
                    onChange={e => {
                      const n = Math.round(Number(e.target.value));
                      if (!Number.isFinite(n)) return;
                      setSliceZ(Math.max(0, Math.min(maxZ, n)));
                    }}
                  />
                  <div className="text-xs text-muted-foreground">/ {maxZ}</div>
                  <Button type="button" size="sm" variant="outline" onClick={() => setSliceZ(v => Math.min(maxZ, v + 1))} disabled={sliceZ >= maxZ}>
                    +
                  </Button>
                </div>
              </div>

            </>
          )}
        </div>
      </Card>
    </div>
  );
}
