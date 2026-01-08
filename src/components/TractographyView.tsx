import { useEffect, useMemo, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { engine } from '@/lib/engine';
import type { TractographyData } from '@/types';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface TractographyViewProps {
  subjectId: string;
}

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

function disposeObject(root: THREE.Object3D, disposeMaterials: boolean = true) {
  root.traverse(obj => {
    const anyObj = obj as any;
    const geom = anyObj.geometry as THREE.BufferGeometry | undefined;
    if (geom) geom.dispose();

    if (disposeMaterials) {
      const mat = anyObj.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach(m => m.dispose());
      else if (mat) mat.dispose();
    }
  });
}

export function TractographyView({ subjectId }: TractographyViewProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<TractographyData | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const selectedRef = useRef<number | null>(null);
  const cssFullscreenRef = useRef(false);
  const autoSpinRef = useRef(false);
  const idleTimerRef = useRef<number | undefined>(undefined);

  const containerRef = useRef<HTMLDivElement | null>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  const groupRef = useRef<THREE.Group | null>(null);
  const linesRef = useRef<THREE.Line[]>([]);
  const baseMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);
  const selectedMaterialRef = useRef<THREE.LineBasicMaterial | null>(null);

  const pickThresholdRef = useRef<number>(1);

  const streamlines = useMemo(() => data?.streamlines ?? [], [data]);
  const colours = useMemo(() => data?.colors ?? [], [data]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSelected(null);

    (async () => {
      try {
        const payload = await engine.getSubjectTractography(subjectId);
        if (cancelled) return;
        setData(payload);
      } catch (e: any) {
        if (cancelled) return;
        setData(null);
        const msg = typeof e?.message === 'string' ? e.message : 'Failed to load tractography.';
        setError(msg);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [subjectId]);

  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100000);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(container.clientWidth || 1, container.clientHeight || 1);
    // Force a dark canvas; some themes set card/background to pure white.
    const bgColor = '#060912';
    renderer.setClearColor(new THREE.Color(bgColor), 1);
    // Ensure the canvas actually occupies the container even if the initial
    // WebView layout reports 0x0 (common when mounted in a hidden tab).
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.backgroundColor = bgColor;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.zoomSpeed = 1.0;
    controls.rotateSpeed = 1.3; // slightly faster mouse/touch rotation
    controls.maxPolarAngle = Math.PI; // allow full orbit
    controls.screenSpacePanning = false;

    const group = new THREE.Group();
    scene.add(group);

    const baseColor = '#50c6ff';
    const accentColor = '#ff8a8a';

    const baseMaterial = new THREE.LineBasicMaterial({ color: new THREE.Color(baseColor || 'rgb(46, 196, 255)'), transparent: false, opacity: 1 });
    const selectedMaterial = new THREE.LineBasicMaterial({ color: new THREE.Color(accentColor || 'rgb(255, 110, 110)'), transparent: false, opacity: 1 });

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    controlsRef.current = controls;
    groupRef.current = group;
    baseMaterialRef.current = baseMaterial;
    selectedMaterialRef.current = selectedMaterial;

    const resize = () => {
      const w = Math.max(1, container.clientWidth);
      const h = Math.max(1, container.clientHeight);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };

    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();
    // Best-effort: re-measure after layout settles (tab switches can delay sizing).
    window.requestAnimationFrame(resize);
    window.setTimeout(resize, 120);

    const raycaster = new THREE.Raycaster();
    const restartIdle = () => {
      autoSpinRef.current = false;
      if (idleTimerRef.current) window.clearTimeout(idleTimerRef.current);
      idleTimerRef.current = window.setTimeout(() => {
        autoSpinRef.current = true;
      }, 1000);
    };

    const onPointerDown = (ev: PointerEvent) => {
      restartIdle();
      const canvas = renderer.domElement;
      const rect = canvas.getBoundingClientRect();
      const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);

      raycaster.setFromCamera({ x, y }, camera);
      raycaster.params.Line = raycaster.params.Line || {};
      raycaster.params.Line.threshold = pickThresholdRef.current;

      const hits = raycaster.intersectObjects(linesRef.current, false);
      if (!hits.length) return;

      const hit = hits[0].object as THREE.Line;
      const idx = typeof hit.userData?.streamlineIndex === 'number' ? (hit.userData.streamlineIndex as number) : null;
      if (idx == null) return;

      // revert previous selection
      if (selectedRef.current != null) {
        const prev = linesRef.current.find(l => l.userData?.streamlineIndex === selectedRef.current);
        if (prev) {
          const original = (prev.userData?.baseMaterial as THREE.Material | undefined) ?? baseMaterialRef.current;
          if (original) prev.material = original;
        }
      }

      // apply selection
      if (selectedMaterialRef.current) hit.material = selectedMaterialRef.current;
      setSelected(idx);
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('wheel', restartIdle, { passive: true });
    window.addEventListener('keydown', restartIdle, { capture: true });
    controls.addEventListener('start', restartIdle);
    const onFullscreenChange = () => {
      const fullscreenElement = document.fullscreenElement || (document as any).webkitFullscreenElement;
      const active = Boolean(fullscreenElement) || cssFullscreenRef.current;
      setIsFullscreen(active);
      // Fix a common issue where exiting fullscreen leaves a stale canvas size.
      resize();
    };
    document.addEventListener('fullscreenchange', onFullscreenChange, { capture: true });
    document.addEventListener('webkitfullscreenchange', onFullscreenChange as any, { capture: true });

    let raf = 0;
    const tick = () => {
      if (autoSpinRef.current && cameraRef.current && controlsRef.current) {
        const controls = controlsRef.current;
        const camera = cameraRef.current;
        const target = controls.target.clone();
        const offset = camera.position.clone().sub(target);
        const qYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.0035);
        offset.applyQuaternion(qYaw);
        camera.position.copy(target).add(offset);
        camera.up.set(0, 0, 1);
        camera.lookAt(target);
      }
      controls.update();
      renderer.render(scene, camera);
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('wheel', restartIdle as any);
      window.removeEventListener('keydown', restartIdle as any, { capture: true } as any);
      controls.removeEventListener('start', restartIdle as any);
      document.removeEventListener('fullscreenchange', onFullscreenChange, { capture: true } as any);
      document.removeEventListener('webkitfullscreenchange', onFullscreenChange as any, { capture: true } as any);
      ro.disconnect();

      if (group) disposeObject(group);
      baseMaterial.dispose();
      selectedMaterial.dispose();
      renderer.dispose();

      try {
        container.removeChild(renderer.domElement);
      } catch {
        // ignore
      }

      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
      groupRef.current = null;
      baseMaterialRef.current = null;
      selectedMaterialRef.current = null;
      linesRef.current = [];
      if (idleTimerRef.current) window.clearTimeout(idleTimerRef.current);
      autoSpinRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Re-measure canvas when fullscreen state flips (some platforms skip the resize event).
    const container = containerRef.current;
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    if (!container || !renderer || !camera) return;
    const resize = () => {
      const w = Math.max(1, container.clientWidth || window.innerWidth || 1);
      const h = Math.max(1, container.clientHeight || window.innerHeight || 1);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    window.setTimeout(resize, 30);
    window.setTimeout(resize, 120);
  }, [isFullscreen]);

  useEffect(() => {
    const group = groupRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const baseMaterial = baseMaterialRef.current;

    if (!group || !scene || !camera || !controls || !baseMaterial) return;

    // clear existing
    linesRef.current.forEach(line => {
      const mat = (line as any).material as THREE.Material | undefined;
      if (mat && mat !== baseMaterial && mat !== selectedMaterial) {
        try {
          mat.dispose();
        } catch {
          // ignore
        }
      }
      line.geometry?.dispose();
    });
    group.clear();
    linesRef.current = [];
    setSelected(null);

    if (!streamlines.length) return;

    // Add a faint bounding cube to verify geometry is in view.
    const boundsMaterial = new THREE.LineBasicMaterial({ color: new THREE.Color('#2f3645'), transparent: true, opacity: 0.35 });
    const boundsGeom = new THREE.BufferGeometry();
    const boxVerts = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
      [0, 0, 0],
      [0, 0, 1],
      [1, 0, 1],
      [1, 1, 1],
      [0, 1, 1],
      [0, 0, 1],
      [1, 0, 1],
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
      [0, 1, 1],
      [0, 1, 0],
    ].flat();
    boundsGeom.setAttribute('position', new THREE.Float32BufferAttribute(boxVerts, 3));
    const boundsLine = new THREE.Line(boundsGeom, boundsMaterial);
    boundsLine.visible = false; // toggled on once we know bounds
    group.add(boundsLine);

    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity,
      maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (let i = 0; i < streamlines.length; i++) {
      const sl = streamlines[i];
      if (!Array.isArray(sl) || sl.length < 2) continue;

      const points: THREE.Vector3[] = new Array(sl.length);
      for (let j = 0; j < sl.length; j++) {
        const p = sl[j];
        const x = Number(p?.[0] ?? 0);
        const y = Number(p?.[1] ?? 0);
        const z = Number(p?.[2] ?? 0);
        points[j] = new THREE.Vector3(x, y, z);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
      }

      const color = colours[i];
      const usePerLineColour = Array.isArray(color) && color.length >= 3 && color.every(c => Number.isFinite(c));
      const material = usePerLineColour
        ? new THREE.LineBasicMaterial({ color: new THREE.Color(color[0], color[1], color[2]), transparent: false, opacity: 1 })
        : baseMaterial;

      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, material);
      line.userData.streamlineIndex = i;
      line.userData.baseMaterial = material;
      group.add(line);
      linesRef.current.push(line);
    }

    if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return;

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    const sizeX = Math.max(1e-6, maxX - minX);
    const sizeY = Math.max(1e-6, maxY - minY);
    const sizeZ = Math.max(1e-6, maxZ - minZ);
    const maxDim = Math.max(sizeX, sizeY, sizeZ);

    // heuristic pick threshold (world units)
    pickThresholdRef.current = maxDim * 0.01;

    const fov = THREE.MathUtils.degToRad(camera.fov);
    const dist = (maxDim * 1.05) / Math.tan(fov / 2);

    // View from the front (anterior) looking posteriorly: place camera in -Y looking +Y.
    camera.position.set(cx, cy - dist, cz);
    camera.up.set(0, 0, 1);
    camera.near = Math.max(0.001, dist / 100);
    camera.far = dist * 100;
    camera.updateProjectionMatrix();

    controls.target.set(cx, cy, cz);
    controls.minDistance = dist / 8;
    controls.maxDistance = dist * 2.2;
    controls.update();

    // Position and scale the debug cube around the data.
    boundsLine.position.set(minX, minY, minZ);
    boundsLine.scale.set(sizeX, sizeY, sizeZ);
    boundsLine.visible = true;
  }, [streamlines]);

  useEffect(() => {
    const rotateBy = (dYaw: number, dPitch: number) => {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      if (!camera || !controls) return;

      const target = controls.target.clone();
      const offset = camera.position.clone().sub(target);
      const distance = offset.length() || 1;

      // Yaw around world Z (patient head-foot axis)
      const qYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), dYaw);
      offset.applyQuaternion(qYaw);

      // Pitch around camera-right axis (perpendicular to up and view)
      const up = new THREE.Vector3(0, 0, 1);
      const right = new THREE.Vector3().crossVectors(offset, up).normalize();
      if (right.lengthSq() > 1e-6) {
        const qPitch = new THREE.Quaternion().setFromAxisAngle(right, dPitch);
        offset.applyQuaternion(qPitch);
      }

      // Avoid gimbal lock near poles
      const dir = offset.clone().normalize();
      const dotUp = Math.abs(dir.dot(new THREE.Vector3(0, 0, 1)));
      if (dotUp > 0.999) {
        return;
      }

      offset.setLength(distance);
      camera.position.copy(target).add(offset);
      camera.up.set(0, 0, 1);
      camera.lookAt(target);
      controls.update();
    };

    const isTextInput = (el: EventTarget | null) => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName.toLowerCase();
      return tag === 'input' || tag === 'textarea' || el.isContentEditable;
    };

    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.defaultPrevented) return;
      if (isTextInput(ev.target)) return;
      const stepAz = 0.3;
      const stepPol = 0.22;

      switch (ev.key) {
        case 'ArrowLeft':
          ev.preventDefault();
          rotateBy(-stepAz, 0);
          break;
        case 'ArrowRight':
          ev.preventDefault();
          rotateBy(stepAz, 0);
          break;
        case 'ArrowUp':
          ev.preventDefault();
          rotateBy(0, -stepPol);
          break;
        case 'ArrowDown':
          ev.preventDefault();
          rotateBy(0, stepPol);
          break;
        case 'Escape':
          cssFullscreenRef.current = false;
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', onKeyDown, { capture: true });
    return () => window.removeEventListener('keydown', onKeyDown, { capture: true } as any);
  }, []);

  return (
    <Card className="border-0 shadow-sm">
      <div className="p-5">
        <div className="mb-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div>
              <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground">Tractography</h2>
              <div className="text-xs text-muted-foreground">
                {loading
                  ? 'Loading streamlines…'
                  : error
                    ? 'Unavailable'
                    : streamlines.length
                      ? `${streamlines.length} streamlines${selected != null ? ` • selected ${selected + 1}` : ''}`
                      : 'No tractography data found.'}
              </div>
            </div>
            <button
              type="button"
              onClick={() => {
                const el = containerRef.current;
                const target = el || rendererRef.current?.domElement || document.documentElement;
                const exit =
                  document.exitFullscreen ||
                  (document as any).webkitExitFullscreen ||
                  (document as any).mozCancelFullScreen ||
                  (document as any).msExitFullscreen;

                const request =
                  (target as any).requestFullscreen ||
                  (target as any).webkitRequestFullscreen ||
                  (target as any).mozRequestFullScreen ||
                  (target as any).msRequestFullscreen;

                if (!document.fullscreenElement && !(document as any).webkitFullscreenElement) {
                  if (request) {
                    Promise.resolve(request.call(target)).catch(() => {
                      // Fallback: CSS fullscreen if the API is denied in Tauri/WebView
                      cssFullscreenRef.current = true;
                      setIsFullscreen(true);
                    });
                  } else {
                    cssFullscreenRef.current = true;
                    setIsFullscreen(true);
                  }
                } else if (exit) {
                  Promise.resolve(exit.call(document)).catch(() => {
                    cssFullscreenRef.current = false;
                    setIsFullscreen(false);
                  });
                } else {
                  cssFullscreenRef.current = false;
                  setIsFullscreen(false);
                }
              }}
              className="rounded-md border border-border px-3 py-1 text-xs font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            >
              {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
            </button>
          </div>
          {data?.path ? (
            <div className="mono text-xs text-muted-foreground">
              {data.path}
              {data?.returned != null && data?.totalStreamlines != null
                ? ` • returned ${data.returned}/${data.totalStreamlines}`
                : ''}
            </div>
          ) : null}
        </div>

        {error ? <div className="text-sm text-destructive">{error}</div> : null}
        {data?.error ? <div className="text-xs text-amber-600">{data.error}</div> : null}

        <div
          ref={containerRef}
          className={`relative ${isFullscreen ? 'fixed inset-0 z-50 h-screen w-screen' : 'h-[520px] w-full'} rounded-md border border-border bg-card`}
          style={isFullscreen ? { backgroundColor: '#060912' } : undefined}
        />
      </div>
    </Card>
  );
}
