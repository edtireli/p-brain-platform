import * as nifti from 'nifti-reader-js';

export type ParsedNiftiVolume = {
  dims: [number, number, number, number];
  pixDims: [number, number, number];
  data: Float32Array;
  min: number;
  max: number;
};

function toFloat32(header: any, imageBuffer: ArrayBuffer): Float32Array {
  // nifti-reader-js returns raw bytes; choose a typed view based on datatype.
  const code = Number(header?.datatypeCode ?? header?.datatype ?? 0);

  let arr: ArrayLike<number>;
  switch (code) {
    case (nifti as any).NIFTI1.TYPE_UINT8:
      arr = new Uint8Array(imageBuffer);
      break;
    case (nifti as any).NIFTI1.TYPE_INT16:
      arr = new Int16Array(imageBuffer);
      break;
    case (nifti as any).NIFTI1.TYPE_INT32:
      arr = new Int32Array(imageBuffer);
      break;
    case (nifti as any).NIFTI1.TYPE_FLOAT32:
      return new Float32Array(imageBuffer);
    case (nifti as any).NIFTI1.TYPE_FLOAT64: {
      const src = new Float64Array(imageBuffer);
      const out = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) out[i] = src[i];
      return out;
    }
    case (nifti as any).NIFTI1.TYPE_INT8:
      arr = new Int8Array(imageBuffer);
      break;
    case (nifti as any).NIFTI1.TYPE_UINT16:
      arr = new Uint16Array(imageBuffer);
      break;
    case (nifti as any).NIFTI1.TYPE_UINT32:
      arr = new Uint32Array(imageBuffer);
      break;
    default:
      // Best-effort fallback.
      return new Float32Array(imageBuffer);
  }

  const out = new Float32Array((arr as any).length ?? 0);
  for (let i = 0; i < out.length; i++) out[i] = Number((arr as any)[i]);
  return out;
}

function sampleMinMax(data: Float32Array): { min: number; max: number } {
  if (!data.length) return { min: 0, max: 0 };
  // Sample up to ~1e6 points for speed.
  const step = Math.max(1, Math.floor(data.length / 1_000_000));
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < data.length; i += step) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 0 };
  return { min, max };
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch NIfTI (${res.status})`);
  return await res.arrayBuffer();
}

export async function loadNiftiFromUrl(url: string): Promise<ParsedNiftiVolume> {
  const raw = await fetchArrayBuffer(url);
  const data = (nifti as any).isCompressed(raw) ? (nifti as any).decompress(raw) : raw;
  if (!(nifti as any).isNIFTI(data)) throw new Error('Not a NIfTI file');

  const header = (nifti as any).readHeader(data);
  const image = (nifti as any).readImage(header, data);
  const floatData = toFloat32(header, image);

  const x = Number(header?.dims?.[1] ?? header?.dim?.[1] ?? 0);
  const y = Number(header?.dims?.[2] ?? header?.dim?.[2] ?? 0);
  const z = Number(header?.dims?.[3] ?? header?.dim?.[3] ?? 0);
  const t = Number(header?.dims?.[4] ?? header?.dim?.[4] ?? 1);
  const dims: [number, number, number, number] = [x || 0, y || 0, z || 0, t || 1];

  const px = Number(header?.pixDims?.[1] ?? header?.pixdim?.[1] ?? 1);
  const py = Number(header?.pixDims?.[2] ?? header?.pixdim?.[2] ?? 1);
  const pz = Number(header?.pixDims?.[3] ?? header?.pixdim?.[3] ?? 1);
  const pixDims: [number, number, number] = [px || 1, py || 1, pz || 1];

  const { min, max } = sampleMinMax(floatData);
  return { dims, pixDims, data: floatData, min, max };
}

export function sliceZ(
  vol: ParsedNiftiVolume,
  z: number,
  t: number
): number[][] {
  const [X, Y, Z, T] = vol.dims;
  const zz = Math.max(0, Math.min(Z - 1, z));
  const tt = Math.max(0, Math.min((T || 1) - 1, t));

  const slice = Array.from({ length: Y }, () => new Array(X).fill(0));
  const planeSize = X * Y;
  const volSize = planeSize * Z;
  const base = tt * volSize + zz * planeSize;

  for (let y = 0; y < Y; y++) {
    const rowBase = base + y * X;
    const row = slice[y];
    for (let x = 0; x < X; x++) {
      row[x] = vol.data[rowBase + x];
    }
  }

  return slice;
}

// Simple in-memory cache for parsed volumes.
const _cache = new Map<string, Promise<ParsedNiftiVolume>>();

export function cachedLoadNifti(url: string): Promise<ParsedNiftiVolume> {
  const existing = _cache.get(url);
  if (existing) return existing;
  const p = loadNiftiFromUrl(url);
  _cache.set(url, p);
  return p;
}
