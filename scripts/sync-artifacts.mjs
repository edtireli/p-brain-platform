import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createClient } from '@supabase/supabase-js';

function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) {
      out._.push(a);
      continue;
    }
    const key = a.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

function usage(msg) {
  if (msg) console.error(msg);
  console.error(
    [
      'Usage:',
      '  node scripts/sync-artifacts.mjs --project <projectId> --subject-dir <path> [--bucket <bucket>]',
      '',
      'Env:',
      '  SUPABASE_URL',
      '  SUPABASE_SERVICE_ROLE_KEY   (recommended; needs storage + table access)',
      '  (optional) SUPABASE_ANON_KEY (only if your RLS/storage permits)',
      '',
      'Notes:',
      '  - Uploads a small set of p-brain outputs so the GitHub Pages UI can render in Supabase-only mode.',
      '  - Upload paths are normalized to avoid spaces and macOS junk files.',
    ].join('\n')
  );
  process.exit(1);
}

function isJunk(name) {
  return name === '.DS_Store' || name.startsWith('._');
}

async function exists(p) {
  try {
    await fs.stat(p);
    return true;
  } catch {
    return false;
  }
}

async function listFilesRecursive(root, exts) {
  const out = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      if (isJunk(e.name)) continue;
      const full = path.join(dir, e.name);
      if (e.isDirectory()) await walk(full);
      else {
        const lower = e.name.toLowerCase();
        if (!exts || exts.some(x => lower.endsWith(x))) out.push(full);
      }
    }
  }
  if (await exists(root)) await walk(root);
  return out;
}

function normalizeRel(rel) {
  // Convert Windows separators + strip leading ./
  const r = rel.split(path.sep).join('/');
  return r.replace(/^\.\//, '');
}

function decodeNpy(buffer) {
  // Minimal NumPy .npy v1/v2 reader for little-endian float32/float64 arrays.
  // Ref: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
  const magic = buffer.subarray(0, 6).toString('binary');
  if (magic !== '\x93NUMPY') throw new Error('Not a .npy file');
  const major = buffer[6];
  const minor = buffer[7];
  let headerLen;
  let offset;
  if (major === 1) {
    headerLen = buffer.readUInt16LE(8);
    offset = 10;
  } else if (major === 2) {
    headerLen = buffer.readUInt32LE(8);
    offset = 12;
  } else {
    throw new Error(`Unsupported .npy version ${major}.${minor}`);
  }

  const header = buffer.subarray(offset, offset + headerLen).toString('latin1');
  const descrMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^\)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) throw new Error('Invalid .npy header');

  const descr = descrMatch[1];
  const fortran = fortranMatch[1] === 'True';
  if (fortran) throw new Error('Fortran-ordered arrays not supported');

  const shapeParts = shapeMatch[1]
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .map(s => Number(s));
  const count = shapeParts.reduce((a, b) => a * b, 1);

  const dataOffset = offset + headerLen;
  const data = buffer.subarray(dataOffset);

  // Support: <f4, <f8
  if (descr === '<f4') {
    const out = new Array(count);
    for (let i = 0; i < count; i++) out[i] = data.readFloatLE(i * 4);
    return { shape: shapeParts, data: out };
  }
  if (descr === '<f8') {
    const out = new Array(count);
    for (let i = 0; i < count; i++) out[i] = data.readDoubleLE(i * 8);
    return { shape: shapeParts, data: out };
  }

  throw new Error(`Unsupported dtype ${descr}`);
}

async function main() {
  const args = parseArgs(process.argv);

  const projectId = args.project;
  const subjectDir = args['subject-dir'];
  const bucket = args.bucket || process.env.SUPABASE_BUCKET || 'pbrain';
  if (!projectId || !subjectDir) usage('Missing --project or --subject-dir');

  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRole = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anon = process.env.SUPABASE_ANON_KEY;
  const key = serviceRole || anon;
  if (!supabaseUrl || !key) usage('Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)');

  const sb = createClient(supabaseUrl, key, { auth: { persistSession: false } });

  const subjectName = path.basename(subjectDir);
  const { data: subjects, error: sErr } = await sb
    .from('subjects')
    .select('id, name, source_path')
    .eq('project_id', projectId);
  if (sErr) throw sErr;

  const match = (subjects || []).find(s => s.source_path === subjectDir) || (subjects || []).find(s => s.name === subjectName);
  if (!match) {
    throw new Error(
      `Could not find subject row in Supabase for project ${projectId}. Expected source_path=${subjectDir} or name=${subjectName}`
    );
  }

  const subjectId = match.id;

  const uploads = [];

  async function uploadFile(localPath, destPath, contentType) {
    const buf = await fs.readFile(localPath);
    const { error } = await sb.storage.from(bucket).upload(destPath, buf, {
      upsert: true,
      contentType,
      cacheControl: '3600',
    });
    if (error) throw error;
    uploads.push(destPath);
  }

  const montagesRoot = path.join(subjectDir, 'Images', 'AI', 'Montages');
  const montagePngs = await listFilesRecursive(montagesRoot, ['.png']);
  for (const f of montagePngs) {
    const name = path.basename(f);
    const dest = `projects/${projectId}/subjects/${subjectId}/images/ai/montages/${name}`;
    await uploadFile(f, dest, 'image/png');
  }

  const fitRoot = path.join(subjectDir, 'Images', 'Fit');
  const fitPngs = await listFilesRecursive(fitRoot, ['.png']);
  for (const f of fitPngs) {
    const name = path.basename(f);
    const dest = `projects/${projectId}/subjects/${subjectId}/images/fit/${name}`;
    await uploadFile(f, dest, 'image/png');
  }

  // Curves: convert a small, predictable set of .npy curves to JSON.
  const curves = [];

  async function addCurve(label, npyPath) {
    if (!(await exists(npyPath))) return;
    const buf = await fs.readFile(npyPath);
    const parsed = decodeNpy(buf);
    // Flatten to 1D
    const values = parsed.data;
    const timePoints = values.map((_, i) => i);
    curves.push({
      id: normalizeRel(label).replace(/[^a-z0-9_\-]+/gi, '_'),
      name: label,
      timePoints,
      values,
      unit: 'frame',
    });
  }

  await addCurve(
    'Artery CTC (slice 1)',
    path.join(subjectDir, 'Analysis', 'CTC Data', 'Artery', 'Right Interior Carotid', 'CTC_slice_1.npy')
  );
  await addCurve(
    'Grey Matter CTC (slice 5)',
    path.join(subjectDir, 'Analysis', 'CTC Data', 'Tissue', 'Grey Matter', 'CTC_slice_5.npy')
  );
  await addCurve(
    'White Matter CTC (slice 6)',
    path.join(subjectDir, 'Analysis', 'CTC Data', 'Tissue', 'White Matter', 'CTC_slice_6.npy')
  );

  const curvesJson = JSON.stringify({ curves }, null, 2);
  const curvesDest = `projects/${projectId}/subjects/${subjectId}/curves/curves.json`;
  const { error: cErr } = await sb.storage.from(bucket).upload(curvesDest, Buffer.from(curvesJson, 'utf8'), {
    upsert: true,
    contentType: 'application/json',
    cacheControl: '60',
  });
  if (cErr) throw cErr;
  uploads.push(curvesDest);

  const index = {
    subjectId,
    subjectName: match.name,
    uploadedAt: new Date().toISOString(),
    bucket,
    artifacts: {
      montages: montagePngs.length,
      fitImages: fitPngs.length,
      curves: curves.length,
    },
    paths: uploads,
  };

  const indexDest = `projects/${projectId}/subjects/${subjectId}/artifacts/index.json`;
  const { error: iErr } = await sb.storage.from(bucket).upload(indexDest, Buffer.from(JSON.stringify(index, null, 2), 'utf8'), {
    upsert: true,
    contentType: 'application/json',
    cacheControl: '60',
  });
  if (iErr) throw iErr;

  console.log(`Synced artifacts for subject ${match.name} (${subjectId})`);
  console.log(`Bucket: ${bucket}`);
  console.log(`Uploaded: ${uploads.length + 1} objects`);
  console.log(`Index: ${indexDest}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
