import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { createClient } from '@supabase/supabase-js';
import readline from 'node:readline';

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
      '  (optional) SUPABASE_ANON_KEY or SUPABASE_PUBLISHABLE_KEY (user-mode upload; requires Storage policies)',
      '  (user-mode) SUPABASE_EMAIL / SUPABASE_PASSWORD (or you will be prompted)',
      '',
      'Notes:',
      '  - Uploads a small set of p-brain outputs so the GitHub Pages UI can render in Supabase-only mode.',
      '  - Upload paths are normalized to avoid spaces and macOS junk files.',
    ].join('\n')
  );
  process.exit(1);
}

async function promptHidden(prompt) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    const mutableOut = rl.output;
    const write = mutableOut.write.bind(mutableOut);
    mutableOut.write = (str, ...rest) => {
      if (rl.stdoutMuted) return true;
      return write(str, ...rest);
    };
    rl.stdoutMuted = false;

    const answer = await new Promise(resolve => {
      rl.question(prompt, a => resolve(a));
      rl.stdoutMuted = true;
    });
    write.call(mutableOut, '\n');
    return String(answer || '');
  } finally {
    rl.close();
  }
}

async function promptVisible(prompt) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  try {
    return await new Promise(resolve => rl.question(prompt, a => resolve(String(a || ''))));
  } finally {
    rl.close();
  }
}

function isJunk(name) {
  return name === '.DS_Store' || name.startsWith('._');
}

function isNiftiFilename(name) {
  const lower = name.toLowerCase();
  return lower.endsWith('.nii') || lower.endsWith('.nii.gz');
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

function parseCommaPatterns(raw) {
  return String(raw || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
}

function firstMatchingByPatterns(filenames, patterns) {
  const pats = patterns.map(p => {
    // Very small glob-to-regexp: treat '*' as '.*' and escape the rest.
    const re = '^' + p.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*') + '$';
    return new RegExp(re, 'i');
  });

  for (const pat of pats) {
    const hit = filenames.find(f => pat.test(f));
    if (hit) return hit;
  }
  return null;
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

  const uploadPngs = args['upload-pngs'] === false ? false : (String(args['upload-pngs'] || 'true').toLowerCase() !== 'false');
  const uploadNiftis = args['upload-niftis'] === false ? false : (String(args['upload-niftis'] || 'true').toLowerCase() !== 'false');
  const uploadAllSourceNiftis = String(args['upload-all-source-niftis'] || 'false').toLowerCase() === 'true' || args['upload-all-source-niftis'] === true;
  const maxUploadMb = Number(args['max-upload-mb'] || process.env.PBRAIN_MAX_UPLOAD_MB || 45);
  const maxUploadBytes = Number.isFinite(maxUploadMb) && maxUploadMb > 0 ? Math.floor(maxUploadMb * 1024 * 1024) : 45 * 1024 * 1024;

  const supabaseUrl = process.env.SUPABASE_URL;
  const serviceRole = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const anon = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_PUBLISHABLE_KEY;
  const key = serviceRole || anon;
  if (!supabaseUrl || !key) usage('Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)');

  const sb = createClient(supabaseUrl, key, { auth: { persistSession: false } });

  // Preflight bucket existence. (In admin mode we can auto-create it.)
  {
    const { error } = await sb.storage.from(bucket).list('', { limit: 1 });
    const msg = String((error && (error.message || error.error)) || '');
    const looksMissing = !!error && /bucket not found/i.test(msg);
    if (looksMissing) {
      if (!serviceRole) {
        throw new Error(
          [
            `Storage bucket "${bucket}" does not exist in this Supabase project.`,
            'Create it in Supabase Dashboard → Storage, or pass an existing bucket via --bucket / SUPABASE_BUCKET.',
            'Tip: your UI uses VITE_SUPABASE_STORAGE_BUCKET (defaults to "pbrain").',
          ].join(' ')
        );
      }

      const { error: cErr } = await sb.storage.createBucket(bucket, { public: false });
      if (cErr && !/already exists/i.test(String(cErr.message || ''))) throw cErr;

      const { error: lErr } = await sb.storage.from(bucket).list('', { limit: 1 });
      if (lErr) throw lErr;
    }
  }

  if (!serviceRole) {
    // User-mode: sign in so Storage policies can apply.
    const email = process.env.SUPABASE_EMAIL || (await promptVisible('Supabase email: '));
    const password = process.env.SUPABASE_PASSWORD || (await promptHidden('Supabase password (hidden): '));
    if (!email || !password) {
      throw new Error('Missing SUPABASE_EMAIL/SUPABASE_PASSWORD (set env vars for non-interactive runs)');
    }
    const { data, error } = await sb.auth.signInWithPassword({ email, password });
    if (error) throw error;
    if (!data?.session) throw new Error('Failed to obtain session');
  }

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
  const uploadedVolumes = [];
  const uploadedMaps = [];
  const uploadedMetrics = [];
  const skipped = [];

  async function uploadFile(localPath, destPath, contentType) {
    const st = await fs.stat(localPath);
    if (st.size > maxUploadBytes) {
      skipped.push({ localPath, destPath, reason: `too_large_${st.size}` });
      console.warn(`[skip] ${destPath} (${st.size} bytes) exceeds limit ${maxUploadBytes} bytes`);
      return false;
    }

    try {
      const buf = await fs.readFile(localPath);
      const { error } = await sb.storage.from(bucket).upload(destPath, buf, {
        upsert: true,
        contentType,
        cacheControl: '3600',
      });
      if (error) {
        const sc = String(error.statusCode || error.status || '');
        const msg = String(error.message || error.error || '');
        if (sc === '413' || /exceeded the maximum allowed size/i.test(msg)) {
          skipped.push({ localPath, destPath, reason: `storage_${sc || '413'}` });
          console.warn(`[skip] ${destPath} rejected by Storage (413 too large)`);
          return false;
        }
        throw error;
      }
      uploads.push(destPath);
      return true;
    } catch (err) {
      // If Storage is configured with a size cap, skip large objects instead of aborting the whole sync.
      const msg = String((err && err.message) || err || '');
      if (/413|exceeded the maximum allowed size/i.test(msg)) {
        skipped.push({ localPath, destPath, reason: 'storage_413' });
        console.warn(`[skip] ${destPath} rejected by Storage (413 too large)`);
        return false;
      }
      throw err;
    }
  }

  let montagePngs = [];
  let fitPngs = [];
  if (uploadPngs) {
    const montagesRoot = path.join(subjectDir, 'Images', 'AI', 'Montages');
    montagePngs = await listFilesRecursive(montagesRoot, ['.png']);
    for (const f of montagePngs) {
      const name = path.basename(f);
      const dest = `projects/${projectId}/subjects/${subjectId}/images/ai/montages/${name}`;
      await uploadFile(f, dest, 'image/png');
    }

    const fitRoot = path.join(subjectDir, 'Images', 'Fit');
    fitPngs = await listFilesRecursive(fitRoot, ['.png']);
    for (const f of fitPngs) {
      const name = path.basename(f);
      const dest = `projects/${projectId}/subjects/${subjectId}/images/fit/${name}`;
      await uploadFile(f, dest, 'image/png');
    }
  }

  // Source volumes (NIfTI directory): upload selected volumes so the Viewer can load them.
  if (uploadNiftis) {
    const niftiRoot = path.join(subjectDir, 'NIfTI');
    const allNifti = (await exists(niftiRoot)) ? (await fs.readdir(niftiRoot)).filter(f => !isJunk(f) && isNiftiFilename(f)) : [];

    // Patterns aligned with DEFAULT_FOLDER_STRUCTURE in p-brain-web.
    const t1Patterns = parseCommaPatterns(process.env.PBRAIN_T1_PATTERN || 'WIPcs_T1W_3D_TFE_32channel.nii*,*T1*.nii*');
    const t2Patterns = parseCommaPatterns(
      process.env.PBRAIN_T2_PATTERN ||
        'WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,ax*WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,WIPAxT2TSEmatrix.nii*,*T2*.nii*'
    );
    const flairPatterns = parseCommaPatterns(
      process.env.PBRAIN_FLAIR_PATTERN ||
        'WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,ax*WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,*FLAIR*.nii*'
    );
    const dcePatterns = parseCommaPatterns(process.env.PBRAIN_DCE_PATTERN || 'WIPDelRec-hperf120long.nii*,WIPhperf120long.nii*,*DCE*.nii*');
    const diffusionPatterns = parseCommaPatterns(
      process.env.PBRAIN_DIFFUSION_PATTERN ||
        'Reg-DWInySENSE.nii*,Reg-DWInySENSE_ADC.nii*,isoDWIb-1000*.nii*,WIPDTI_RSI_*.nii*,WIPDWI_RSI_*.nii*,*DTI*.nii*'
    );

    const t1File = firstMatchingByPatterns(allNifti, t1Patterns);
    const t2File = firstMatchingByPatterns(allNifti, t2Patterns);
    const flairFile = firstMatchingByPatterns(allNifti, flairPatterns);
    const dceFile = firstMatchingByPatterns(allNifti, dcePatterns);
    const diffFile = firstMatchingByPatterns(allNifti, diffusionPatterns);

    const wanted = [];
    if (t1File) wanted.push({ kind: 't1', file: t1File });
    if (t2File) wanted.push({ kind: 't2', file: t2File });
    if (flairFile) wanted.push({ kind: 'flair', file: flairFile });
    if (dceFile) wanted.push({ kind: 'dce', file: dceFile });
    if (diffFile) wanted.push({ kind: 'diffusion', file: diffFile });

    const toUpload = uploadAllSourceNiftis
      ? allNifti.map(f => ({ kind: 'source', file: f }))
      : wanted;

    for (const { kind, file } of toUpload) {
      const local = path.join(niftiRoot, file);
      const dest = `projects/${projectId}/subjects/${subjectId}/volumes/source/${file}`;
      const ok = await uploadFile(local, dest, 'application/octet-stream');
      if (ok) uploadedVolumes.push({ id: file, name: file, path: dest, kind });
    }
  }

  // Analysis map volumes (nii/nii.gz) - these are the "real" p-brain computed maps.
  if (uploadNiftis) {
    const analysisRoot = path.join(subjectDir, 'Analysis');
    const niftiLike = await listFilesRecursive(analysisRoot, ['.nii', '.nii.gz']);
    for (const f of niftiLike) {
      const rel = normalizeRel(path.relative(analysisRoot, f));
      const dest = `projects/${projectId}/subjects/${subjectId}/analysis/${rel}`;
      const ok = await uploadFile(f, dest, 'application/octet-stream');
      if (ok) uploadedMaps.push({ id: path.basename(f), name: path.basename(f), path: dest });
    }
  }

  // Metrics JSON (used by Tables): upload a small, stable set of outputs.
  {
    const analysisRoot = path.join(subjectDir, 'Analysis');
    const wanted = [
      'Ki_values_atlas_patlak.json',
      'Ki_values_atlas_tikhonov.json',
      'AI_values_median_patlak.json',
      'AI_values_median_tikhonov.json',
    ];

    for (const name of wanted) {
      const local = path.join(analysisRoot, name);
      if (!(await exists(local))) continue;
      const dest = `projects/${projectId}/subjects/${subjectId}/analysis/metrics/${name}`;
      const ok = await uploadFile(local, dest, 'application/json');
      if (ok) uploadedMetrics.push({ id: name, name, path: dest });
    }
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
      volumes: uploadedVolumes.length,
      maps: uploadedMaps.length,
      metrics: uploadedMetrics.length,
    },
    paths: uploads,
    volumes: uploadedVolumes,
    maps: uploadedMaps,
    metrics: uploadedMetrics,
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
  if (skipped.length) console.log(`Skipped: ${skipped.length} objects (size-limited)`);
  console.log(`Index: ${indexDest}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
