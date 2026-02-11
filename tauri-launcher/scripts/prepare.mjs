import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: 'inherit', ...opts });
  if (res.status !== 0) {
    throw new Error(`Command failed: ${cmd} ${args.join(' ')}`);
  }
}

function rmrf(p) {
  fs.rmSync(p, { recursive: true, force: true });
}

function copyFile(src, dst) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
}

function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function which(cmd) {
  const res = spawnSync('bash', ['-lc', `command -v ${cmd}`], { encoding: 'utf8' });
  if (res.status !== 0) return null;
  const out = String(res.stdout || '').trim();
  return out.length ? out : null;
}

function expandHome(p) {
  if (!p) return p;
  if (p === '~') return os.homedir();
  if (p.startsWith('~/')) return path.join(os.homedir(), p.slice(2));
  return p;
}

function looksLikeBackendBundle(dir) {
  if (!dir) return false;
  const exe = path.join(dir, 'pbrain-web-backend');
  const internal = path.join(dir, '_internal');
  return exists(exe) && exists(internal);
}

function truthy(v) {
  const s = String(v || '').trim().toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'y' || s === 'on';
}

function pythonVersion(pythonPath) {
  const res = spawnSync(pythonPath, ['-c', 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'], {
    encoding: 'utf8',
  });
  if (res.status !== 0) return null;
  const out = String(res.stdout || '').trim();
  if (!/^\d+\.\d+$/.test(out)) return null;
  const [maj, min] = out.split('.').map(Number);
  if (!Number.isFinite(maj) || !Number.isFinite(min)) return null;
  return { maj, min, raw: out };
}

function pickPython(webRoot) {
  if (process.env.PYTHON) return process.env.PYTHON;

  // Prefer a non-.venv python to avoid accidental Python 3.14 builds.
  const candidates = ['python3', 'python'];
  for (const c of candidates) {
    const v = pythonVersion(c);
    if (!v) continue;
    if (v.maj === 3 && v.min <= 12) return c;
  }

  // Fall back to repo venv only if it is also <= 3.12.
  const venvPython = path.join(webRoot, '.venv', 'bin', 'python');
  if (exists(venvPython)) {
    const v = pythonVersion(venvPython);
    if (v && v.maj === 3 && v.min <= 12) return venvPython;
  }

  // Last resort.
  return 'python3';
}

const launcherDir = path.resolve(import.meta.dirname, '..');
const webRoot = path.resolve(launcherDir, '..');

const outRoot = path.join(launcherDir, 'src-tauri', 'resources');
const outBackend = path.join(outRoot, 'backend');

console.log('Building frontend…');
run('npm', ['run', 'build'], { cwd: webRoot });

rmrf(outBackend);
fs.mkdirSync(outBackend, { recursive: true });

// Optional: reuse a previously-built backend bundle (PyInstaller onedir output)
// to avoid slow rebuilds when frontend changes only.
const prebuiltEnv = expandHome(process.env.PBRAIN_PREBUILT_BACKEND_DIR);
const allowDefaultPrebuilt = truthy(process.env.PBRAIN_USE_PREBUILT_BACKEND);
const prebuiltDefault = path.join(
  os.homedir(),
  'Library',
  'Application Support',
  'com.edt.pbrain',
  'backend',
  'pbrain-web-backend'
);

// Policy:
// - If PBRAIN_PREBUILT_BACKEND_DIR is set, treat it as an explicit request.
// - Otherwise only use the default prebuilt backend when PBRAIN_USE_PREBUILT_BACKEND=1.
const prebuiltCandidates = [prebuiltEnv, allowDefaultPrebuilt ? prebuiltDefault : null].filter(Boolean);
const prebuiltDir = prebuiltCandidates.find((p) => looksLikeBackendBundle(p));

const zipPath = path.join(outBackend, 'pbrain-web-backend.zip');
rmrf(zipPath);

const ditto = process.platform === 'darwin' ? '/usr/bin/ditto' : null;
function zipFolderKeepParent(folderPath) {
  if (ditto && exists(ditto)) {
    run(ditto, ['-c', '-k', '--sequesterRsrc', '--keepParent', folderPath, zipPath]);
    return;
  }
  const zipBin = which('zip');
  if (!zipBin) {
    throw new Error('Neither /usr/bin/ditto nor zip is available to create pbrain-web-backend.zip');
  }
  run(zipBin, ['-qry', zipPath, path.basename(folderPath)], { cwd: path.dirname(folderPath) });
}

if (prebuiltDir) {
  console.log(`Using prebuilt backend bundle: ${prebuiltDir}`);
  zipFolderKeepParent(prebuiltDir);
  console.log('Done.');
  process.exit(0);
}

const python = pickPython(webRoot);
const pyVer = pythonVersion(python);
console.log(`Using Python for PyInstaller: ${python}${pyVer ? ` (v${pyVer.raw})` : ''}`);

console.log('Installing PyInstaller (dev requirement)…');
run(python, ['-m', 'pip', 'install', '-r', path.join(webRoot, 'backend', 'dev-requirements.txt')], {
  cwd: webRoot,
});

console.log('Building backend binary (PyInstaller)…');
const excludedModules = [
  'torch',
  'torchvision',
  'torchaudio',
  'tensorflow',
  'tensorflow_macos',
  'keras',
  'onnx',
  'onnxruntime',
  'cv2',
  'opencv',
  'opencv-python',
  'PIL',
  'Pillow',
  'matplotlib',
  'pandas',
  'sklearn',
  'scikit-learn',
];

const pyInstallerArgs = [
  '-m',
  'PyInstaller',
  '--paths',
  path.join(webRoot, 'backend'),
  '--hidden-import',
  'anyio._backends._asyncio',
  '--name',
  'pbrain-web-backend',
  '--noupx',
  '--clean',
  '--distpath',
  outBackend,
  '--workpath',
  path.join(outRoot, 'pyi-work'),
  '--specpath',
  path.join(outRoot, 'pyi-spec'),
];

for (const mod of excludedModules) {
  pyInstallerArgs.push('--exclude-module', mod);
}

pyInstallerArgs.push(path.join(webRoot, 'backend', 'launcher_entry.py'));

run(python, pyInstallerArgs, { cwd: webRoot });

// PyInstaller (onedir) emits a folder into resources/backend/.
// macOS/Linux: resources/backend/pbrain-web-backend/pbrain-web-backend
// Windows: resources/backend/pbrain-web-backend/pbrain-web-backend.exe
const producedDir = path.join(outBackend, 'pbrain-web-backend');
if (!exists(producedDir)) {
  throw new Error('Expected PyInstaller output dir not found in resources/backend/pbrain-web-backend/.');
}

// Normalize Windows executable name for the launcher.
const winExe = path.join(producedDir, 'pbrain-web-backend.exe');
const normalized = path.join(producedDir, 'pbrain-web-backend');
if (exists(winExe) && !exists(normalized)) {
  copyFile(winExe, normalized);
}

// Bundle the onedir output as a single zip file so Tauri's build-time
// resource scanning doesn't have to traverse thousands of files.
// The launcher will extract this zip to the app data directory at runtime.
zipFolderKeepParent(producedDir);

rmrf(producedDir);

console.log('Done.');
