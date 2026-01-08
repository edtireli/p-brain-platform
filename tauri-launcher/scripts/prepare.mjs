import fs from 'node:fs';
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

const python = pickPython(webRoot);
const pyVer = pythonVersion(python);
console.log(`Using Python for PyInstaller: ${python}${pyVer ? ` (v${pyVer.raw})` : ''}`);

console.log('Installing PyInstaller (dev requirement)…');
run(python, ['-m', 'pip', 'install', '-r', path.join(webRoot, 'backend', 'dev-requirements.txt')], {
  cwd: webRoot,
});

console.log('Building backend binary (PyInstaller)…');
rmrf(outBackend);
fs.mkdirSync(outBackend, { recursive: true });
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
  '--onefile',
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

// PyInstaller emits a single binary into resources/backend/.
const produced = [
  path.join(outBackend, 'pbrain-web-backend'),
  path.join(outBackend, 'pbrain-web-backend.exe'),
];
const producedPath = produced.find(exists);
if (!producedPath) {
  throw new Error('Expected PyInstaller output not found in resources/backend/.');
}

// Normalize name for bundling.
if (producedPath.endsWith('.exe')) {
  copyFile(producedPath, path.join(outBackend, 'pbrain-web-backend'));
}

console.log('Done.');
