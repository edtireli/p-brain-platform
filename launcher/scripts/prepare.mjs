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

function copyDir(src, dst) {
  rmrf(dst);
  fs.mkdirSync(dst, { recursive: true });
  fs.cpSync(src, dst, { recursive: true });
}

function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

const args = new Set(process.argv.slice(2));
const isDev = args.has('--dev');

const launcherDir = path.resolve(import.meta.dirname, '..');
const webRoot = path.resolve(launcherDir, '..');

const distDir = path.join(webRoot, 'dist');
const outRoot = path.join(launcherDir, 'build-resources');
const outUi = path.join(outRoot, 'ui');
const outBackend = path.join(outRoot, 'backend');

console.log('Building frontend…');
run('npm', ['run', 'build'], { cwd: webRoot });

if (isDev) {
  console.log('Dev mode: skipping backend binary build.');
  process.exit(0);
}

console.log('Preparing launcher resources…');
copyDir(distDir, outUi);

const venvPython = path.join(webRoot, '.venv', 'bin', 'python');
const python = process.env.PYTHON || (exists(venvPython) ? venvPython : 'python3');

console.log('Installing PyInstaller (dev requirement)…');
run(python, ['-m', 'pip', 'install', '-r', path.join(webRoot, 'backend', 'dev-requirements.txt')], {
  cwd: webRoot,
});

console.log('Building backend binary (PyInstaller)…');
rmrf(outBackend);
fs.mkdirSync(outBackend, { recursive: true });

run(
  python,
  [
    '-m',
    'PyInstaller',
    '--name',
    'pbrain-web-backend',
    '--onedir',
    '--clean',
    '--distpath',
    outBackend,
    '--workpath',
    path.join(outRoot, 'pyi-work'),
    '--specpath',
    path.join(outRoot, 'pyi-spec'),
    path.join(webRoot, 'backend', 'launcher_entry.py'),
  ],
  { cwd: webRoot }
);

console.log('Done.');
