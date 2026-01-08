const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const net = require('net');

const DEFAULT_PORT = 8787;

function fileExists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function httpGetOk(url, timeoutMs = 800) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve(res.statusCode && res.statusCode >= 200 && res.statusCode < 300);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForHealth(baseUrl, timeoutMs = 12000) {
  const start = Date.now();
  const healthUrl = `${baseUrl}/health`;
  while (Date.now() - start < timeoutMs) {
    // eslint-disable-next-line no-await-in-loop
    const ok = await httpGetOk(healthUrl);
    if (ok) return true;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

function resolveDevPython(repoRoot) {
  const venvPython = path.join(repoRoot, '.venv', 'bin', 'python');
  if (fileExists(venvPython)) return venvPython;
  return 'python3';
}

function tryListen(host, port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.on('error', () => resolve(null));
    server.listen({ host, port }, () => {
      const addr = server.address();
      const picked = addr && typeof addr === 'object' ? addr.port : null;
      server.close(() => resolve(typeof picked === 'number' ? picked : null));
    });
  });
}

async function pickPort(host, preferredPort) {
  const preferred = await tryListen(host, preferredPort);
  if (preferred === preferredPort) return preferredPort;
  const any = await tryListen(host, 0);
  return typeof any === 'number' ? any : preferredPort;
}

async function spawnBackend() {
  const host = '127.0.0.1';
  const requestedPort = Number(process.env.PBRAIN_PORT || DEFAULT_PORT);
  const port = await pickPort(host, requestedPort);

  const isPackaged = app.isPackaged;

  if (isPackaged) {
    const backendExe = path.join(process.resourcesPath, 'backend', 'pbrain-web-backend', 'pbrain-web-backend');
    const uiDir = path.join(process.resourcesPath, 'ui');

    const child = spawn(backendExe, [], {
      env: {
        ...process.env,
        PBRAIN_HOST: host,
        PBRAIN_PORT: String(port),
        PBRAIN_WEB_DIST: uiDir,
      },
      stdio: 'ignore',
    });

    return { child, baseUrl: `http://${host}:${port}` };
  }

  // Dev: run uvicorn from source.
  const repoRoot = path.resolve(__dirname, '..');
  const backendDir = path.join(repoRoot, 'backend');
  const distDir = path.join(repoRoot, 'dist');

  const python = resolveDevPython(repoRoot);
  const child = spawn(
    python,
    ['-m', 'uvicorn', 'app:app', '--host', host, '--port', String(port)],
    {
      cwd: backendDir,
      env: {
        ...process.env,
        PBRAIN_HOST: host,
        PBRAIN_PORT: String(port),
        PBRAIN_WEB_DIST: distDir,
      },
      stdio: 'inherit',
    }
  );

  return { child, baseUrl: `http://${host}:${port}` };
}

let backendProcess = null;

async function createWindow(baseUrl) {
  const win = new BrowserWindow({
    width: 1280,
    height: 840,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      sandbox: false,
    },
  });

  await win.loadURL(baseUrl);
  return win;
}

function killBackend() {
  if (!backendProcess) return;
  try {
    backendProcess.kill('SIGTERM');
  } catch {
    // ignore
  }
  backendProcess = null;
}

app.on('before-quit', () => {
  killBackend();
});

ipcMain.handle('pick-folder', async () => {
  const res = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });
  if (res.canceled) return null;
  return res.filePaths && res.filePaths[0] ? res.filePaths[0] : null;
});

app.whenReady().then(async () => {
  const { child, baseUrl } = await spawnBackend();
  backendProcess = child;

  const ok = await waitForHealth(baseUrl);
  if (!ok) {
    dialog.showErrorBox(
      'p-brain failed to start',
      `The local backend did not become ready at ${baseUrl}.`
    );
    killBackend();
    app.quit();
    return;
  }

  await createWindow(baseUrl);

  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await createWindow(baseUrl);
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
