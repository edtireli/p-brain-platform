import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

function findBuiltApp(macosBundleDir) {
  if (!fs.existsSync(macosBundleDir)) return null;

  const entries = fs
    .readdirSync(macosBundleDir)
    .filter((name) => name.endsWith(".app"))
    .sort();

  if (entries.length === 0) return null;

  const preferred = entries.find((n) => n === "p-brain.app");
  return path.join(macosBundleDir, preferred ?? entries[0]);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

const cwd = process.cwd();

const srcBackendPath = path.join(
  cwd,
  "src-tauri",
  "resources",
  "backend",
  "pbrain-web-backend"
);

const macosBundleDir = path.join(
  cwd,
  "src-tauri",
  "target",
  "release",
  "bundle",
  "macos"
);

if (!fs.existsSync(srcBackendPath)) {
  console.error(
    `Backend not found: ${srcBackendPath}\nDid prepare.mjs run successfully?`
  );
  process.exit(1);
}

const appPath = findBuiltApp(macosBundleDir);
if (!appPath) {
  console.error(`No .app found under: ${macosBundleDir}`);
  process.exit(1);
}

const destBackendPath = path.join(
  appPath,
  "Contents",
  "Resources",
  "resources",
  "backend",
  "pbrain-web-backend"
);

ensureDir(path.dirname(destBackendPath));

const srcStat = fs.lstatSync(srcBackendPath);
if (srcStat.isDirectory()) {
  // Use rsync to preserve symlinks/perms and keep updates fast.
  const src = srcBackendPath.endsWith(path.sep) ? srcBackendPath : srcBackendPath + path.sep;
  const dest = destBackendPath.endsWith(path.sep) ? destBackendPath : destBackendPath + path.sep;
  try {
    execFileSync("rsync", ["-a", "--delete", src, dest], { stdio: "inherit" });
  } catch (e) {
    console.error("Failed to rsync backend into app bundle.");
    throw e;
  }
  console.log(`Bundled backend directory into: ${destBackendPath}`);
} else {
  try {
    fs.copyFileSync(srcBackendPath, destBackendPath);
    // Ensure executable bit is set (PyInstaller produces an executable).
    fs.chmodSync(destBackendPath, 0o755);
  } catch (e) {
    console.error("Failed to copy backend executable into app bundle.");
    throw e;
  }
  console.log(`Bundled backend executable into: ${destBackendPath}`);
}
