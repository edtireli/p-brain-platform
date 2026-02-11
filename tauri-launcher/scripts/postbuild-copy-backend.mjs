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

const srcBackendDirPath = path.join(
  cwd,
  "src-tauri",
  "resources",
  "backend",
  "pbrain-web-backend"
);

const srcBackendZipPath = path.join(
  cwd,
  "src-tauri",
  "resources",
  "backend",
  "pbrain-web-backend.zip"
);

const macosBundleDir = path.join(
  cwd,
  "src-tauri",
  "target",
  "release",
  "bundle",
  "macos"
);

const hasZip = fs.existsSync(srcBackendZipPath);
const hasDir = fs.existsSync(srcBackendDirPath);

if (!hasZip && !hasDir) {
  console.error(
    `Backend not found:\n- ${srcBackendZipPath}\n- ${srcBackendDirPath}\nDid prepare.mjs run successfully?`
  );
  process.exit(1);
}

const appPath = findBuiltApp(macosBundleDir);
if (!appPath) {
  console.error(`No .app found under: ${macosBundleDir}`);
  process.exit(1);
}

if (hasZip) {
  const destZipPath = path.join(
    appPath,
    "Contents",
    "Resources",
    "resources",
    "backend",
    "pbrain-web-backend.zip"
  );

  ensureDir(path.dirname(destZipPath));

  try {
    fs.copyFileSync(srcBackendZipPath, destZipPath);
  } catch (e) {
    console.error("Failed to copy backend zip into app bundle.");
    throw e;
  }

  console.log(`Bundled backend zip into: ${destZipPath}`);
} else {
  // Legacy fallback: directory exists but zip does not.
  // Copy the directory into the app bundle so older builds still work.
  const destBackendPath = path.join(
    appPath,
    "Contents",
    "Resources",
    "resources",
    "backend",
    "pbrain-web-backend"
  );

  ensureDir(path.dirname(destBackendPath));

  const src = srcBackendDirPath.endsWith(path.sep)
    ? srcBackendDirPath
    : srcBackendDirPath + path.sep;
  const dest = destBackendPath.endsWith(path.sep)
    ? destBackendPath
    : destBackendPath + path.sep;

  try {
    execFileSync("rsync", ["-a", "--delete", src, dest], { stdio: "inherit" });
  } catch (e) {
    console.error("Failed to rsync backend directory into app bundle.");
    throw e;
  }

  console.log(`Bundled backend directory into: ${destBackendPath}`);
}
