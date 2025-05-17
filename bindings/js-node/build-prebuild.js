const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");

const sharedLibFiles = [
  "build/Release/libtvm_runtime.dylib",
  "build/Release/libtvm_runtime.so",
  "build/Release/tvm_runtime.dll",
  "build/Release/mlc_llm.dll",
];

// === Step 1: Run `npx prebuild ...` ===

const buildArgs = [
  "prebuild",
  "--napi",
  "--backend",
  "cmake-js",
  "--strip",
  "--",
  "-d",
  "../..",
  "-O",
  "build",
  "--CDNODE:BOOL=ON",
  "--CDAILOY_WITH_TEST:BOOL=OFF",
  `--parallel ${require("os").cpus().length}`,
];

const buildResult = spawnSync("npx", buildArgs, {
  stdio: "inherit",
  cwd: path.resolve(__dirname),
});

if (buildResult.error || buildResult.status !== 0) {
  console.error("prebuild failed.");
  process.exit(buildResult.status || 1);
}

// === Step 2: Patch the .tar.gz to include shared libraries ===

const prebuildsDir = path.resolve(__dirname, "prebuilds");

if (!fs.existsSync(prebuildsDir)) {
  console.error("prebuilds/ directory not found.");
  process.exit(1);
}

// Filter files that exist
const existingLibs = sharedLibFiles.filter((f) =>
  fs.existsSync(path.resolve(__dirname, f))
);

if (existingLibs.length === 0) {
  console.log("No shared libraries found to include. Skipping patch.");
  process.exit(0);
}

// Find the first prebuilt .tar.gz archive
const tarFiles = fs
  .readdirSync(prebuildsDir)
  .filter((f) => f.endsWith(".tar.gz"));
if (tarFiles.length === 0) {
  console.error("No .tar.gz file found in prebuilds/");
  process.exit(1);
}

const tarFilePath = path.join(prebuildsDir, tarFiles[0]);
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "prebuild-unpack-"));

console.log(`Unpacking ${tarFilePath}...`);
spawnSync("tar", ["-xzf", tarFilePath, "-C", tmpDir], { stdio: "inherit" });

// Recursively find the Release directory
function findReleaseDir(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory() && entry.name === "Release") {
      return fullPath;
    } else if (entry.isDirectory()) {
      const nested = findReleaseDir(fullPath);
      if (nested) return nested;
    }
  }
  return null;
}

const releaseDir = findReleaseDir(tmpDir);
if (!releaseDir) {
  console.error("Could not find Release directory inside unpacked archive");
  process.exit(1);
}

// Copy each existing shared library to Release/
for (const relPath of existingLibs) {
  const absPath = path.resolve(__dirname, relPath);
  const targetPath = path.join(releaseDir, path.basename(relPath));
  console.log(`Copying ${absPath} → ${targetPath}`);
  fs.copyFileSync(absPath, targetPath);
}

// Backup original .tar.gz
const backupPath = `${tarFilePath}.bak`;
fs.renameSync(tarFilePath, backupPath);
console.log(`Backup saved as ${backupPath}`);

console.log(`Repacking to ${tarFilePath}...`);
spawnSync("tar", ["-czf", tarFilePath, "-C", tmpDir, "."], {
  stdio: "inherit",
});

fs.rmSync(tmpDir, { recursive: true, force: true });

console.log("Prebuild and fix completed successfully.");
