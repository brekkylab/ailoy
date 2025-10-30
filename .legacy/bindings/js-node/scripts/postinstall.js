const fs = require("fs");
const path = require("path");

const baseDir = path.resolve(__dirname, "..");
const buildDir = path.resolve(baseDir, "build");
const buildReleaseDir = path.resolve(buildDir, "Release");
const distDir = path.resolve(baseDir, "dist");
const prebuildsDir = path.resolve(baseDir, "prebuilds");
const scriptsDir = path.resolve(baseDir, "scripts");

if (!fs.existsSync(buildReleaseDir)) {
  console.warn(`No prebuilt binary found at ${buildReleaseDir}`);
  process.exit(1);
}

fs.mkdirSync(distDir, { recursive: true });

const files = fs.readdirSync(buildReleaseDir);
for (const file of files) {
  const from = path.join(buildReleaseDir, file);
  const to = path.join(distDir, file);
  fs.copyFileSync(from, to);
}

// cleanup
if (fs.existsSync(buildDir)) {
  fs.rmSync(buildDir, { recursive: true, force: true });
}
if (fs.existsSync(prebuildsDir)) {
  fs.rmSync(prebuildsDir, { recursive: true, force: true });
}
if (fs.existsSync(scriptsDir)) {
  fs.rmSync(scriptsDir, { recursive: true, force: true });
}
