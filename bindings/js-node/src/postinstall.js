const fs = require("fs");
const path = require("path");

const buildDir = path.resolve(__dirname, "..", "build", "Release");
const distDir = path.resolve(__dirname, "..", "dist");
const prebuildsDir = path.resolve(__dirname, "..", "prebuilds");

if (!fs.existsSync(buildDir)) {
  console.warn(`No prebuilt binary found at ${buildDir}`);
  process.exit(1);
}

fs.mkdirSync(distDir, { recursive: true });

const files = fs.readdirSync(buildDir);
for (const file of files) {
  const from = path.join(buildDir, file);
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
