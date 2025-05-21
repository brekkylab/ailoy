const fs = require("fs");
const path = require("path");

const pkg = require("../package.json");
const abi = process.versions.modules;
const platform = process.platform;
const arch = process.arch;
const runtime = "node";

const name = pkg.name.replace(/^@/, "").replace(/\//, "-");
const prebuildDir = path.resolve(
  __dirname,
  "..",
  "prebuilds",
  `${name}-${runtime}-v${abi}-${platform}-${arch}`,
  "build",
  "Release"
);
const distDir = path.resolve(__dirname, "..", "dist");

if (!fs.existsSync(prebuildDir)) {
  console.warn(`⚠️ No prebuilt binary found at ${prebuildDir}`);
  process.exit(1);
}

fs.mkdirSync(distDir, { recursive: true });

const files = fs.readdirSync(prebuildDir);
for (const file of files) {
  const from = path.join(prebuildDir, file);
  const to = path.join(distDir, file);
  fs.copyFileSync(from, to);
  console.log(`✔ Copied ${file} → dist/`);
}

console.log("✅ Native addon installed.");

// Optional cleanup
const prebuildRoot = path.resolve(__dirname, "..", "prebuilds");
fs.rmSync(prebuildRoot, { recursive: true, force: true });
console.log("🧹 Cleaned up prebuilds/");
