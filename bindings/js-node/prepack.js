const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const pkg = require("./package.json");

const binaryName = "ailoy_addon.node";
const libPattern = /\.(dylib|so(\.\d+)*|dll)$/i;

const version = pkg.version;
const name = pkg.name.replace(/^@/, "").replace(/\//, "-");
const abi = process.versions.modules;
const runtime = "node";
const platform = process.platform;
const arch = process.arch;

const srcDir = path.resolve(__dirname, "src");
const outDir = path.resolve(
  __dirname,
  "prebuilds",
  `${name}-${runtime}-v${abi}-${platform}-${arch}`,
  "build",
  "Release"
);

fs.mkdirSync(outDir, { recursive: true });

const allFiles = fs
  .readdirSync(srcDir)
  .filter((f) => f === binaryName || libPattern.test(f));

for (const file of allFiles) {
  const from = path.join(srcDir, file);
  const to = path.join(outDir, file);
  fs.copyFileSync(from, to);
  console.log(`✔ Moved ${file}`);
}

const nodeBinary = path.join(outDir, binaryName);
const libSet = new Set(allFiles.filter((f) => f !== binaryName));

// rpath 패치
if (platform === "darwin") {
  const otoolOut = spawnSync("otool", ["-L", nodeBinary], {
    encoding: "utf8",
    shell: true,
  });
  if (otoolOut.error) throw otoolOut.error;

  const lines = otoolOut.stdout.split("\n").slice(1);
  for (const line of lines) {
    const match = line.trim().match(/^(.+?\.dylib)/);
    if (!match) continue;

    const depPath = match[1];
    const base = path.basename(depPath);
    if (libSet.has(base)) {
      console.log(`→ patch: ${depPath} → @loader_path/${base}`);
      spawnSync(
        "install_name_tool",
        ["-change", depPath, `@loader_path/${base}`, nodeBinary],
        { shell: true }
      );
    }
  }

  spawnSync("install_name_tool", ["-add_rpath", "@loader_path", nodeBinary], {
    shell: true,
  });
} else if (platform === "linux") {
  const readelfOut = spawnSync("readelf", ["-d", nodeBinary], {
    encoding: "utf8",
    shell: true,
  });
  if (readelfOut.error) throw readelfOut.error;

  const lines = readelfOut.stdout.split("\n");
  const needed = lines
    .filter((l) => l.includes("(NEEDED)"))
    .map((l) => l.match(/\[(.+?)\]/)?.[1])
    .filter(Boolean);

  const matchLibs = needed.filter((name) => libSet.has(name));
  if (matchLibs.length > 0) {
    console.log(`→ patch: rpath set to $ORIGIN for ${matchLibs.join(", ")}`);
    spawnSync("patchelf", ["--set-rpath", "$ORIGIN", nodeBinary], {
      shell: true,
    });
  }
}

if (fs.existsSync(path.resolve(__dirname, "dist", "ailoy_addon.node")))
  fs.unlinkSync(path.resolve(__dirname, "dist", "ailoy_addon.node"));

console.log("✅ Prebuild directory ready.");
