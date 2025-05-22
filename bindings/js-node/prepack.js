const fs = require("fs");
const path = require("path");
const { spawn, spawnSync } = require("child_process");
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
const prebuildSubdir = `${name}-${runtime}-v${abi}-${platform}-${arch}`;
const buildDir = path.resolve(
  __dirname,
  "prebuilds",
  prebuildSubdir,
  "build",
  "Release"
);
const tarballName = `ailoy-node-v${version}-node-v${abi}-${platform}-${arch}.tar.gz`;
const tarballPath = path.resolve(__dirname, "prebuilds", tarballName);

function runCommand(command, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      shell: opts.shell || true,
      cwd: opts.cwd || process.cwd(),
      env: process.env,
    });

    child.stdout.on("data", (data) => process.stdout.write(data));
    child.stderr.on("data", (data) => process.stderr.write(data));

    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${command} exited with code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

(async () => {
  try {
    fs.mkdirSync(buildDir, { recursive: true });

    const allFiles = fs
      .readdirSync(srcDir)
      .filter((f) => f === binaryName || libPattern.test(f));

    // copy files
    for (const file of allFiles) {
      const from = path.join(srcDir, file);
      const to = path.join(buildDir, file);
      if (!fs.existsSync(from)) {
        console.warn(`File not exists ${file}`);
        continue;
      }
      fs.copyFileSync(from, to);
      console.log(`✔ Copied ${file} → prebuild`);
    }

    const nodeBinary = path.join(buildDir, binaryName);
    const libSet = new Set(allFiles.filter((f) => f !== binaryName));

    // Patch library file
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
          await runCommand("install_name_tool", [
            "-change",
            depPath,
            `@loader_path/${base}`,
            nodeBinary,
          ]);
        }
      }

      await runCommand("install_name_tool", [
        "-add_rpath",
        "@loader_path",
        nodeBinary,
      ]);
    } else if (platform === "linux") {
      const readelfOut = spawnSync("readelf", ["-d", nodeBinary], {
        encoding: "utf8",
        shell: true,
      });
      if (readelfOut.error) throw readelfOut.error;
      const lines = readelfOut.stdout.split("\n");
      const entries = lines
        .filter((l) => l.includes("(NEEDED)"))
        .map((l) => l.match(/\[(.+?)\]/)?.[1])
        .filter(Boolean);
      console.log(entries);

      const patchedLibs = [];

      for (const lib of entries) {
        const localPath = path.join(buildDir, lib);
        if (fs.existsSync(localPath)) {
          console.log(`→ patch: replace ${lib} → \$ORIGIN/${lib}`);
          await runCommand(
            "patchelf",
            ["--replace-needed", lib, `$ORIGIN/${lib}`, nodeBinary],
            { shell: false }
          );
          patchedLibs.push(lib);
        }
      }
    }

    const distNode = path.resolve(__dirname, "dist", binaryName);
    if (fs.existsSync(distNode)) {
      fs.unlinkSync(distNode);
      console.log(`🧹 Removed existing dist/${binaryName}`);
    }

    console.log(`📦 Creating tarball: ${tarballName}`);
    await runCommand("tar", [
      "-czf",
      tarballPath,
      "-C",
      path.join(buildDir, "..", ".."),
      "build",
    ]);

    const prebuildDir = path.resolve(__dirname, "prebuilds", prebuildSubdir);
    fs.rmSync(prebuildDir, { recursive: true, force: true });
    console.log(`🧹 Removed ${prebuildSubdir}/`);

    console.log(`✅ Prebuilt tarball ready at: prebuilds/${tarballName}`);
  } catch (err) {
    console.error("❌ Error during prebuild process:", err.message);
    process.exit(1);
  }
})();
