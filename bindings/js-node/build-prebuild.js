const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");

const platform = os.platform();
const buildDir = path.resolve(__dirname, "build");
const buildBinaryPath = path.resolve(buildDir, "Release", "ailoy_addon.node");
const prebuildsDir = path.resolve(__dirname, "prebuilds");

function failIfError(res, msg) {
  if (res.error || res.status !== 0) {
    console.error(msg);
    process.exit(res.status || 1);
  }
}

function isSystemLibrary(libPath) {
  const systemPaths = ["/lib", "/lib64", "/usr/lib", "/usr/lib64"];
  const systemLibs = [
    "libc.so",
    "libm.so",
    "libpthread.so",
    "libdl.so",
    "librt.so",
    "libgcc_s.so",
    "libstdc++.so",
    "ld-linux",
    "libselinux.so",
    "libcrypt.so",
    "libz.so",
  ];
  return (
    systemPaths.some((p) => libPath.startsWith(p)) &&
    systemLibs.some((name) => libPath.includes(name))
  );
}

function getDynamicLibs(binaryPath) {
  if (platform === "darwin") {
    const res = spawnSync("otool", ["-L", binaryPath], { encoding: "utf-8" });
    failIfError(res, "otool failed");
    return res.stdout
      .split("\n")
      .slice(1)
      .map((line) => line.trim().split(" ")[0])
      .filter(
        (lib) =>
          lib &&
          !lib.startsWith("/System/") &&
          !lib.startsWith("/usr/lib") &&
          (fs.existsSync(lib) || lib.startsWith("@rpath"))
      );
  }

  if (platform === "linux") {
    const res = spawnSync("ldd", [binaryPath], { encoding: "utf-8" });
    failIfError(res, "ldd failed");
    return res.stdout
      .split("\n")
      .map((line) => line.trim().split("=>")[1]?.trim().split(" ")[0])
      .filter((lib) => lib && fs.existsSync(lib) && !isSystemLibrary(lib));
  }

  return [];
}

function getWindowsDLLs(binaryPath) {
  const res = spawnSync("dumpbin", ["/DEPENDENTS", binaryPath], {
    encoding: "utf-8",
  });
  if (res.status !== 0) {
    console.warn("Warning: dumpbin failed. Skipping DLL detection.");
    return [];
  }

  return res.stdout
    .split("\n")
    .map((line) => line.trim())
    .filter(
      (line) =>
        line.endsWith(".dll") &&
        !line.startsWith("KERNEL") &&
        !line.startsWith("API-MS")
    )
    .map((dll) => path.resolve(__dirname, "build/Release", dll))
    .filter(fs.existsSync);
}

// Step 1: Build with prebuild
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
  `--parallel=${os.cpus().length}`,
];
const res = spawnSync("npx", buildArgs, {
  cwd: path.resolve(__dirname),
  encoding: "utf-8",
  stdio: "pipe",
  shell: true,
  env: {
    ...process.env,
  },
});

if (res.stdout) console.log(res.stdout);
if (res.stderr) console.error(res.stderr);
failIfError(res, "Build failed");

// Step 2: Validate build output
if (!fs.existsSync(buildBinaryPath)) {
  console.error(`Built binary not found: ${buildBinaryPath}`);
  process.exit(1);
}

// Step 3: Get runtime dynamic library dependencies
const depLibs =
  platform === "win32"
    ? getWindowsDLLs(buildBinaryPath)
    : getDynamicLibs(buildBinaryPath);
if (depLibs.length === 0) {
  console.log("No dynamic libraries found. Skipping patch.");
  process.exit(0);
}

// Step 4: Locate .tar.gz output from prebuild
if (!fs.existsSync(prebuildsDir)) {
  console.error("prebuilds/ directory not found.");
  process.exit(1);
}
const tarFiles = fs
  .readdirSync(prebuildsDir)
  .filter((f) => f.endsWith(".tar.gz"));
if (tarFiles.length === 0) {
  console.error("No .tar.gz file found in prebuilds/");
  process.exit(1);
}
const tarFilePath = path.join(prebuildsDir, tarFiles[0]);

// Step 5: Unpack tar.gz into temporary directory
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "prebuild-unpack-"));
console.log(`Unpacking ${tarFilePath}...`);
failIfError(
  spawnSync("tar", ["-xzf", tarFilePath, "-C", tmpDir]),
  "Unpack failed"
);
const auditDir = path.join(tmpDir, "build", "Release");

// Step 7: Copy dependent libraries and patch binary for platform-specific behavior
const auditBinaryPath = path.join(auditDir, "ailoy_addon.node");
if (platform === "darwin") {
  for (const libPath of depLibs) {
    const libName = path.basename(libPath);
    const targetPath = path.join(auditDir, libName);

    // Case 1: If absolute path (exists), use directly
    let sourcePath = libPath;
    if (!fs.existsSync(sourcePath)) {
      // Case 2: If it's @rpath or unresolved, try build/Release
      const candidate = path.join(path.dirname(buildBinaryPath), libName);
      if (fs.existsSync(candidate)) {
        sourcePath = candidate;
      } else {
        console.warn(`Warning: Could not resolve ${libName}. Skipping copy.`);
        continue;
      }
    }

    // Copy to unpacked audit dir
    fs.copyFileSync(sourcePath, targetPath);
    console.log(`Copied ${libName} → ${targetPath}`);

    // Set install_name of copied dylib
    failIfError(
      spawnSync("install_name_tool", ["-id", `@rpath/${libName}`, targetPath], {
        stdio: "inherit",
      }),
      `install_name_tool -id failed for ${libName}`
    );

    // Patch .node if original was absolute
    if (!libPath.startsWith("@rpath/")) {
      failIfError(
        spawnSync(
          "install_name_tool",
          ["-change", libPath, `@rpath/${libName}`, auditBinaryPath],
          { stdio: "inherit" }
        ),
        `install_name_tool -change failed for ${libName}`
      );
    }
  }

  console.log(`Adding @loader_path rpath to ${auditBinaryPath}`);
  const rpathResult = spawnSync(
    "install_name_tool",
    ["-add_rpath", "@loader_path", auditBinaryPath],
    {
      stdio: "inherit",
    }
  );
  if (
    rpathResult.status !== 0 &&
    !rpathResult.stderr?.toString().includes("would duplicate path")
  ) {
    console.error("Failed to add @loader_path");
    process.exit(1);
  }
} else if (platform === "linux") {
  for (const libPath of depLibs) {
    const libName = path.basename(libPath);
    const targetPath = path.join(auditDir, libName);

    // Case 1: absolute path exists
    let sourcePath = libPath;
    if (!fs.existsSync(sourcePath)) {
      // Case 2: try from build/Release/
      const candidate = path.join(path.dirname(buildBinaryPath), libName);
      if (fs.existsSync(candidate)) {
        sourcePath = candidate;
      } else {
        console.warn(`Warning: Could not resolve ${libName}. Skipping copy.`);
        continue;
      }
    }

    fs.copyFileSync(sourcePath, targetPath);
    console.log(`Copied ${libName} → ${targetPath}`);
  }

  // Set RPATH to $ORIGIN for all ELF binaries (.node and copied .so)
  console.log(`Setting RPATH to $ORIGIN for ELF binaries...`);
  const elfTargets = [
    auditBinaryPath,
    ...depLibs
      .map((lib) => path.basename(lib))
      .map((name) => path.join(auditDir, name))
      .filter((p) => fs.existsSync(p)),
  ];

  for (const target of elfTargets) {
    failIfError(
      spawnSync("patchelf", ["--set-rpath", "$ORIGIN", target], {
        stdio: "inherit",
      }),
      `patchelf failed for ${target}`
    );
  }
} else if (platform === "win32") {
  for (const dllPath of depLibs) {
    const dllName = path.basename(dllPath);
    const targetPath = path.join(auditDir, dllName);

    // Case 1: use absolute path if exists
    let sourcePath = dllPath;
    if (!fs.existsSync(sourcePath)) {
      // Case 2: fallback to build/Release
      const candidate = path.join(path.dirname(buildBinaryPath), dllName);
      if (fs.existsSync(candidate)) {
        sourcePath = candidate;
      } else {
        console.warn(`Warning: Could not resolve ${dllName}. Skipping copy.`);
        continue;
      }
    }

    fs.copyFileSync(sourcePath, targetPath);
    console.log(`Copied ${dllName} → ${targetPath}`);
  }
}

// Step 8: Repack tar.gz with patched contents
const backupPath = `${tarFilePath}.bak`;
fs.renameSync(tarFilePath, backupPath);
console.log(`Backup created: ${backupPath}`);
console.log(`Repacking ${tarFilePath}...`);
failIfError(
  spawnSync("tar", ["-czf", tarFilePath, "-C", tmpDir, "."], {
    stdio: "inherit",
  }),
  "Repack failed"
);

// Step 9: Clean up temporary files
fs.rmSync(tmpDir, { recursive: true, force: true });
console.log("✅ Prebuild and dependency patch complete.");
