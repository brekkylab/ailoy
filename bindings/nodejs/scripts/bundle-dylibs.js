#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { execSync } = require("node:child_process");
const crypto = require("node:crypto");
const os = require("os");

const IGNORED_LIBS_LINUX = new Set([
  // System libs
  "libc.so.6",
  "libm.so.6",
  "libpthread.so.0",
  "libdl.so.2",
  "librt.so.1",
  "libutil.so.1",
  "ld-linux-x86-64.so.2",
  "ld-linux.so.2",
  "libcrypt.so.1",
  "libnsl.so.1",
  "libresolv.so.2",
  "libnss_files.so.2",
  "libnss_dns.so.2",
  "libnss_compat.so.2",
  "libnss_nis.so.2",
  "libnss_nisplus.so.2",
  "libnss_hesiod.so.2",
  "libnss_myhostname.so.2",
  "libnss_mymachines.so.2",
  "libnss_resolve.so.2",
  "libnss_systemd.so.2",
  "libnss_sss.so.2",
  "libnss_winbind.so.2",
  "libnss_mdns4_minimal.so.2",
  "libnss_mdns4.so.2",
  "libnss_mdns_minimal.so.2",
  "libnss_mdns.so.2",
  "libgcc_s.so.1",
  "libz.so.1",
  "libbz2.so.1",
  "liblzma.so.5",
  "libtinfo.so.5",
  "libncurses.so.5",
  "libstdc++.so.6",
  "libmvec.so.1",
  "libselinux.so.1",
  "libpcre.so.3",
  "libaudit.so.1",

  // Exclude vulkan
  "libvulkan.so.1",
]);

const IGNORED_LIBS_WINDOWS = new Set([
  // Windows system DLLs (case-insensitive comparison will be used)
  "kernel32.dll",
  "user32.dll",
  "advapi32.dll",
  "ws2_32.dll",
  "msvcrt.dll",
  "shell32.dll",
  "ole32.dll",
  "oleaut32.dll",
  "gdi32.dll",
  "comdlg32.dll",
  "comctl32.dll",
  "ntdll.dll",
  "rpcrt4.dll",
  "secur32.dll",
  "crypt32.dll",
  "winmm.dll",
  "version.dll",
  "bcrypt.dll",
  "userenv.dll",
  "netapi32.dll",
  "winspool.drv",
  "psapi.dll",
  "iphlpapi.dll",
  "setupapi.dll",
  "cfgmgr32.dll",
  "shlwapi.dll",
  "imm32.dll",
  "msimg32.dll",
  "dwmapi.dll",
  "uxtheme.dll",
  "winhttp.dll",
  "wininet.dll",
  "dbghelp.dll",
  "imagehlp.dll",
  "powrprof.dll",
  "mpr.dll",
  "credui.dll",
  "wtsapi32.dll",
  "wldap32.dll",
  "vcruntime140.dll",
  "vcruntime140_1.dll",
  "msvcp140.dll",
  "ucrtbase.dll",
  "api-ms-win-crt-runtime-l1-1-0.dll",
  "api-ms-win-crt-stdio-l1-1-0.dll",
  "api-ms-win-crt-heap-l1-1-0.dll",
  "api-ms-win-crt-locale-l1-1-0.dll",
  "api-ms-win-crt-math-l1-1-0.dll",
  "api-ms-win-crt-string-l1-1-0.dll",
  "api-ms-win-crt-time-l1-1-0.dll",
  "api-ms-win-crt-filesystem-l1-1-0.dll",
  "api-ms-win-crt-environment-l1-1-0.dll",
  "api-ms-win-crt-convert-l1-1-0.dll",
  "api-ms-win-crt-process-l1-1-0.dll",
  "api-ms-win-crt-utility-l1-1-0.dll",
  "api-ms-win-crt-multibyte-l1-1-0.dll",
  "api-ms-win-core-synch-l1-2-0.dll",
  "api-ms-win-core-processthreads-l1-1-1.dll",
  "api-ms-win-core-file-l1-2-0.dll",
  "api-ms-win-core-localization-l1-2-0.dll",
  "api-ms-win-core-sysinfo-l1-2-0.dll",
  "api-ms-win-core-handle-l1-1-0.dll",

  // Exclude vulkan
  "vulkan-1.dll",
]);

class DylibsBundler {
  constructor(binaryPath, outputDir = null) {
    this.binaryPath = path.resolve(binaryPath);
    this.outputDir = outputDir || path.dirname(this.binaryPath);
    this.platform = os.platform();
    this.isLinux = this.platform === "linux";
    this.isMacOS = this.platform === "darwin";
    this.isWindows = this.platform === "win32";
    this.libDir = this.isWindows ? this.outputDir : path.join(this.outputDir, ".libs");
    this.processedLibs = new Set();
    this.targetLibs = [];
  }

  // Get dependencies using ldd (Linux) or otool (macOS)
  getDependencies(binaryPath) {
    try {
      if (this.isLinux) {
        return this.getDependenciesLinux(binaryPath);
      } else if (this.isMacOS) {
        return this.getDependenciesMacOS(binaryPath);
      } else if (this.isWindows) {
        return this.getDependenciesWindows(binaryPath);
      } 
      else {
        console.error(`Unsupported platform: ${this.platform}`);
        return [];
      }
    } catch (error) {
      console.error(
        `Failed to get dependencies for ${binaryPath}: ${error.message}`
      );
      return [];
    }
  }

  getDependenciesLinux(binaryPath) {
    const output = execSync(`ldd "${binaryPath}"`, { encoding: "utf-8" });
    const deps = [];

    // Get exact names from DT_NEEDED using patchelf
    let needed = [];
    try {
      needed = execSync(`patchelf --print-needed "${binaryPath}"`, {
        encoding: "utf-8",
      })
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
    } catch (e) {
      // Fallback or ignore if patchelf fails
    }

    for (const line of output.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let libNameStr, libPath;

      // Case 1: libfoo.so => /path/to/libfoo.so (0x...)
      // Case 2: libfoo.so => not found
      const matchRel = trimmed.match(/^(.+?)\s*=>\s*(.+?)\s*(\(|$)/);
      if (matchRel) {
        libNameStr = matchRel[1].trim();
        libPath = matchRel[2].trim();
      } else {
        // Case 3: /path/to/libfoo.so (0x...)
        const matchAbs = trimmed.match(/^\s*(\/.+?)\s+\(/);
        if (matchAbs) {
          libPath = matchAbs[1].trim();
          libNameStr = libPath;
        }
      }

      if (libNameStr && libPath) {
        const libName = path.basename(libNameStr);
        if (libPath === "not found") {
          // Keep it as "not found"
        }

        // Find the original name in the DT_NEEDED list that corresponds to this lib
        const originalPath =
          needed.find((n) => n === libNameStr || n === libName || path.basename(n) === libName) ||
          libNameStr;

        deps.push({ name: libName, path: libPath, originalPath });
      }
    }

    return deps;
  }

  getDependenciesMacOS(binaryPath) {
    const output = execSync(`otool -L "${binaryPath}"`, { encoding: "utf-8" });
    const deps = [];
    const lines = output.split("\n").slice(1); // Skip first line (binary name)

    const rpaths = this.getRpaths(binaryPath);
    const loaderPath = path.dirname(binaryPath);

    for (const line of lines) {
      const match = line.trim().match(/^(.+?)\s+\(/);
      if (match) {
        let libPath = match[1].trim();
        const libName = path.basename(libPath);
        const originalPath = libPath;

        // Resolve @rpath, @loader_path, @executable_path
        if (libPath.startsWith("@rpath/")) {
          const suffix = libPath.substring(7);
          for (const rpathVal of rpaths) {
            let resolvedPath = rpathVal;
            if (resolvedPath.startsWith("@loader_path")) {
              resolvedPath = resolvedPath.replace("@loader_path", loaderPath);
            }
            const testPath = path.join(resolvedPath, suffix);
            if (fs.existsSync(testPath)) {
              libPath = testPath;
              break;
            }
          }
        } else if (libPath.startsWith("@loader_path/")) {
          libPath = libPath.replace("@loader_path", loaderPath);
        }

        deps.push({ name: libName, path: libPath, originalPath });
      }
    }

    return deps;
  }

  getRpaths(binaryPath) {
    try {
      const output = execSync(`otool -l "${binaryPath}"`, { encoding: "utf-8" });
      const rpaths = [];
      const lines = output.split("\n");
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === "cmd LC_RPATH") {
          const pathLine = lines[i + 2].trim();
          const match = pathLine.match(/path (.+?) \(offset \d+\)/);
          if (match) {
            rpaths.push(match[1]);
          }
        }
      }
      return rpaths;
    } catch (error) {
      return [];
    }
  }

  getDependenciesWindows(binaryPath) {
    const deps = [];

    try {
      const output = execSync(`dumpbin /DEPENDENTS "${binaryPath}"`, {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"]
      });
      const lines = output.split(/\r?\n/);
      let inDepsSection = false;

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.includes("has the following dependencies:")) {
          inDepsSection = true;
          continue;
        }

        if (inDepsSection) {
          // Continue on empty line
          if (!trimmed) continue;

          // "Summary" marks end of dependencies section
          if (trimmed.includes("Summary")) {
            break;
          }

          // DLL names are listed one per line, indented
          if (trimmed && trimmed.endsWith(".dll")) {
            const dllName = trimmed;
            const dllPath = this.findDllPath(dllName);

            if (dllPath) {
              deps.push({name: dllName, path: dllPath});
            } else {
              console.warn(`Warning: Could not find DLL: ${dllName}`);
            }
          }
        }
      }
    } catch (error) {
      console.error(`Failed to run dumpbin on ${binaryPath}: ${error.message}`);
      console.error("Make sure dumpbin is available (Visual Studio Developer Command Prompt)");
    }

    return deps;
  }
  
  // Try to find a DLL in common locations
  findDllPath(dllName) {
    const searchPaths = [
      path.dirname(this.binaryPath),
      process.cwd(),
      ...process.env.PATH.split(path.delimiter),
    ];

    console.log("[findDllPath] dllName: ", dllName);
    for (const searchPath of searchPaths) {
      const fullPath = path.join(searchPath, dllName);
      console.log("[findDllPath] fullPath: ", fullPath);
      if (fs.existsSync(fullPath)) {
        return fullPath;
      }
    }

    return null;
  }

  shouldBundle(libName, libPath) {
    if (this.isLinux) {
      // Ignore system libraries
      if (IGNORED_LIBS_LINUX.has(libName)) {
        return false;
      }
    } else if (this.isMacOS) {
      // Ignore system libraries and frameworks
      if (libPath.startsWith("/usr/lib/") || libPath.startsWith("/System/")) {
        return false;
      }

      // Ignore libailoy.dylib if it's the one we are currently processing or it's a known artifact
      if (libName === "libailoy.dylib") {
        return false;
      }
    } else if (this.isWindows) {
      const lowerName = libName.toLowerCase();
      if (IGNORED_LIBS_WINDOWS.has(lowerName)) {
        return false;
      }

      const lowerPath = libPath.toLowerCase();
      if (lowerPath.includes("\\windows\\") || lowerPath.includes("\\winsxs\\") || lowerPath.includes("\\system32\\") || lowerPath.includes("\\syswow64\\")) {
        return false;
      }
    }

    // Ignore if already processed
    if (this.processedLibs.has(libName)) {
      return false;
    }

    return true;
  }

  copyLibrary(oldPath, name) {
    try {
      const oldName = path.basename(name);
      const data = fs.readFileSync(oldPath);
      const hash = crypto
        .createHash("sha256")
        .update(data)
        .digest("hex")
        .substring(0, 8);

      let newName;
      if (this.isLinux) {
        // Create versioned name: libfoo.so.1 -> libfoo-a1b2c3d4.so.1
        const parts = oldName.split(".so");
        newName =
          parts.length > 1
            ? `${parts[0]}-${hash}.so${parts.slice(1).join(".so")}`
            : `${oldName}-${hash}`;
      } else if (this.isMacOS) {
        // Create versioned name: libfoo.1.dylib -> libfoo-a1b2c3d4.1.dylib
        const parts = oldName.split(".dylib");
        newName =
          parts.length > 1 && parts[0]
            ? `${parts[0]}-${hash}.dylib${parts.slice(1).join(".dylib")}`
            : `${oldName.replace(".dylib", "")}-${hash}.dylib`;
      } else if (this.isWindows) {
        // Windows: Don't rename, just use original name
        // DLLs are loaded by exact name and can't be patched
        newName = oldName;
      }

      const newPath = path.join(this.libDir, newName);

      if (!fs.existsSync(newPath)) {
        fs.copyFileSync(oldPath, newPath);
        fs.chmodSync(newPath, 0o755);
        console.log(`Copied: ${oldName} -> ${newName}`);
      }

      return [newName, newPath];
    } catch (error) {
      console.error(`Failed to copy library ${oldName}: ${error}`);
      return null;
    }
  }

  collectDependencies(binaryPath, depth = 0) {
    const deps = this.getDependencies(binaryPath);
    const toBundle = [];

    for (const { name, path: libPath, originalPath } of deps) {
      if (!this.shouldBundle(name, libPath)) {
        continue;
      }

      if (libPath === "not found") {
        console.error(`Dependency not found: ${name}`);
        continue;
      }

      // Resolve actual path for macOS if needed
      let actualPath = libPath;
      if (this.isMacOS && !fs.existsSync(libPath)) {
        // Try to find the library
        console.error(`Library path not found: ${libPath}`);
        continue;
      }

      this.processedLibs.add(name);
      toBundle.push({ name, path: actualPath, originalPath });

      // Recursively process this library's dependencies
      const subDeps = this.collectDependencies(actualPath, depth + 1);
      toBundle.push(...subDeps);
    }

    return toBundle;
  }

  // Patch binary RPATH using patchelf (Linux) or install_name_tool (macOS)
  patchRpath(binaryPath, rpath) {
    try {
      if (this.isLinux) {
        return this.patchRpathLinux(binaryPath, rpath);
      } else if (this.isMacOS) {
        return this.patchRpathMacOS(binaryPath, rpath);
      }
      return false;
    } catch (error) {
      console.error(`Failed to patch RPATH: ${error.message}`);
      return false;
    }
  }

  patchRpathLinux(binaryPath, rpath) {
    // Remove existing rpath
    execSync(`patchelf --remove-rpath "${binaryPath}"`, { stdio: "pipe" });

    // Set new rpath
    execSync(`patchelf --set-rpath '${rpath}' "${binaryPath}"`, {
      stdio: "pipe",
    });

    console.log(`Patched RPATH for ${path.basename(binaryPath)}: ${rpath}`);
    return true;
  }

  patchRpathMacOS(binaryPath, rpath) {
    // Remove all existing RPATHs
    const existingRpaths = this.getRpaths(binaryPath);
    for (const oldRpath of existingRpaths) {
      try {
        execSync(`install_name_tool -delete_rpath "${oldRpath}" "${binaryPath}"`, {
          stdio: "pipe",
        });
      } catch (error) {
        // Ignore errors if it fails to delete
      }
    }

    // Add new rpath
    try {
      execSync(`install_name_tool -add_rpath '${rpath}' "${binaryPath}"`, {
        stdio: "pipe",
      });
      console.log(`Added RPATH for ${path.basename(binaryPath)}: ${rpath}`);
    } catch (error) {
      // RPATH might already exist, which is fine
      if (!error.message.includes("would duplicate path")) {
        throw error;
      }
    }
    return true;
  }

  // Patch library dependencies
  patchLibraryDependencies(libPath, targetLibs) {
    try {
      if (this.isLinux) {
        return this.patchLibraryDependenciesLinux(libPath, targetLibs);
      } else if (this.isMacOS) {
        return this.patchLibraryDependenciesMacOS(libPath, targetLibs);
      }
      return false;
    } catch (error) {
      console.error(
        `Failed to patch dependencies for ${path.basename(libPath)}: ${error.message}`
      );
      return false;
    }
  }

  patchLibraryDependenciesLinux(libPath, targetLibs) {
    const deps = this.getDependenciesLinux(libPath);
    for (const dep of deps) {
      const target = targetLibs.find((t) => t.oldName === dep.name);
      if (target) {
        try {
          execSync(
            `patchelf --replace-needed "${dep.originalPath}" "${target.newName}" "${libPath}"`,
            { stdio: "pipe" }
          );
        } catch (e) {
          // Library might not depend on this one, continue
        }
      }
    }
    return true;
  }

  patchLibraryDependenciesMacOS(libPath, targetLibs) {
    const deps = this.getDependenciesMacOS(libPath);
    for (const dep of deps) {
      const target = targetLibs.find((t) => t.oldName === dep.name);
      if (target) {
        try {
          execSync(
            `install_name_tool -change "${dep.originalPath}" '@rpath/${target.newName}' "${libPath}"`,
            { stdio: "pipe" }
          );
        } catch (e) {
          // Continue on error
        }
      }
    }
    return true;
  }

  // Change install name for macOS dylib
  changeInstallName(libPath, newName) {
    if (!this.isMacOS) return;

    try {
      execSync(`install_name_tool -id "@loader_path/${newName}" "${libPath}"`, {
        stdio: "pipe",
      });
    } catch (error) {
      console.error(
        `Failed to change install name for ${path.basename(libPath)}`
      );
    }
  }

  // Re-sign binary/library for macOS (ad-hoc signing)
  reSignMacOS(path) {
    if (!this.isMacOS) return;
    try {
      execSync(`codesign -f -s - "${path}"`, { stdio: "pipe" });
      console.log(`Re-signed: ${path}`);
    } catch (error) {
      console.error(`Failed to re-sign ${path}: ${error.message}`);
    }
  }

  // Check if required tools are available
  checkTools() {
    try {
      if (this.isLinux) {
        execSync("patchelf --version", { stdio: "pipe" });
        return true;
      } else if (this.isMacOS) {
        execSync("which install_name_tool", { stdio: "pipe" });
        execSync("which otool", { stdio: "pipe" });
        return true;
      } else if (this.isWindows) {
        try {
          execSync("Get-Command dumpbin", { shell: "powershell", stdio:"pipe"});
        } catch(e) {
          console.error("dumpbin is not available.");
          return false;
        }
        return true;
      }
    } catch (error) {
      if (this.isLinux) {
        console.error("patchelf is not installed. Please install it first.");
      } else if (this.isMacOS) {
        console.error(
          "install_name_tool or otool is not available. Please install Xcode Command Line Tools."
        );
      }
      return false;
    }

    console.error(`Unsupported platform: ${this.platform}`);
    return false;
  }

  bundle() {
    console.log(`Bundling dependencies for: ${this.binaryPath}`);

    // Check if binary exists
    if (!fs.existsSync(this.binaryPath)) {
      console.error(`Binary not found: ${this.binaryPath}`);
      return false;
    }

    // Check if required tools are available
    if (!this.checkTools()) {
      return false;
    }

    // Create libs directory
    if (!fs.existsSync(this.libDir)) {
      fs.mkdirSync(this.libDir, { recursive: true });
      console.log(`Created library directory: ${this.libDir}`);
    }

    // Collect all dependencies
    console.log("Collecting dependencies...");
    const dependencies = this.collectDependencies(this.binaryPath);

    if (dependencies.length === 0) {
      console.log("No dependencies to bundle.");
      return true;
    }

    console.log(`Found ${dependencies.length} dependencies to bundle`);

    // Copy all dependencies and build mapping
    for (const { name: oldName, path: oldPath } of dependencies) {
      const [newName, newPath] = this.copyLibrary(oldPath, oldName);
      this.targetLibs.push({
        oldName,
        oldPath,
        newName,
        newPath,
      });
    }

    if (!this.isWindows) {
      // Patch main binary RPATH
      const rpath = this.isLinux ? "$ORIGIN/.libs" : "@loader_path/.libs";
      if (!this.patchRpath(this.binaryPath, rpath)) {
        return false;
      }

      // Patch main binary to use renamed libraries
      console.log("Patching main binary dependencies...");
      this.patchLibraryDependencies(this.binaryPath, this.targetLibs);

      // Recursive patching for all bundled libraries
      console.log("Patching bundled libraries...");
      const libRpath = this.isLinux ? "$ORIGIN" : "@loader_path";
      for (const lib of this.targetLibs) {
        // Change its own ID (for macOS)
        this.changeInstallName(lib.newPath, lib.newName);
        // Patch its dependencies
        this.patchLibraryDependencies(lib.newPath, this.targetLibs);
        // Clear and set RPATH for the library itself as well
        this.patchRpath(lib.newPath, libRpath);
        // Re-sign the library
        this.reSignMacOS(lib.newPath);
      }

      // Finally re-sign the main binary
      this.reSignMacOS(this.binaryPath);
    }

    console.log("✓ Bundling complete!");
    console.log(`  Binary: ${this.binaryPath}`);
    console.log(`  Libraries: ${this.libDir}`);
    console.log(`  Bundled ${this.targetLibs.length} libraries`);

    return true;
  }
}

function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`
Usage: node bundle-dylibs.js <binary-path> [output-dir]

Bundle dynamic libraries with a napi-rs binary, similar to Python's auditwheel.

Arguments:
  binary-path    Path to the .node binary file
  output-dir     Optional output directory (defaults to binary's directory)

Example:
  node bundle-dylibs.js ./index.node
  node bundle-dylibs.js ./index.node ./dist

Requirements:
  - patchelf must be installed
`);
    process.exit(0);
  }

  const binaryPath = args[0];
  const outputDir = args[1];

  const bundler = new DylibsBundler(binaryPath, outputDir);
  const success = bundler.bundle();

  process.exit(success ? 0 : 1);
}

if (require.main === module) {
  main();
}

module.exports = { DylibsBundler };
