#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { execSync } = require("node:child_process");
const crypto = require("node:crypto");
const os = require("os");

const SYSTEM_LIBS_LINUX = new Set([
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
]);

class DylibsBundler {
  constructor(binaryPath, outputDir = null) {
    this.binaryPath = path.resolve(binaryPath);
    this.outputDir = outputDir || path.dirname(this.binaryPath);
    this.libDir = path.join(this.outputDir, ".libs");
    this.processedLibs = new Set();
    this.libMapping = new Map();
    this.platform = os.platform();
    this.isLinux = this.platform === "linux";
    this.isMacOS = this.platform === "darwin";
  }

  // Get dependencies using ldd (Linux) or otool (macOS)
  getDependencies(binaryPath) {
    try {
      if (this.isLinux) {
        return this.getDependenciesLinux(binaryPath);
      } else if (this.isMacOS) {
        return this.getDependenciesMacOS(binaryPath);
      } else {
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

    for (const line of output.split("\n")) {
      const match = line.match(/\s*(.+?)\s*=>\s*(.+?)\s*\(/);
      if (match) {
        const [, libName, libPath] = match;
        deps.push({ name: libName.trim(), path: libPath.trim() });
      }
    }

    return deps;
  }

  getDependenciesMacOS(binaryPath) {
    const output = execSync(`otool -L "${binaryPath}"`, { encoding: "utf-8" });
    const deps = [];
    const lines = output.split("\n").slice(1); // Skip first line (binary name)

    for (const line of lines) {
      const match = line.trim().match(/^(.+?)\s+\(/);
      if (match) {
        const libPath = match[1].trim();
        const libName = path.basename(libPath);
        deps.push({ name: libName, path: libPath });
      }
    }

    return deps;
  }

  shouldBundle(libName, libPath) {
    if (this.isLinux) {
      // Ignore system libraries
      if (SYSTEM_LIBS_LINUX.has(libName)) {
        return false;
      }
    } else if (this.isMacOS) {
      // Ignore system libraries and frameworks
      if (libPath.startsWith("/usr/lib/") || libPath.startsWith("/System/")) {
        return false;
      }

      // Ignore @rpath, @loader_path, @executable_path if they don't exist
      if (libPath.startsWith("@")) {
        return false;
      }

      // Ignore libailoy.dylib
      if (libName === "libailoy.dylib") {
        return false;
      }
    }

    // Ignore if already processed
    if (this.processedLibs.has(libName)) {
      return false;
    }

    return true;
  }

  copyLibrary(sourcePath, libName) {
    try {
      const data = fs.readFileSync(sourcePath);
      const hash = crypto
        .createHash("sha256")
        .update(data)
        .digest("hex")
        .substring(0, 8);

      let newName;
      if (this.isLinux) {
        // Create versioned name: libfoo.so.1 -> libfoo-a1b2c3d4.so.1
        const parts = libName.split(".so");
        newName =
          parts.length > 1
            ? `${parts[0]}-${hash}.so${parts.slice(1).join(".so")}`
            : `${libName}-${hash}`;
      } else if (this.isMacOS) {
        // Create versioned name: libfoo.1.dylib -> libfoo-a1b2c3d4.1.dylib
        const parts = libName.split(".dylib");
        newName =
          parts.length > 1 && parts[0]
            ? `${parts[0]}-${hash}.dylib${parts.slice(1).join(".dylib")}`
            : `${libName.replace(".dylib", "")}-${hash}.dylib`;
      }

      const destPath = path.join(this.libDir, newName);

      if (!fs.existsSync(destPath)) {
        fs.copyFileSync(sourcePath, destPath);
        fs.chmodSync(destPath, 0o755);
        console.log(`Copied: ${libName} -> ${newName}`);
      }

      return newName;
    } catch (error) {
      console.error(`Failed to copy library ${libName}: ${error}`);
      return null;
    }
  }

  collectDependencies(binaryPath, depth = 0) {
    const deps = this.getDependencies(binaryPath);
    const toBundle = [];

    for (const { name, path: libPath } of deps) {
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
      toBundle.push({ name, path: actualPath });

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
    // Add rpath if not exists
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
  patchLibraryDependencies(libPath, libMapping) {
    try {
      if (this.isLinux) {
        return this.patchLibraryDependenciesLinux(libPath, libMapping);
      } else if (this.isMacOS) {
        return this.patchLibraryDependenciesMacOS(libPath, libMapping);
      }
      return false;
    } catch (error) {
      console.error(
        `Failed to patch dependencies for ${path.basename(libPath)}: ${error.message}`
      );
      return false;
    }
  }

  patchLibraryDependenciesLinux(libPath, libMapping) {
    for (const [oldName, newName] of libMapping.entries()) {
      try {
        execSync(
          `patchelf --replace-needed "${oldName}" "${newName}" "${libPath}"`,
          { stdio: "pipe" }
        );
      } catch (e) {
        // Library might not depend on this one, continue
      }
    }
    return true;
  }

  patchLibraryDependenciesMacOS(libPath, libMapping) {
    const deps = this.getDependenciesMacOS(libPath);

    for (const { name, path: depPath } of deps) {
      if (libMapping.has(name)) {
        const newName = libMapping.get(name);
        try {
          execSync(
            `install_name_tool -change "${depPath}" '@loader_path/${newName}' "${libPath}"`,
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
    console.log(`Bundling dependenceis for: ${this.binaryPath}`);

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
    for (const { name, path: libPath } of dependencies) {
      const newName = this.copyLibrary(libPath, name);
      if (newName) {
        this.libMapping.set(name, newName);
        // Also map by full path for macOS
        if (this.isMacOS) {
          this.libMapping.set(libPath, newName);
        }
      }
    }

    // Patch main binary RPATH
    const rpath = this.isLinux ? "$ORIGIN/.libs" : "@loader_path/.libs";
    if (!this.patchRpath(this.binaryPath, rpath)) {
      return false;
    }

    // Patch main binary to use renamed libraries
    console.log("Patching main binary dependencies...");
    this.patchLibraryDependencies(this.binaryPath, this.libMapping);

    // Patch all bundled libraries
    console.log("Patching bundled library dependencies...");
    for (const [oldName, newName] of this.libMapping.entries()) {
      // Skip path-based keys on macOS
      if (this.isMacOS && oldName.includes("/")) {
        continue;
      }

      const libPath = path.join(this.libDir, newName);

      if (this.isLinux) {
        this.patchRpath(libPath, "$ORIGIN");
      } else if (this.isMacOS) {
        this.changeInstallName(libPath, newName);
        this.patchRpath(libPath, "@loader_path");
      }

      this.patchLibraryDependencies(libPath, this.libMapping);
    }

    console.log("✓ Bundling complete!");
    console.log(`  Binary: ${this.binaryPath}`);
    console.log(`  Libraries: ${this.libDir}`);
    console.log(`  Bundled ${this.libMapping.size} libraries`);

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
