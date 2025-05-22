const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const srcDir = path.resolve(__dirname, "../..");
const buildDir = path.resolve(__dirname, "build");
const installDir = path.resolve(__dirname, "src");

function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      shell: true,
      cwd: options.cwd || process.cwd(),
      env: process.env,
    });

    child.stdout.on("data", (data) => {
      process.stdout.write(data);
    });

    child.stderr.on("data", (data) => {
      process.stderr.write(data);
    });

    child.on("error", (err) => {
      reject(err);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${command} exited with code ${code}`));
      } else {
        resolve();
      }
    });
  });
}

function listDir(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relPath = path.relative(dir, fullPath);
    if (entry.isDirectory()) {
      console.log(`[DIR]  ${relPath}/`);
    } else {
      const size = fs.statSync(fullPath).size;
      console.log(`[FILE] ${relPath} (${size} bytes)`);
    }
  }
}

(async () => {
  try {
    console.log("Building...");
    const buildArgs = [
      "cmake-js",
      "-d",
      srcDir,
      "-O",
      buildDir,
      "--CDNODE:BOOL=ON",
      "--CDAILOY_WITH_TEST:BOOL=OFF",
      `--parallel ${require("os").cpus().length}`,
    ];
    await runCommand("npx", buildArgs, { cwd: __dirname });

    console.log("Installing...");
    const installArgs = ["--install", buildDir, "--prefix", installDir];
    await runCommand("cmake", installArgs, { cwd: __dirname });

    console.log(`📂 Files under ${installDir}:\n`);
    listDir(installDir);
  } catch (err) {
    console.error("Build failed:", err.message);
    process.exit(1);
  }
})();
