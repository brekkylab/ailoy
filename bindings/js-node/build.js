const { spawn } = require("child_process");
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

(async () => {
  try {
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

    const installArgs = ["--install", buildDir, "--prefix", installDir];
    await runCommand("cmake", installArgs, { cwd: __dirname });
  } catch (err) {
    console.error("Build failed:", err.message);
    process.exit(1);
  }
})();
