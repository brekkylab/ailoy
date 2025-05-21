const { spawnSync } = require("child_process");
const path = require("path");

const srcDir = path.resolve(__dirname, "../..");
const buildDir = path.resolve(__dirname, "build");
const installDir = path.resolve(__dirname, "src");

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

const buildResult = spawnSync("npx", buildArgs, {
  stdio: "inherit",
  cwd: path.resolve(__dirname),
  shell: true,
});

if (buildResult.error || buildResult.status !== 0) {
  console.log(buildResult);
  console.error("build failed.");
  process.exit(buildResult.status || 1);
}

const installResult = spawnSync(
  "cmake",
  ["--install", buildDir, "--prefix", installDir],
  {
    stdio: "inherit",
  }
);

if (installResult.error || installResult.status !== 0) {
  console.error("install failed.");
  process.exit(installResult.status || 1);
}
