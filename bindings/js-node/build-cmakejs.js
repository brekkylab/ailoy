const { spawnSync } = require("child_process");
const path = require("path");

const buildArgs = [
  "prebuild",
  "--napi",
  "--backend",
  "cmake-js",
  "--",
  "-d",
  "../..",
  "-O",
  "build",
  "--CDNODE:BOOL=ON",
  "--CDAILOY_WITH_TEST:BOOL=OFF",
  `--parallel ${require("os").cpus().length}`,
];

const buildResult = spawnSync("npx", buildArgs, {
  stdio: "inherit",
  cwd: path.resolve(__dirname),
});

if (buildResult.error || buildResult.status !== 0) {
  console.error("build failed.");
  process.exit(buildResult.status || 1);
}
