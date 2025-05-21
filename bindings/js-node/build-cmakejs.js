const { spawnSync } = require("child_process");
const path = require("path");

const buildDir = "build";
const installDir = path.resolve(__dirname, "src"); // ⬅️ 절대경로로 변경

const buildArgs = [
  "cmake-js",
  "-d",
  "../..",
  "-O",
  buildDir,
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

// 2. Install (runtime deps 포함)
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
