import commonjs from "@rollup/plugin-commonjs";
import json from "@rollup/plugin-json";
import resolve from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath } from "node:url";
import copy from "rollup-plugin-copy";
import { dts } from "rollup-plugin-dts";
import tscAlias from "rollup-plugin-tsc-alias";

import pkg from "./package.json" with { type: "json" };

const tsconfig = "tsconfig.build.json";

export default [
  // main build
  {
    input: "src/index.ts",
    output: [
      {
        file: pkg.main,
        format: "cjs",
      },
      {
        file: pkg.module,
        format: "esm",
        exports: "named",
      },
    ],
    external: [fileURLToPath(new URL("src/ailoy_addon.node", import.meta.url))],
    plugins: [
      resolve(),
      commonjs(),
      json(),
      typescript({
        tsconfig,
      }),
      tscAlias({
        tsconfig,
      }),
      copy({
        targets: [
          {
            src: ["src/ailoy_addon.node", "src/postinstall.js"],
            dest: "dist",
          },
          {
            src:
              os.platform() === "win32"
                ? ["src/tvm_runtime.dll", "src/mlc_llm.dll"]
                : [],
            dest: "dist",
          },
          {
            src:
              os.platform() === "win32"
                ? [
                    "src/ailoy_addon.pdb",
                    "src/tvm_runtime.pdb",
                    "src/mlc_llm.pdb",
                  ].filter((f) => fs.existsSync(f))
                : [],
            dest: "dist",
          },
          {
            src: ["src/presets/*"],
            dest: "dist/presets",
          },
        ],
      }),
      terser(),
    ],
  },
  // index.d.ts
  {
    input: "src/index.ts",
    output: [{ file: "dist/index.d.ts", format: "esm" }],
    plugins: [dts({ tsconfig })],
  },
  // CLI
  {
    input: "src/cli/index.ts",
    output: [{ file: "dist/cli.cjs", format: "cjs" }],
    plugins: [
      typescript({ tsconfig }),
      tscAlias({ tsconfig }),
      resolve(),
      commonjs(),
      terser(),
    ],
  },
];
