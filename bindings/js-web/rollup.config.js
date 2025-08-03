import { nodeResolve } from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import typescript from "@rollup/plugin-typescript";
import ignore from "rollup-plugin-ignore";
import copy from "rollup-plugin-copy";
import serve from "rollup-plugin-serve";
import json from "@rollup/plugin-json";
import { dts } from "rollup-plugin-dts";
import terser from "@rollup/plugin-terser";

const enableDevServer = Boolean(process.env.AILOY_WEB_DEV_SERVER);

module.exports = [
  {
    input: "src/index.ts",
    output: {
      file: "dist/index.js",
      format: "es",
    },
    plugins: [
      ignore(["fs", "path", "crypto", "perf_hooks", "ws"]),
      nodeResolve({
        browser: true,
      }),
      typescript(),
      commonjs(),
      json(),
      copy({
        targets: [
          {
            src: ["src/ailoy_js_web.js", "src/ailoy_js_web.wasm"],
            dest: "dist",
          },
        ],
      }),
      terser(),
      enableDevServer
        ? copy({
            targets: [
              {
                src: ["src/index.html"],
                dest: "dist",
              },
            ],
          })
        : undefined,
      enableDevServer
        ? serve({
            contentBase: "dist",
            port: 8000,
            headers: {
              "Cross-Origin-Embedder-Policy": "require-corp",
              "Cross-Origin-Opener-Policy": "same-origin",
            },
          })
        : undefined,
    ],
  },
  // index.d.ts
  {
    input: "src/index.ts",
    output: [{ file: "dist/index.d.ts" }],
    plugins: [dts()],
  },
];
