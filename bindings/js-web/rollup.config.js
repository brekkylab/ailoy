import { nodeResolve } from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import typescript from "@rollup/plugin-typescript";
import ignore from "rollup-plugin-ignore";
import copy from "rollup-plugin-copy";
import serve from "rollup-plugin-serve";

const enableDevServer = Boolean(process.env.AILOY_WEB_DEV_SERVER);

module.exports = {
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
    copy({
      targets: [
        {
          src: ["src/ailoy_js_web.js", "src/ailoy_js_web.wasm"],
          dest: "dist",
        },
      ],
    }),
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
};
