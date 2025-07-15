import typescript from "rollup-plugin-typescript2";
import copy from "rollup-plugin-copy";
import serve from "rollup-plugin-serve";
import commonjs from "@rollup/plugin-commonjs";

const enableDevServer = Boolean(process.env.DEV_SERVER);

export default [
  {
    input: "src/index.ts",
    output: {
      dir: "dist",
      format: "esm",
      sourcemap: true,
    },
    plugins: [
      commonjs(),
      typescript(),
      enableDevServer
        ? copy({
            targets: [{ src: "src/index.html", dest: "dist" }],
          })
        : undefined,
      enableDevServer
        ? serve({
            contentBase: "dist",
            port: 3000,
            headers: {
              "Cross-Origin-Embedder-Policy": "require-corp",
              "Cross-Origin-Opener-Policy": "same-origin",
            },
          })
        : undefined,
    ],
  },
  {
    input: "src/runtime.worker.ts",
    output: {
      dir: "dist",
      format: "esm",
      sourcemap: true,
    },
    external: (id) => {
      if (id.includes("pakky_js_browser.js")) {
        return true;
      }
      return false;
    },
    plugins: [
      commonjs(),
      typescript(),
      copy({
        targets: [
          {
            src: ["src/pakky_js_browser.js", "src/pakky_js_browser.wasm"],
            dest: "dist",
          },
        ],
      }),
    ],
  },
];
