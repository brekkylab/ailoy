import { nodeResolve } from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import ignore from "rollup-plugin-ignore";

module.exports = {
  input: "dist/index.js",
  output: {
    file: "dist/bundle.js",
    format: "es",
  },
  plugins: [
    ignore(["fs", "path", "crypto", "perf_hooks", "ws"]),
    nodeResolve({
      browser: true,
    }),
    commonjs(),
  ],
};
