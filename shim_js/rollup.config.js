import commonjs from "@rollup/plugin-commonjs";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";
import copy from "rollup-plugin-copy";

const tsconfig = "tsconfig.build.json";

export default {
  input: "src/index.ts",
  output: [
    {
      file: "dist/index.js",
      format: "esm",
      exports: "named",
    },
  ],
  plugins: [
    commonjs(),
    typescript({
      tsconfig,
    }),
    copy({
      targets: [{ src: "src/faiss/faiss_bridge.wasm", dest: "dist/" }],
    }),
    terser(),
  ],
};
