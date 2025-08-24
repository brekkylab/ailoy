import commonjs from "@rollup/plugin-commonjs";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";

const tsconfig = "tsconfig.build.json";

export default [
  // main build
  {
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
      terser(),
    ],
  },
];
