import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { defineConfig } from "vite";

const src = path.resolve(import.meta.dirname, "./src");

// Tauri drives this dev server: fixed port, fail if taken.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": src,
      // shiki's WebAssembly engine, cut out of the graph. `lib/highlighter.ts` builds the
      // highlighter on the JavaScript regex engine — first because the packaged app's CSP
      // refused wasm outright, and still, now that it grants `'wasm-unsafe-eval'`, because
      // it is smaller and depends on nothing a policy can withdraw. react-shiki references
      // the Oniguruma one unconditionally all the same — a static re-export plus a
      // string-keyed `ENGINES` map — and neither reference can be tree-shaken. The stub
      // keeps the module graph resolvable and throws if anything ever does reach for it.
      // See `src/lib/onigurumaStub.ts`.
      "shiki/engine/oniguruma": path.join(src, "lib/onigurumaStub.ts"),
      "shiki/wasm": path.join(src, "lib/onigurumaStub.ts"),
      // A Node addon `emf-converter` reaches for only where there is no canvas — never in
      // a webview, and behind a `try`. Vite's dependency scanner resolves it all the same,
      // and fails the graph when it cannot. See `src/lib/napiCanvasStub.ts`.
      "@napi-rs/canvas": path.join(src, "lib/napiCanvasStub.ts"),
    },
  },
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
});
