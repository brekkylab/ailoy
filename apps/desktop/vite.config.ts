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
      // highlighter on the JavaScript regex engine because the packaged app's CSP will not
      // run wasm, but react-shiki references the Oniguruma one unconditionally — a static
      // re-export plus a string-keyed `ENGINES` map — and neither reference can be
      // tree-shaken. The stub keeps the module graph resolvable and throws if anything ever
      // does reach for it. See `src/lib/onigurumaStub.ts`.
      "shiki/engine/oniguruma": path.join(src, "lib/onigurumaStub.ts"),
      "shiki/wasm": path.join(src, "lib/onigurumaStub.ts"),
    },
  },
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
});
