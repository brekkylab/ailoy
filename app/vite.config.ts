import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";

/**
 * pdf.js reads its CMaps (CJK text), standard fonts, ICC profiles and wasm decoders by URL
 * at run time, so they are served as they ship under `/pdfjs/` — by the dev server, and
 * copied into the build — rather than imported.
 */
function pdfjsAssets(): Plugin {
  const root = path.resolve(import.meta.dirname, "node_modules/pdfjs-dist");
  const dirs = ["cmaps", "standard_fonts", "iccs", "wasm"];
  return {
    name: "pdfjs-assets",
    configureServer(server) {
      server.middlewares.use("/pdfjs", (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? "").split("?")[0]).replace(/^\/+/, "");
        if (!dirs.includes(rel.split("/")[0]) || rel.includes("..")) return next();
        fs.readFile(path.join(root, rel), (err, data) => {
          if (err) return next();
          if (rel.endsWith(".wasm")) res.setHeader("Content-Type", "application/wasm");
          res.end(data);
        });
      });
    },
    generateBundle() {
      for (const dir of dirs) {
        for (const name of fs.readdirSync(path.join(root, dir))) {
          const file = path.join(root, dir, name);
          if (!fs.statSync(file).isFile()) continue;
          this.emitFile({ type: "asset", fileName: `pdfjs/${dir}/${name}`, source: fs.readFileSync(file) });
        }
      }
    },
  };
}

// Tauri drives this dev server: fixed port, fail if taken.
export default defineConfig({
  plugins: [svelte(), tailwindcss(), pdfjsAssets()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },
  clearScreen: false,
  server: { port: 1420, strictPort: true, watch: { ignored: ["**/src-tauri/**"] } },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
});
