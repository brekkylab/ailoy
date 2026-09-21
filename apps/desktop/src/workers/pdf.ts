// pdf.js's worker, with the polyfill it needs in front of it.
//
// A worker is its own realm: patching `Map.prototype` on the main thread does not reach
// it, and the worker bundle uses the same proposal methods the main one does. So the
// worker this app starts is this module, which installs them and then *is* pdf.js's
// worker.
//
// Vite compiles this to its own bundle because `PdfViewer` constructs it with
// `new Worker(new URL(...), { type: "module" })`, which is the form it recognises.

import "@/lib/mapUpsert";
import "pdfjs-dist/build/pdf.worker.min.mjs";
