// Stands in for `@napi-rs/canvas` in the bundle. Wired up in `vite.config.ts`; nothing
// imports this file by name.
//
// `emf-converter`, which `pptx-viewer-core` depends on for EMF and WMF pictures, reaches
// for that native Node addon when it finds neither `OffscreenCanvas` nor `document` — so
// never in a webview, and the reach is a dynamic import inside a `try`. Vite's dependency
// scanner does not know that and fails to resolve it, which is enough to stop the whole
// module graph.
//
// Throwing rather than exporting an empty module: the caller's `catch` turns this into the
// `null` its own code already handles, where an object with no `createCanvas` on it would
// become a TypeError further along, outside the `try`.

throw new Error(
  "@napi-rs/canvas is a Node addon and is not bundled: emf-converter only reaches for it where there is no canvas, which cannot happen in the webview",
);
