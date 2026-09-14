// Stands in for `shiki/engine/oniguruma` and `shiki/wasm` in the bundle. Wired up in
// `vite.config.ts`; nothing imports this file by name.
//
// `lib/highlighter.ts` hands shiki the JavaScript regex engine, so the Oniguruma one is
// never built — but react-shiki still names it in two places its bundler cannot drop: a
// static re-export in `react-shiki/core`, and an `ENGINES` lookup keyed by the string
// `"oniguruma"`. Between them they pull `WebAssembly.instantiate` into the entry chunk and
// a 622 kB inlined `onig.wasm` into a chunk of its own — 26% of `dist`, fetched never.
//
// Aliasing them away is what makes that true on disk as well as in practice, and it turns
// the one thing that must not happen in the packaged app — a wasm engine being built
// behind the CSP, failing silently, and leaving every fence unhighlighted — into a throw
// with a sentence in it.

const REASON = "shiki's Oniguruma engine is not bundled: the app's CSP has no wasm, so lib/highlighter.ts uses the JavaScript regex engine";

export function createOnigurumaEngine(): never {
  throw new Error(REASON);
}

export function loadWasm(): never {
  throw new Error(REASON);
}

export function setDefaultWasmLoader(): void {}

export default createOnigurumaEngine;
