// Stands in for `shiki/engine/oniguruma` and `shiki/wasm` in the bundle. Wired up in
// `vite.config.ts`; nothing imports this file by name.
//
// `lib/highlighter.ts` hands shiki the JavaScript regex engine, so the Oniguruma one is
// never built — but react-shiki still names it in two places its bundler cannot drop: a
// static re-export in `react-shiki/core`, and an `ENGINES` lookup keyed by the string
// `"oniguruma"`. Between them they pull `WebAssembly.instantiate` into the entry chunk and
// a 622 kB inlined `onig.wasm` into a chunk of its own — 26% of `dist`, fetched never.
//
// Aliasing them away is what makes that true on disk as well as in practice. It also turns
// a silent failure into a loud one: an engine reached for here would be an engine
// `lib/highlighter.ts` did not build, and the symptom — every fence unhighlighted, in a
// webview whose console the user cannot open — names nothing. A throw carries a sentence.

const REASON = "shiki's Oniguruma engine is not bundled: lib/highlighter.ts uses the JavaScript regex engine, and this file stands in for the one it does not build";

export function createOnigurumaEngine(): never {
  throw new Error(REASON);
}

export function loadWasm(): never {
  throw new Error(REASON);
}

export function setDefaultWasmLoader(): void {}

export default createOnigurumaEngine;
