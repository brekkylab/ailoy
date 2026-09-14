// The syntax highlighter, assembled by hand instead of taken off the shelf.
//
// Two reasons, and the first one is not an optimisation. `react-shiki`'s default export
// carries shiki's Oniguruma engine, which is WebAssembly, and the packaged app's CSP has
// no `script-src` — so `default-src 'self'` applies and WebKit refuses
// `WebAssembly.instantiate` outright. Under `tauri dev` the dev server's own CSP is
// looser and nothing goes wrong; in the `.app` every fence in the thread would fail to
// highlight, in a webview whose console the user cannot open. shiki's JavaScript regex
// engine compiles the same TextMate grammars to native `RegExp`, so there is no wasm for
// the policy to refuse. (`tauri.conf.json` also grants `'wasm-unsafe-eval'` now, but that
// is the belt; this is the braces, and it is the one that removes the failure mode.)
//
// The second reason is size: the default bundle registers ~200 grammars and weighs 11 MB
// of `dist`. `react-shiki/core` refuses to guess and demands a highlighter, which is what
// lets the list below be the fences this app actually sees — the languages of the
// workspace it sits next to, plus the ones a model reaches for when it explains itself.
// Everything else degrades to unhighlighted plaintext, which is what an unknown fence did
// before as well.
//
// `forgiving: true` is about the engine, not about us: a handful of TextMate patterns use
// Oniguruma constructs JavaScript's `RegExp` has no spelling for. Without it, building the
// highlighter throws on the first such pattern and no fence anywhere gets highlighted;
// with it, that one pattern simply never matches and the rest of the grammar works.

import { createHighlighterCore, createJavaScriptRegexEngine } from "react-shiki/core";

export type Highlighter = Awaited<ReturnType<typeof createHighlighterCore>>;

/**
 * Grammar modules to register. `html` embeds its own copies of javascript and css, which
 * is why those three arrive together; the aliases each grammar declares (`ts`, `sh`,
 * `py`, `rs`, `yml`, `md`, `dockerfile`, …) are resolved by shiki, so a fence may be
 * spelled either way.
 */
const LANGS = [
  () => import("@shikijs/langs/typescript"),
  () => import("@shikijs/langs/tsx"),
  () => import("@shikijs/langs/javascript"),
  () => import("@shikijs/langs/json"),
  () => import("@shikijs/langs/bash"),
  () => import("@shikijs/langs/rust"),
  () => import("@shikijs/langs/toml"),
  () => import("@shikijs/langs/python"),
  () => import("@shikijs/langs/yaml"),
  () => import("@shikijs/langs/markdown"),
  () => import("@shikijs/langs/diff"),
  () => import("@shikijs/langs/go"),
  () => import("@shikijs/langs/docker"),
  () => import("@shikijs/langs/html"),
  () => import("@shikijs/langs/css"),
  () => import("@shikijs/langs/sql"),
];

/**
 * Built once per process and shared. Lazy rather than top-level so that importing this
 * module — which a test or a tree-shaken build may do for the type alone — does not start
 * fetching sixteen grammar chunks.
 */
let pending: Promise<Highlighter> | null = null;

export function getHighlighter(): Promise<Highlighter> {
  pending ??= createHighlighterCore({
    engine: createJavaScriptRegexEngine({ forgiving: true }),
    // `github-light` is loaded but not yet chosen: the thread paints `github-dark` in both
    // colour schemes today, and switching that is a visual change this fix does not make.
    // Having the theme registered is what a later `{ light, dark }` needs and costs ~10 KB.
    themes: [import("@shikijs/themes/github-dark"), import("@shikijs/themes/github-light")],
    langs: LANGS.map((load) => load()),
  });
  return pending;
}
