// The highlighter is the one dependency whose failure the app cannot report: it lives in
// a webview with no reachable console, and a grammar the JavaScript regex engine cannot
// compile makes *every* fence render plain, not just that language's. So the build is
// exercised here, where a throw is a red test instead of a silently unstyled thread.

import { describe, expect, it } from "vitest";

import { getHighlighter } from "@/lib/highlighter";

/** Spelled the way a model writes an info string, aliases included. */
const FENCES: Record<string, string> = {
  ts: "const x: number = 1;",
  tsx: "const A = () => <div className=\"a\">hi</div>;",
  js: "export const f = async () => 1;",
  json: '{"a": [1, 2], "b": null}',
  bash: "set -euo pipefail\nfor f in *.txt; do echo \"$f\"; done",
  sh: "echo ${HOME:-/tmp}",
  rust: "pub fn main() -> anyhow::Result<()> { Ok(()) }",
  toml: "[package]\nname = \"ailoy\"",
  python: "def f(x: int) -> int:\n    return x + 1",
  yaml: "a:\n  - b: 1",
  markdown: "# title\n\n- a `b`",
  diff: "--- a\n+++ b\n-old\n+new",
  go: "package main\n\nfunc main() { println(\"hi\") }",
  dockerfile: "FROM alpine:3\nRUN apk add --no-cache curl",
  html: "<p class=\"a\">hi</p>",
  css: ".a { color: red; }",
  sql: "select id from t where a = 1;",
};

describe("highlighter", () => {
  it("builds with no wasm engine and highlights every fence the app claims to support", async () => {
    const hl = await getHighlighter();
    const loaded = hl.getLoadedLanguages();
    for (const [lang, code] of Object.entries(FENCES)) {
      expect(loaded, lang).toContain(lang);
      // A grammar that failed to compile throws here rather than returning plain text.
      const html = hl.codeToHtml(code, { lang, theme: "github-dark" });
      expect(html, lang).toContain("<span");
    }
  }, 30_000);

  it("registers both themes", async () => {
    const hl = await getHighlighter();
    expect(hl.getLoadedThemes()).toEqual(expect.arrayContaining(["github-dark", "github-light"]));
  });

  it("is built once and shared", async () => {
    expect(await getHighlighter()).toBe(await getHighlighter());
  });
});
