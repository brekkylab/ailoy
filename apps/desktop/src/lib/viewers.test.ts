import { describe, expect, it } from "vitest";

import { extOf, viewerFor } from "@/lib/viewers";

describe("extOf", () => {
  it("takes what follows the last dot, lowercased", () => {
    expect(extOf("report.CSV")).toBe("csv");
    expect(extOf("archive.tar.gz")).toBe("gz");
  });

  it("is empty for a name with no extension", () => {
    expect(extOf("Makefile")).toBe("");
  });

  it("is empty for a dotfile, whose dot starts the name rather than an extension", () => {
    expect(extOf(".gitignore")).toBe("");
  });
});

describe("viewerFor", () => {
  it("picks by extension", () => {
    expect(viewerFor("notes.md").kind).toBe("markdown");
    expect(viewerFor("rows.csv")).toMatchObject({ kind: "table", delimiter: "," });
    expect(viewerFor("rows.tsv")).toMatchObject({ kind: "table", delimiter: "\t" });
    expect(viewerFor("main.rs")).toMatchObject({ kind: "code", lang: "rust" });
  });

  it("reads a path, not just a bare name", () => {
    expect(viewerFor("/archive/2026/data.csv").kind).toBe("table");
  });

  it("falls back to text rather than to nothing", () => {
    // Every file opens; an unknown one opens as characters, which is what it did before.
    expect(viewerFor("mystery.bin").kind).toBe("text");
    expect(viewerFor("LICENSE").kind).toBe("text");
  });

  it("names a log a log, while showing it the same way", () => {
    expect(viewerFor("run.log")).toMatchObject({ kind: "text", label: "Log" });
  });

  it("knows a Dockerfile by its whole name", () => {
    expect(viewerFor("/app/Dockerfile")).toMatchObject({ kind: "code", lang: "docker" });
  });

  it("shows HTML as source rather than rendering it", () => {
    // The webview carries the app's own origin and its bridge to the engine, so a
    // document off a bucket must not run in it.
    expect(viewerFor("page.html")).toMatchObject({ kind: "code", lang: "html" });
  });

  it("asks only for grammars the highlighter registers", () => {
    const registered = new Set([
      "typescript", "tsx", "javascript", "json", "bash", "rust", "toml", "python",
      "yaml", "markdown", "diff", "go", "docker", "html", "css", "sql",
    ]);
    for (const name of [
      "a.ts", "a.tsx", "a.js", "a.jsx", "a.mjs", "a.cjs", "a.json", "a.jsonl", "a.py",
      "a.rs", "a.go", "a.sql", "a.css", "a.toml", "a.yaml", "a.yml", "a.sh", "a.bash",
      "a.zsh", "a.diff", "a.patch", "a.html", "a.htm", "Dockerfile",
    ]) {
      const v = viewerFor(name);
      expect(v.kind).toBe("code");
      expect(registered.has(v.lang!), `${name} asks for ${v.lang}`).toBe(true);
    }
  });
});
