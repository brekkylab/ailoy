import { describe, expect, it } from "vitest";

import { CHUNK_ABOVE, splitMarkdown } from "./markdownChunks";

/** A document of `blocks` paragraphs, each one long enough to count. */
const doc = (blocks: number, body = "x".repeat(1000)) =>
  Array.from({ length: blocks }, (_, i) => `## Heading ${i}\n\n${body}`).join("\n\n");

describe("splitMarkdown", () => {
  it("leaves a short document alone", () => {
    const text = doc(3);
    expect(text.length).toBeLessThan(CHUNK_ABOVE);
    expect(splitMarkdown(text)).toEqual([text]);
  });

  it("cuts a long one into pieces that join back into it", () => {
    const text = doc(400);
    const pieces = splitMarkdown(text);
    expect(pieces.length).toBeGreaterThan(1);
    // Nothing added, nothing lost: the pieces are the document.
    expect(pieces.join("\n")).toBe(text);
  });

  it("never cuts inside a fence", () => {
    // A fence long enough to span a chunk on its own, with prose either side.
    const fence = ["```ts", ...Array.from({ length: 4000 }, (_, i) => `const x${i} = ${i};`), "```"].join("\n");
    const text = `${doc(60)}\n\n${fence}\n\n${doc(60)}`;
    for (const piece of splitMarkdown(text)) {
      const fences = piece.split("\n").filter((l) => /^\s{0,3}(```|~~~)/.test(l)).length;
      expect(fences % 2, `a piece that opens a fence has to close it:\n${piece.slice(0, 80)}`).toBe(0);
    }
  });

  it("cuts on blank lines, so a block is never halved", () => {
    const text = doc(400);
    for (const piece of splitMarkdown(text)) {
      expect(piece.startsWith("## ") || piece.startsWith("\n")).toBe(true);
    }
  });

  it("keeps going when one block is larger than a chunk", () => {
    // A single paragraph of half a megabyte: there is nowhere to cut, and the answer is one
    // piece rather than a loop that never advances.
    const text = `# One block\n\n${"y".repeat(500_000)}`;
    expect(splitMarkdown(text)).toHaveLength(1);
  });
});
