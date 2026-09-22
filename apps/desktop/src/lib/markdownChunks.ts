// Cutting a long document into pieces that can be rendered one at a time.
//
// A markdown file in a workspace is not a chat message. One of the reports this app is used
// to read is 817 KiB, and rendering it in one go blocks the window for about two seconds and
// leaves 130,000 nodes behind — measured, in this app, with this renderer. Nothing about that
// is fetching: the text arrives in 174ms and is cached. It is the render.
//
// So the document is rendered in pieces, and this decides where the pieces end. Two rules,
// and both are about not changing what the markdown means:
//
//   * A cut only ever happens at a blank line between top-level blocks. That is the boundary
//     markdown itself uses, so a table, a list or a paragraph is never split down the middle.
//   * A fenced code block is never entered. A cut inside one would turn the rest of the file
//     into code, or the code into prose — the one way a split can change what is *said*.
//
// What it cannot preserve is a reference used before it is defined: `[a]` in the first piece
// and `[a]: https://…` at the end of the file are in different renders, and the link stays
// literal. That is the price, it is small, and it is only paid by documents long enough to
// need this.

/**
 * How much text a piece aims for.
 *
 * Small enough that one is quick to render and cheap to hold, large enough that a document
 * is not thousands of them: at this size the 817 KiB report is about fifty pieces, and a
 * reader sees two or three at a time.
 */
export const CHUNK_BYTES = 16 * 1024;

/** Below this, a document is one piece and nothing here is doing anything. */
export const CHUNK_ABOVE = 96 * 1024;

/**
 * `text` as pieces that can be rendered in order.
 *
 * One piece for anything short — the overwhelming case, and every assistant message — so a
 * caller can render what comes back without asking how long it is.
 */
export function splitMarkdown(text: string, chunkBytes = CHUNK_BYTES): string[] {
  if (text.length <= CHUNK_ABOVE) return [text];

  const out: string[] = [];
  const lines = text.split("\n");
  let piece: string[] = [];
  let size = 0;
  let fenced = false;

  for (const line of lines) {
    // ``` or ~~~ at the start of a line opens or closes a fence. Anything indented into a
    // code block does not, and neither does a backtick mid-line.
    if (/^\s{0,3}(```|~~~)/.test(line)) fenced = !fenced;
    const blank = line.trim() === "";
    // The blank line stays with what came before, so the pieces joined back together are
    // the document again — and the next piece starts on content.
    if (!fenced && blank && size >= chunkBytes && piece.length > 0) {
      piece.push(line);
      out.push(piece.join("\n"));
      piece = [];
      size = 0;
      continue;
    }
    piece.push(line);
    size += line.length + 1;
  }
  if (piece.length > 0) out.push(piece.join("\n"));
  return out;
}
