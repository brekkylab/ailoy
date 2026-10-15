// Streamed text, a word at a time.
//
// A reply arrives as the model writes it, in chunks of a few tokens, and painted as it comes
// each chunk lands in one hard step. This wraps every word of the rendered markdown in a
// `span.fade-word`, whose animation (`index.css`) fades it up from nothing — and only a
// word that has just arrived plays it. That part is React's doing, not this plugin's:
// react-markdown keys an element by its tag and its place among its siblings (`span-0`,
// `span-1`, …), so when the text grows the spans already on screen keep their keys and
// their DOM, and only the ones added at the end mount and animate.
//
// For the live reply only. A stored message has nothing arriving, and a document is long
// enough that a span per word would be real weight.

import type { Element, ElementContent, Root } from "hast";

/** Where words are left alone: code keeps its highlighting and its exact characters. */
const SKIP = new Set(["pre", "code"]);

function split(node: Root | Element) {
  const out: ElementContent[] = [];
  for (const child of node.children) {
    if (child.type === "text") {
      for (const token of child.value.split(/(\s+)/)) {
        if (!token) continue;
        out.push(
          /^\s+$/.test(token)
            ? { type: "text", value: token }
            : {
                type: "element",
                tagName: "span",
                properties: { className: ["fade-word"] },
                children: [{ type: "text", value: token }],
              },
        );
      }
    } else if (child.type === "element") {
      if (!SKIP.has(child.tagName)) split(child);
      out.push(child);
    } else if (child.type !== "doctype") {
      out.push(child);
    }
  }
  node.children = out;
}

/** The rehype plugin: every word outside code in a span of its own. */
export function rehypeFadeWords() {
  return (tree: Root) => split(tree);
}
