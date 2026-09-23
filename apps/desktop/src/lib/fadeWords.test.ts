import type { Element, Root } from "hast";
import { describe, expect, it } from "vitest";

import { rehypeFadeWords } from "./fadeWords";

const el = (tagName: string, children: Element["children"]): Element => ({ type: "element", tagName, properties: {}, children });
const words = (node: Root | Element): string[] =>
  node.children.flatMap((c) =>
    c.type === "element" ? (c.tagName === "span" ? [`[${(c.children[0] as { value: string }).value}]`] : words(c)) : c.type === "text" ? [c.value] : [],
  );

describe("rehypeFadeWords", () => {
  it("wraps each word in a span and leaves the space between them as text", () => {
    const tree: Root = { type: "root", children: [el("p", [{ type: "text", value: "Hello  wide world" }])] };
    rehypeFadeWords()(tree);
    expect(words(tree)).toEqual(["[Hello]", "  ", "[wide]", " ", "[world]"]);
    const span = (tree.children[0] as Element).children[0] as Element;
    expect(span.properties.className).toEqual(["fade-word"]);
  });

  it("reaches into inline markup but not into code", () => {
    const tree: Root = {
      type: "root",
      children: [
        el("p", [el("strong", [{ type: "text", value: "bold words" }]), { type: "text", value: " and " }, el("code", [{ type: "text", value: "a b" }])]),
        el("pre", [el("code", [{ type: "text", value: "let x = 1" }])]),
      ],
    };
    rehypeFadeWords()(tree);
    expect(words(tree.children[0] as Element)).toEqual(["[bold]", " ", "[words]", " ", "[and]", " ", "a b"]);
    expect(words(tree.children[1] as Element)).toEqual(["let x = 1"]);
  });

  it("keeps the words already there in the same places as the text grows", () => {
    // What keeps them from fading again: the first spans of a longer text are the spans of
    // the shorter one, at the same positions, so React keys them the same.
    const render = (text: string) => {
      const tree: Root = { type: "root", children: [el("p", [{ type: "text", value: text }])] };
      rehypeFadeWords()(tree);
      return words(tree);
    };
    const before = render("The quick brown");
    const after = render("The quick brown fox jumps");
    expect(after.slice(0, before.length)).toEqual(before);
  });
});
