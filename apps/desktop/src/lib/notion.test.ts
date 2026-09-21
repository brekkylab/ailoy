import { describe, expect, it } from "vitest";

import { notionHeading, notionMarkdown, notionTitle } from "@/lib/notion";

const page = (extra: Record<string, unknown> = {}) =>
  JSON.stringify({
    page_id: "2f1c",
    title: "Roadmap",
    properties: {},
    blocks: [],
    markdown: "# Roadmap\n\nShipping this quarter.",
    ...extra,
  });

describe("notionTitle", () => {
  it("drops the page id cortex appends to the directory name", () => {
    expect(notionTitle("Roadmap__2f1c9a0e")).toBe("Roadmap");
  });

  it("splits from the right, so a title containing __ survives", () => {
    expect(notionTitle("draft__notes__2f1c9a0e")).toBe("draft__notes");
  });

  it("leaves a name with no id alone", () => {
    expect(notionTitle("pages")).toBe("pages");
  });

  it("leaves a name that is only an id alone", () => {
    // `__2f1c` has its separator at 0: there is no title to take, and returning "" would
    // put a nameless row in the tree.
    expect(notionTitle("__2f1c")).toBe("__2f1c");
  });
});

describe("notionMarkdown", () => {
  it("picks the rendered body out of a page", () => {
    expect(notionMarkdown(page())).toBe("# Roadmap\n\nShipping this quarter.");
  });

  it("is null for a database index, which carries no body", () => {
    const db = JSON.stringify({ database_id: "d1", title: "Tasks", rows: [] });
    expect(notionMarkdown(db)).toBeNull();
  });

  it("is null for a page whose body is empty, so the JSON shows instead", () => {
    expect(notionMarkdown(page({ markdown: "   " }))).toBeNull();
  });

  it("is null for a body that is not a string", () => {
    expect(notionMarkdown(page({ markdown: 42 }))).toBeNull();
  });

  it("is null for anything that is not a JSON object", () => {
    expect(notionMarkdown("not json at all")).toBeNull();
    expect(notionMarkdown("[1, 2, 3]")).toBeNull();
    expect(notionMarkdown("null")).toBeNull();
  });
});

describe("notionHeading", () => {
  it("reads the title recorded in the page", () => {
    expect(notionHeading(page())).toBe("Roadmap");
  });

  it("is null when there is none to read", () => {
    expect(notionHeading(JSON.stringify({ page_id: "x" }))).toBeNull();
    expect(notionHeading("not json")).toBeNull();
  });
});
