import { describe, expect, it } from "vitest";

import {
  isNotionDatabase,
  notionChildDir,
  notionChildren,
  notionHeading,
  notionIcon,
  notionMarkdown,
  notionRowCount,
  notionTitle,
  withChildLinks,
} from "@/lib/notion";

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

const child = (kind: "page" | "database", title: string, id: string) => ({
  type: `child_${kind}`,
  id,
  [`child_${kind}`]: { title },
});

describe("isNotionDatabase and notionTitle", () => {
  it("reads the kind off the directory name", () => {
    expect(isNotionDatabase("Tasks__db__2f1c9a0e")).toBe(true);
    expect(isNotionDatabase("Roadmap__2f1c9a0e")).toBe(false);
  });

  it("takes the marker off a database's title too", () => {
    // Left in, the row would read "Tasks__db" — which is the bug this pairs with.
    expect(notionTitle("Tasks__db__2f1c9a0e")).toBe("Tasks");
  });
});

describe("notionIcon", () => {
  it("is the page's own emoji", () => {
    expect(notionIcon(page({ icon: "🚀" }))).toBe("🚀");
  });

  it("is null when there is none, or when cortex never wrote the key", () => {
    expect(notionIcon(page({ icon: null }))).toBeNull();
    expect(notionIcon(page())).toBeNull();
    expect(notionIcon("not json")).toBeNull();
  });
});

describe("notionChildren", () => {
  it("finds a child however deep the layout buried it", () => {
    // Notion's two-column layout puts sub-pages two blocks below the page, and they are
    // children of the page all the same — which is how cortex decides what directories to
    // put beside `page.json`.
    const text = page({
      blocks: [
        child("page", "회의록", "aaaa"),
        {
          type: "column_list",
          id: "col",
          children: [
            { type: "column", id: "c1", children: [child("database", "Tasks", "bbbb")] },
          ],
        },
      ],
    });
    expect(notionChildren(text)).toEqual([
      { id: "aaaa", title: "회의록", kind: "page" },
      { id: "bbbb", title: "Tasks", kind: "database" },
    ]);
  });

  it("does not walk through a child, whose blocks are its own", () => {
    const nested = { ...child("page", "부모", "aaaa"), children: [child("page", "자식", "bbbb")] };
    expect(notionChildren(page({ blocks: [nested] }))).toEqual([
      { id: "aaaa", title: "부모", kind: "page" },
    ]);
  });

  it("is empty for a database index, which has no blocks", () => {
    expect(notionChildren(JSON.stringify({ rows: [{ dir: "a__1" }] }))).toEqual([]);
    expect(notionRowCount(JSON.stringify({ rows: [{ dir: "a__1" }] }))).toBe(1);
  });
});

describe("notionChildDir", () => {
  const dirs = ["page.json", "회의록__aaaa", "Tasks__db__bbbb", "다른회의록__cccc"];

  it("matches on the id, which no sanitizer touches", () => {
    expect(notionChildDir(dirs, { id: "aaaa", title: "회의록", kind: "page" })).toBe("회의록__aaaa");
    expect(notionChildDir(dirs, { id: "bbbb", title: "Tasks", kind: "database" })).toBe(
      "Tasks__db__bbbb",
    );
  });

  it("does not hand a page a database's directory, or the reverse", () => {
    expect(notionChildDir(dirs, { id: "bbbb", title: "Tasks", kind: "page" })).toBeNull();
  });

  it("is null when the listing does not have it", () => {
    expect(notionChildDir(dirs, { id: "zzzz", title: "gone", kind: "page" })).toBeNull();
  });
});

describe("withChildLinks", () => {
  const links = [
    { title: "회의록", kind: "page" as const, icon: "📝", href: "n:1" },
    { title: "Tasks", kind: "database" as const, icon: "🗂️", href: "n:2" },
  ];

  it("turns cortex's markers into links, keeping the line around them", () => {
    const body = "- [page: 회의록]\n\ntext\n\n[database: Tasks]";
    expect(withChildLinks(body, links)).toBe(
      "- [📝 회의록](n:1)\n\ntext\n\n[🗂️ Tasks](n:2)",
    );
  });

  it("leaves prose that merely looks like a marker alone", () => {
    // Order alone would send this one to the first child's page.
    expect(withChildLinks("see [page: something else]", links)).toBe(
      "see [page: something else]",
    );
  });

  it("escapes a title that would end the label early", () => {
    const odd = [{ title: "a]b", kind: "page" as const, icon: "📄", href: "n:1" }];
    expect(withChildLinks("[page: a]b]", odd)).toBe("[page: a]b]");
  });

  it("leaves the body alone when there is nothing to link", () => {
    expect(withChildLinks("[page: 회의록]", [])).toBe("[page: 회의록]");
  });
});
