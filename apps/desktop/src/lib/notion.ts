// Reading what cortex's Notion filesystem actually writes.
//
// It does not hand back markdown. Every page is a directory holding a `page.json` whose
// shape is Notion's API response, normalized — page_id, title, properties, blocks, and a
// `markdown` key with the body already rendered out of those blocks. Picking that key out
// is ours to do: cortex's job ends at serializing what Notion said.
//
// A database is the same layout with a `database.json` that has no body at all — it is a
// schema and an index of rows — so "is there a markdown key" is also how a reader tells
// the two apart, without a second request to find out which it is holding.

/** Marks a database directory: `<title>__db__<database-id>`. Cortex's own separator. */
const DB_MARKER = "__db__";

/** What a page with no icon of its own is drawn with, as Notion draws it. */
export const PAGE_ICON = "\u{1f4c4}";
/** The same, for a database. */
export const DB_ICON = "\u{1f5c2}\ufe0f";

/**
 * The title out of a page directory's name.
 *
 * Cortex names them `<sanitized-title>__<page-id>`, so the id is everything after the last
 * double underscore. A title that itself contains `__` keeps it: the split is from the
 * right, and the id is the part that never does.
 */
export function notionTitle(dirName: string): string {
  const db = dirName.lastIndexOf(DB_MARKER);
  if (db > 0) return dirName.slice(0, db);
  const cut = dirName.lastIndexOf("__");
  if (cut <= 0) return dirName;
  return dirName.slice(0, cut);
}

/**
 * Whether a directory is a database rather than a page.
 *
 * The name is the whole answer, and deliberately so on cortex's side: a path there
 * resolves without fetching its parent, so the kind has to be in the segment. It means a
 * reader knows which of `page.json` / `database.json` to ask for without spending a
 * refused request to find out.
 */
export function isNotionDatabase(dirName: string): boolean {
  return dirName.lastIndexOf(DB_MARKER) > 0;
}

/**
 * The rendered body of a `page.json`, or `null` for anything that is not one — a database
 * index, a file that is not JSON, JSON that is not an object, or a page whose `markdown`
 * is missing or empty.
 *
 * Empty counts as absent on purpose: a page with no blocks renders to an empty string, and
 * an empty markdown pane says less than the JSON behind it would.
 */
export function notionMarkdown(text: string): string | null {
  const body = parseObject(text)?.markdown;
  if (typeof body !== "string" || body.trim() === "") return null;
  return body;
}

/** The title recorded inside a `page.json`, which beats the one sanitized into its path. */
export function notionHeading(text: string): string | null {
  const title = parseObject(text)?.title;
  return typeof title === "string" && title.trim() !== "" ? title : null;
}

/**
 * The page's emoji, or `null` — including for a page whose icon is an uploaded image,
 * which cortex does not carry because its URL is signed and expires.
 *
 * `null` is also what an older cortex gives, which had no `icon` key at all. The caller
 * falls back to [`PAGE_ICON`] either way, so the tree draws the same rows against either
 * one rather than refusing to draw at all.
 */
export function notionIcon(text: string): string | null {
  const obj = parseObject(text);
  const icon = obj?.icon;
  return typeof icon === "string" && icon !== "" ? icon : null;
}

/** A page or database sitting inside another page, as its own block says it. */
export type NotionChild = {
  id: string;
  title: string;
  kind: "page" | "database";
};

/**
 * Every page and database nested in this one, in the order the body mentions them.
 *
 * Depth is the point: Notion puts the sub-pages of a two-column page inside a `column`
 * two blocks down, and they are children of the page all the same — which is exactly how
 * cortex decides what directories to put beside `page.json`, so walking the same tree the
 * same way is what keeps the two agreeing.
 *
 * A database index has no blocks and answers empty; its children are its rows, which
 * [`notionRowCount`] counts instead.
 */
export function notionChildren(text: string): NotionChild[] {
  const blocks = parseObject(text)?.blocks;
  const out: NotionChild[] = [];
  if (Array.isArray(blocks)) collectChildren(blocks, out);
  return out;
}

function collectChildren(blocks: unknown[], out: NotionChild[]): void {
  for (const block of blocks) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    const type = b.type;
    if (type === "child_page" || type === "child_database") {
      const payload = b[type];
      const title =
        typeof payload === "object" && payload !== null
          ? (payload as Record<string, unknown>).title
          : undefined;
      out.push({
        id: typeof b.id === "string" ? b.id : "",
        title: typeof title === "string" ? title : "",
        kind: type === "child_page" ? "page" : "database",
      });
      // Never descended into: the child's own blocks live behind its own directory.
      continue;
    }
    if (Array.isArray(b.children)) collectChildren(b.children, out);
  }
}

/** How many rows a `database.json` indexes, for a tree deciding whether to offer an expander. */
export function notionRowCount(text: string): number {
  const rows = parseObject(text)?.rows;
  return Array.isArray(rows) ? rows.length : 0;
}

/**
 * The directory in `dirs` that holds `child`, matched on the id rather than the title.
 *
 * Cortex builds the name from a *sanitized* title, so rebuilding it here would mean
 * carrying a copy of its sanitizer and keeping the two in step forever. The id is in both
 * and is altered by neither.
 *
 * The kind is checked as well as the id, because a database's `__db__<id>` ends with
 * `__<id>` too — so an id alone would hand a page its namesake database's directory.
 */
export function notionChildDir(dirs: string[], child: NotionChild): string | null {
  const wantDb = child.kind === "database";
  const tail = (wantDb ? DB_MARKER : "__") + child.id;
  return dirs.find((d) => d.endsWith(tail) && isNotionDatabase(d) === wantDb) ?? null;
}

/** A child page the body mentions, and where a reader should be sent for it. */
export type ChildLink = {
  title: string;
  kind: "page" | "database";
  /** `null` when the listing has no directory for it, and the marker stays as it was. */
  href: string | null;
  icon: string;
};

/**
 * Cortex's marker for a child page — `[page: <title>]` — so the rendered body says where
 * the content went rather than reading as a gap in the page.
 */
const MARKER = /\[(page|database): ([^\]\n]*)\]/g;

/**
 * The markers turned into links, so a page tree reads the way Notion's does.
 *
 * Matched in order and confirmed by title: the body is rendered from the same block tree
 * [`notionChildren`] walks, so the nth marker is the nth child. The title check is what
 * keeps a line of prose that happens to read `[page: x]` from becoming a link to
 * somewhere else — on a mismatch the marker is left exactly as cortex wrote it.
 */
export function withChildLinks(body: string, links: ChildLink[]): string {
  let next = 0;
  return body.replace(MARKER, (marker, kind: string, title: string) => {
    const link = links[next];
    if (!link || link.kind !== kind || link.title !== title) return marker;
    // Consumed either way: it is this child's marker whether or not there is anywhere to
    // send a reader, and the next marker belongs to the next child.
    next += 1;
    if (!link.href) return marker;
    return `[${escapeLabel(`${link.icon} ${title}`)}](${link.href})`;
  });
}

/** `[` and `]` would end the label early, and a backslash would escape whatever follows. */
function escapeLabel(label: string): string {
  return label.replace(/[\\[\]]/g, "\\$&");
}

function parseObject(text: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return null;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}
