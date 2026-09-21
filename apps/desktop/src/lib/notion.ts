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

/**
 * The title out of a page directory's name.
 *
 * Cortex names them `<sanitized-title>__<page-id>`, so the id is everything after the last
 * double underscore. A title that itself contains `__` keeps it: the split is from the
 * right, and the id is the part that never does.
 */
export function notionTitle(dirName: string): string {
  const cut = dirName.lastIndexOf("__");
  if (cut <= 0) return dirName;
  return dirName.slice(0, cut);
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
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
  const body = (parsed as Record<string, unknown>).markdown;
  if (typeof body !== "string" || body.trim() === "") return null;
  return body;
}

/** The title recorded inside a `page.json`, which beats the one sanitized into its path. */
export function notionHeading(text: string): string | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (typeof parsed !== "object" || parsed === null) return null;
    const title = (parsed as Record<string, unknown>).title;
    return typeof title === "string" && title.trim() !== "" ? title : null;
  } catch {
    return null;
  }
}
