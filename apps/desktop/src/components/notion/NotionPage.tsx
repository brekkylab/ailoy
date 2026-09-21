// A Notion page's body, with its sub-pages as links.
//
// Cortex renders a `child_page` block as a marker — `[page: 회의록]` — because the content
// is not there: it lives behind that child's own directory, and a line saying where it
// went reads better than a gap where a page used to be. Here, where the directory is one
// click away, the marker becomes the link it stands for.
//
// The child's directory is found by id rather than by rebuilding the name cortex
// sanitized, and its icon comes from its own json, on the same query the tree row beside
// it already made.

import { useQueries, useQuery } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";

import * as api from "@/api";
import { Markdown } from "@/components/Markdown";
import { notionNodeQuery } from "@/components/notion/node";
import {
  DB_ICON,
  PAGE_ICON,
  type ChildLink,
  notionChildDir,
  notionChildren,
  notionHeading,
  notionIcon,
  withChildLinks,
} from "@/lib/notion";
import { S } from "@/strings";

/** Marks a link this panel handles itself. Percent-encoded, so any page name survives it. */
const INTERNAL = "ailoy-page:";

export function NotionPage({
  path,
  text,
  body,
  truncated,
  onOpen,
}: {
  /** The page's directory, which its children hang off. */
  path: string;
  /** The whole `page.json`, for the title, the icon, and the child blocks. */
  text: string;
  /** The rendered body, already picked out of it. */
  body: string;
  truncated: boolean;
  onOpen: (path: string) => void;
}) {
  const children = useMemo(() => notionChildren(text), [text]);
  // Only when there is something to place: a leaf page would pay cortex a whole render for
  // a listing with nothing in it to match against.
  const listing = useQuery({
    queryKey: ["fs", path],
    queryFn: () => api.fsList(path),
    enabled: children.length > 0,
  });
  const placed = useMemo(() => {
    const names = (listing.data ?? []).map((e) => e.name);
    return children.map((child) => ({ child, dir: notionChildDir(names, child) }));
  }, [children, listing.data]);

  const dirs = useMemo(
    () => placed.map((p) => p.dir).filter((d): d is string => d !== null),
    [placed],
  );
  const nodes = useQueries({ queries: dirs.map((d) => notionNodeQuery(`${path}/${d}`)) });
  const icons = new Map<string, string>();
  dirs.forEach((d, i) => {
    const node = nodes[i]?.data;
    const icon = node && !node.truncated && node.text ? notionIcon(node.text) : null;
    if (icon) icons.set(d, icon);
  });

  const links: ChildLink[] = placed.map(({ child, dir }) => ({
    title: child.title,
    kind: child.kind,
    href: dir === null ? null : INTERNAL + encodeURIComponent(`${path}/${dir}`),
    icon: (dir && icons.get(dir)) ?? (child.kind === "database" ? DB_ICON : PAGE_ICON),
  }));

  const resolveLink = useCallback(
    (href: string) => {
      if (!href.startsWith(INTERNAL)) return null;
      const target = decodeURIComponent(href.slice(INTERNAL.length));
      return () => onOpen(target);
    },
    [onOpen],
  );

  const heading = notionHeading(text);
  const icon = notionIcon(text);

  return (
    <>
      {/* The title lives beside the body rather than in it: cortex renders the blocks, and
          a page's name is not one of them. */}
      {heading && (
        <h2 className="mb-2 flex items-baseline gap-2 text-lg font-semibold">
          <span>{icon ?? PAGE_ICON}</span>
          <span>{heading}</span>
        </h2>
      )}
      <Markdown text={withChildLinks(body, links)} resolveLink={resolveLink} />
      {truncated && <p className="mt-2 text-xs text-muted-foreground">… {S.fileTooLarge}</p>}
    </>
  );
}
