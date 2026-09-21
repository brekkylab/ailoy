// A tree over one root, beside whatever is open in it.
//
// Two panes rather than one stacked on the other: these roots are browsed, not skimmed,
// and a tree that keeps its width while the content changes is the shape every file
// explorer settles on. The tree scrolls on its own, so a deep directory does not push the
// document out of view and a long document does not shorten the tree.
//
// `kind` is the only thing that differs between the sources, and only because Notion's
// layout means something. Everything else is files: what you click is what you read, and
// showing bytes as anything but themselves would be a guess about what they are.

import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import * as api from "@/api";
import { FileTree, type TreeAdapter } from "@/components/FileTree";
import { Markdown } from "@/components/Markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CodeViewer } from "@/components/viewers/CodeViewer";
import { TableViewer } from "@/components/viewers/TableViewer";
import { notionHeading, notionMarkdown, notionTitle } from "@/lib/notion";
import { DocxViewer } from "@/components/viewers/DocxViewer";
import { ImageViewer } from "@/components/viewers/ImageViewer";
import { PdfViewer } from "@/components/viewers/PdfViewer";
import { XlsxViewer } from "@/components/viewers/XlsxViewer";
import { readsText, viewerFor, type Viewer } from "@/lib/viewers";
import { S } from "@/strings";
import type { Entry, FileContent } from "@/types";

/**
 * Notion's shape, in one place.
 *
 * A page is a directory holding `page.json`, so the directory *is* the page: it carries
 * the title, and clicking it is asking for the body. The json files are hidden because
 * they are that body rather than siblings of it, and the id cortex sanitizes into the
 * directory name comes off, since the tree is a list of pages and not of paths.
 */
const NOTION: TreeAdapter = {
  label: (e) => (e.kind === "dir" ? notionTitle(e.name) : e.name),
  hide: (e) => e.name === "page.json" || e.name === "database.json",
  openDirs: true,
};

/**
 * Reads what a click asked for.
 *
 * For Notion that is a directory, and the file inside it is `page.json` — unless the
 * directory is a database, which holds a `database.json` instead. Nothing in the listing
 * distinguishes the two (cortex names both `<title>__<id>`), so the fallback is the way to
 * find out, and it costs one refused request on the rarer of the two.
 */
async function readTarget(
  path: string,
  kind: "plain" | "notion",
): Promise<FileContent> {
  if (kind !== "notion") return api.fsRead(path);
  try {
    return await api.fsRead(`${path}/page.json`);
  } catch (err) {
    if (api.kindOf(err) !== "not_found") throw err;
    return api.fsRead(`${path}/database.json`);
  }
}

/** A viewer that opens the file itself, from its address rather than from decoded text. */
function BinaryView({ path, viewer }: { path: string; viewer: Viewer }) {
  switch (viewer.kind) {
    case "image":
      return <ImageViewer path={path} />;
    case "pdf":
      return <PdfViewer path={path} />;
    case "docx":
      return <DocxViewer path={path} />;
    case "xlsx":
      return <XlsxViewer path={path} />;
    default:
      return null;
  }
}

/** The registry's choice, rendered. `text` never reaches here — it is the fallback below. */
function TypedView({
  text,
  viewer,
  truncated,
}: {
  text: string;
  viewer: Viewer;
  truncated: boolean;
}) {
  switch (viewer.kind) {
    case "markdown":
      return (
        <>
          <Markdown text={text} />
          {truncated && (
            <p className="mt-2 text-xs text-muted-foreground">
              … {S.fileTooLarge}
            </p>
          )}
        </>
      );
    case "table":
      return (
        <TableViewer
          text={text}
          delimiter={viewer.delimiter ?? ","}
          truncated={truncated}
        />
      );
    case "code":
      return (
        <>
          <CodeViewer text={text} lang={viewer.lang ?? "text"} />
          {truncated && (
            <p className="mt-2 text-xs text-muted-foreground">
              … {S.fileTooLarge}
            </p>
          )}
        </>
      );
    default:
      return null;
  }
}

export function FileBrowser({
  root,
  kind = "plain",
  hide,
  viewers = false,
}: {
  root: string;
  kind?: "plain" | "notion";
  /** Entries to leave out, on top of whatever the kind already hides. */
  hide?: (e: Entry) => boolean;
  /**
   * Show a file as its type rather than as characters — a table for a `.csv`, highlighted
   * source for a `.rs`, rendered prose for a `.md`. Off by default, and switched on per
   * source rather than everywhere, because it is the sources holding documents that earn
   * it: a Notion page already renders, and what the agent writes is read as it is written.
   */
  viewers?: boolean;
}) {
  // The kind's own rules and the caller's, as one predicate: a Notion tree hides the json
  // that is its body, and the root hides the sources grafted into it, and a tree could
  // want both.
  const adapter: TreeAdapter | undefined = useMemo(() => {
    const base = kind === "notion" ? NOTION : undefined;
    if (!hide) return base;
    return { ...base, hide: (e: Entry) => !!base?.hide?.(e) || hide(e) };
  }, [kind, hide]);
  // Keyed by path, so the pane follows the selection and an unopened browser fetches
  // nothing. Reset when the root changes, because a path under the old root means nothing
  // under the new one — which `WorkspacePanel` gets by keying this component on its root.
  const [selected, setSelected] = useState<string | null>(null);
  // What the registry says about the open file, before anything is fetched: a viewer that
  // opens the bytes itself does not want `fs_read`, which would read the whole container
  // only to decode it to null.
  const viewer =
    viewers && selected && kind !== "notion" ? viewerFor(selected) : null;
  const binary = viewer !== null && !readsText(viewer.kind);
  const fills = !!viewer?.fills;
  const file = useQuery({
    queryKey: ["file", selected, kind],
    queryFn: () => readTarget(selected!, kind),
    enabled: !!selected && !binary,
  });

  const text = file.data?.text ?? null;
  // Markdown for a Notion page, the raw bytes for everything else — including a Notion
  // database, whose json has no body to render and is more use shown as what it is.
  const body = text !== null && kind === "notion" ? notionMarkdown(text) : null;
  const heading =
    text !== null && kind === "notion" ? notionHeading(text) : null;

  return (
    <div className="flex min-h-0 flex-1">
      <ScrollArea className="w-72 shrink-0 border-t border-r px-2 py-1">
        <FileTree
          path={root}
          onOpen={setSelected}
          selected={selected}
          adapter={adapter}
        />
      </ScrollArea>
      {/* A column rather than one scrolling box: the path line stays put, and what is under
          it either scrolls on its own padding or takes the rest of the height outright. */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col border-t">
        {!selected ? (
          <div className="grid h-full place-items-center p-6 text-sm text-muted-foreground">
            {S.pickFile}
          </div>
        ) : (
          <>
            <div className="flex shrink-0 items-baseline gap-2 px-4 pt-4 pb-3 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={selected}>
                {selected}
              </span>
              {viewer && (binary || text !== null) && (
                <span className="shrink-0">· {viewer.label}</span>
              )}
            </div>
            {/* A filling viewer gets the height and no padding; everything else scrolls
                inside its own. */}
            <div
              className={
                fills
                  ? "min-h-0 flex-1"
                  : "min-h-0 flex-1 overflow-auto px-4 pb-4"
              }
            >
              {binary && viewer ? (
                // Keyed: each of these fetches and decodes on mount, so another file is
                // another mount rather than an effect undoing the last one's state.
                <BinaryView key={selected} path={selected} viewer={viewer} />
              ) : body !== null ? (
                <>
                  {/* The title lives beside the body rather than in it: cortex renders the
                    blocks, and a page's name is not one of them. */}
                  {heading && (
                    <h2 className="mb-2 text-lg font-semibold">{heading}</h2>
                  )}
                  <Markdown text={body} />
                  {file.data?.truncated && (
                    <p className="mt-2 text-xs text-muted-foreground">
                      … {S.fileTooLarge}
                    </p>
                  )}
                </>
              ) : text !== null && viewer && viewer.kind !== "text" ? (
                <TypedView
                  text={text}
                  viewer={viewer}
                  truncated={!!file.data?.truncated}
                />
              ) : text !== null ? (
                <pre className="whitespace-pre-wrap font-mono text-xs">
                  {text}
                  {file.data?.truncated && `\n… ${S.fileTooLarge}`}
                </pre>
              ) : file.data ? (
                <p className="text-xs text-muted-foreground">{S.binaryFile}</p>
              ) : null}
              {file.isError && (
                <p className="text-xs text-destructive">
                  {api.messageOf(file.error)}
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
