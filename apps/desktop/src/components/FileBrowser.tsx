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

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCw } from "lucide-react";
import { useMemo, useState } from "react";

import * as api from "@/api";
import { FileTree, type TreeAdapter } from "@/components/FileTree";
import { Markdown } from "@/components/Markdown";
import { NOTION } from "@/components/trees/notion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CodeViewer } from "@/components/viewers/CodeViewer";
import { DocxViewer } from "@/components/viewers/DocxViewer";
import { HwpViewer } from "@/components/viewers/HwpViewer";
import { ImageViewer } from "@/components/viewers/ImageViewer";
import { NotionViewer } from "@/components/viewers/NotionViewer";
import { PdfViewer } from "@/components/viewers/PdfViewer";
import { PptxViewer } from "@/components/viewers/PptxViewer";
import { TableViewer } from "@/components/viewers/TableViewer";
import { XlsxViewer } from "@/components/viewers/XlsxViewer";
import { BYTES_KEY } from "@/lib/bytes";
import { notionMarkdown } from "@/lib/notion";
import { readNotionNode } from "@/lib/notionNode";
import { expandTo, isHidden, loadExpanded, saveExpanded, toggle } from "@/lib/treeState";
import { readsText, viewerFor, type Viewer } from "@/lib/viewers";
import { S } from "@/strings";
import type { Entry } from "@/types";

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
    case "hwp":
      return <HwpViewer path={path} />;
    case "pptx":
      return <PptxViewer path={path} />;
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
        <TableViewer text={text} delimiter={viewer.delimiter} truncated={truncated} />
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
  hiddenFiles = false,
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
  /**
   * Offer a switch for dotfiles, and leave them out until it is on.
   *
   * For a tree over someone's own disk, where the convention still means what it always
   * meant: a home directory is mostly configuration, and a browser that opens onto `.ssh`
   * and `.DS_Store` has buried what the user came for. A connector's namespace carries no
   * such convention, so there is nothing to offer there.
   */
  hiddenFiles?: boolean;
}) {
  const [showHidden, setShowHidden] = useState(false);
  // The kind's own rules, the caller's, and the dotfile switch, as one predicate: a Notion
  // tree hides the json that is its body, the root hides the sources grafted into it, and a
  // tree could want any of them.
  const adapter: TreeAdapter | undefined = useMemo(() => {
    const base = kind === "notion" ? NOTION : undefined;
    const dotfiles = hiddenFiles && !showHidden;
    if (!hide && !dotfiles) return base;
    return {
      ...base,
      hide: (e: Entry) =>
        !!base?.hide?.(e) || (dotfiles && isHidden(e.name)) || !!hide?.(e),
    };
  }, [kind, hide, hiddenFiles, showHidden]);
  // Expansion lives here rather than inside the tree, and is remembered per root: the shape
  // someone left a directory in is worth more than one window's lifetime.
  const [expanded, setExpanded] = useState<Set<string>>(() =>
    loadExpanded(root),
  );
  const onToggle = (path: string) =>
    setExpanded((was) => {
      const next = toggle(was, path);
      saveExpanded(root, next);
      return next;
    });
  // Keyed by path, so the pane follows the selection and an unopened browser fetches
  // nothing. Reset when the root changes, because a path under the old root means nothing
  // under the new one — which `WorkspacePanel` gets by keying this component on its root.
  const [selected, setSelected] = useState<string | null>(null);
  // Selecting also opens the way down to what was selected, for a path that did not come
  // from the tree — a link inside a page. Following one should leave the reader looking at
  // where they landed, not at a tree still showing where they were.
  const openPath = (path: string) => {
    setSelected(path);
    setExpanded((was) => {
      const next = expandTo(was, root, path);
      if (next !== was) saveExpanded(root, next);
      return next;
    });
  };
  // What the registry says about the open file, before anything is fetched: a viewer that
  // opens the bytes itself does not want `fs_read`, which would read the whole container
  // only to decode it to null.
  const viewer =
    viewers && selected && kind !== "notion" ? viewerFor(selected) : null;
  const binary = viewer !== null && !readsText(viewer.kind);
  const fills = !!viewer?.fills;
  const file = useQuery({
    queryKey: ["file", selected, kind],
    queryFn: () =>
      kind === "notion" ? readNotionNode(selected!) : api.fsRead(selected!),
    enabled: !!selected && !binary,
  });

  // The way to ask again. A listing and a file answer are kept for a while (see `App`), which
  // is right for a store across a network and wrong the moment something changed it from the
  // outside — a page edited in Notion, a key written to a bucket. There is no event for that,
  // so there is a button.
  const qc = useQueryClient();
  const refresh = useMutation({
    // The engine first, then this window: dropping the queries here without telling the
    // stores would ask again and be told the same thing, from the render a listing is
    // served out of. That is what a page deleted in Notion looked like — a refresh that
    // could not reach the thing that was keeping it.
    mutationFn: api.workspaceRefresh,
    onSettled: () => {
      for (const key of [["fs"], ["file"], [BYTES_KEY]]) void qc.invalidateQueries({ queryKey: key });
    },
  });

  const text = file.data?.text ?? null;
  // Markdown for a Notion page, the raw bytes for everything else — including a Notion
  // database, whose json has no body to render and is more use shown as what it is.
  const body = text !== null && kind === "notion" ? notionMarkdown(text) : null;

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex w-72 shrink-0 flex-col border-t border-r">
        <div className="flex shrink-0 items-center justify-end gap-1 px-2 pt-1">
          {hiddenFiles && (
            <button
              className="rounded px-1.5 py-0.5 text-[11px] text-muted-foreground hover:bg-accent hover:text-foreground"
              aria-pressed={showHidden}
              onClick={() => setShowHidden((v) => !v)}
            >
              {showHidden ? S.hideHidden : S.showHidden}
            </button>
          )}
          <button
            className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label={S.refresh}
            title={S.refresh}
            onClick={() => refresh.mutate()}
            disabled={refresh.isPending}
          >
            <RotateCw className="size-3.5" />
          </button>
        </div>
        <ScrollArea className="min-h-0 flex-1 px-2 py-1">
          <FileTree
            path={root}
            onOpen={openPath}
            selected={selected}
            adapter={adapter}
            expanded={expanded}
            onToggle={onToggle}
          />
        </ScrollArea>
      </div>
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
              {/* Only when it was worked out rather than assumed. A file that had to be
                  inferred is one whose reader should be told which way it was read. */}
              {file.data?.encoding && file.data.encoding !== "UTF-8" && (
                <span className="shrink-0">· {file.data.encoding}</span>
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
              ) : body !== null && text !== null ? (
                <NotionViewer
                  key={selected}
                  path={selected}
                  text={text}
                  body={body}
                  truncated={!!file.data?.truncated}
                  onOpen={openPath}
                />
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
