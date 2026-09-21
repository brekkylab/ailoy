// A tree over one root, beside whatever is open in it.
//
// Two panes rather than one stacked on the other: these roots are browsed, not skimmed,
// and a tree that keeps its width while the content changes is the shape every file
// explorer settles on. The tree scrolls on its own, so a deep directory does not push the
// document out of view and a long document does not shorten the tree.
//
// `render` is the only thing that differs between the sources. A Notion page is markdown
// and is worth reading as markdown; a file off a disk or a bucket is bytes, and showing
// those as anything but themselves would be a guess about what they are.

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import * as api from "@/api";
import { FileTree } from "@/components/FileTree";
import { Markdown } from "@/components/Markdown";
import { ScrollArea } from "@/components/ui/scroll-area";
import { S } from "@/strings";

export function FileBrowser({ root, render = "text" }: { root: string; render?: "text" | "markdown" }) {
  // Keyed by path, so the pane follows the selection and an unopened browser fetches
  // nothing. Reset when the root changes, because a path under the old root means nothing
  // under the new one — which `WorkspacePanel` gets by keying this component on its root.
  const [selected, setSelected] = useState<string | null>(null);
  const file = useQuery({
    queryKey: ["file", selected],
    queryFn: () => api.fsRead(selected!),
    enabled: !!selected,
  });

  return (
    <div className="flex min-h-0 flex-1">
      <ScrollArea className="w-72 shrink-0 border-t border-r px-2 py-1">
        <FileTree path={root} onOpen={setSelected} selected={selected} />
      </ScrollArea>
      <div className="min-w-0 flex-1 overflow-auto border-t">
        {!selected ? (
          <div className="grid h-full place-items-center p-6 text-sm text-muted-foreground">
            {S.pickFile}
          </div>
        ) : (
          <div className="p-4">
            <div className="mb-3 truncate font-mono text-xs text-muted-foreground" title={selected}>
              {selected}
            </div>
            {file.data?.text != null ? (
              render === "markdown" ? (
                <>
                  <Markdown text={file.data.text} />
                  {file.data.truncated && (
                    <p className="mt-2 text-xs text-muted-foreground">… {S.fileTooLarge}</p>
                  )}
                </>
              ) : (
                <pre className="whitespace-pre-wrap font-mono text-xs">
                  {file.data.text}
                  {file.data.truncated && `\n… ${S.fileTooLarge}`}
                </pre>
              )
            ) : file.data ? (
              <p className="text-xs text-muted-foreground">{S.binaryFile}</p>
            ) : null}
            {file.isError && <p className="text-xs text-destructive">{api.messageOf(file.error)}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
