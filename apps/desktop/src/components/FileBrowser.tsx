// A tree over one root, with a read-only preview of whatever is open in it.
//
// Two of the main panel's views are this: the workspace, rooted at `/`, and the artifacts
// list, rooted where the agent's output lands. They differ in what sits above the tree,
// not in the tree, so the tree and the preview live here and each panel brings its own
// header.
//
// The preview is capped rather than sized to its content, so a long file cannot push the
// tree off the bottom of the panel.

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import * as api from "@/api";
import { FileTree } from "@/components/FileTree";
import { ScrollArea } from "@/components/ui/scroll-area";
import { S } from "@/strings";

export function FileBrowser({ root }: { root: string }) {
  // Keyed by path, so the preview follows the selection and an unopened browser fetches
  // nothing. Reset when the root changes, because a path under the old root means nothing
  // under the new one — React does that for us as long as the two panels stay separate
  // components, which is why this is not one component with a switching `root`.
  const [selected, setSelected] = useState<string | null>(null);
  const file = useQuery({
    queryKey: ["file", selected],
    queryFn: () => api.fsRead(selected!),
    enabled: !!selected,
  });

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ScrollArea className="min-h-0 flex-1 border-t px-2 py-1">
        <FileTree path={root} onOpen={setSelected} selected={selected} />
      </ScrollArea>
      {selected && (
        <div className="max-h-[40%] shrink-0 overflow-auto border-t p-2">
          <div className="mb-1 truncate font-mono text-xs text-muted-foreground">{selected}</div>
          {file.data?.text != null ? (
            <pre className="whitespace-pre-wrap font-mono text-xs">
              {file.data.text}
              {file.data.truncated && `\n… ${S.fileTooLarge}`}
            </pre>
          ) : file.data ? (
            <p className="text-xs text-muted-foreground">{S.binaryFile}</p>
          ) : null}
          {file.isError && <p className="text-xs text-destructive">{api.messageOf(file.error)}</p>}
        </div>
      )}
    </div>
  );
}
