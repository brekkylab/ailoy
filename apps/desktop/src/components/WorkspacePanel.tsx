// The workspace view: the tree over everything connected to it.
//
// The sources themselves — what is mounted, adding one, disconnecting one — are in the
// sidebar beside this, because they are navigation rather than content. That leaves the
// whole panel to the files, which is what someone opening the workspace came for.

import { FileBrowser } from "@/components/FileBrowser";
import { S } from "@/strings";

export function WorkspacePanel() {
  return (
    // `min-h-0 flex-1` rather than `h-full`: the banners above this in `main` are part of
    // the same column, and a full-height panel would push itself off the bottom by theirs.
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="px-6 pt-3 pb-4">
        <h1 className="text-xl font-semibold">{S.workspace}</h1>
      </div>
      <FileBrowser root="/" />
    </section>
  );
}
