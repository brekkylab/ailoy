// The artifacts view: what the agent has produced, as a tree.
//
// The engine grafts the artifacts root into the workspace at `/artifacts`, so the agent
// writes there by an ordinary path and reads it back the same way. This is that directory,
// given its own entry rather than left to be found: what the agent made is the thing a
// user comes back for, and remembering which folder it is under is friction.
//
// Which is also why `WorkspacePanel` leaves it out of My Computer. It is in the tree the
// agent sees; it is not one of the user's own files, and it is already here.

import { FileBrowser } from "@/components/FileBrowser";
import { ARTIFACTS_ROOT } from "@/paths";
import { S } from "@/strings";

export function ArtifactsPanel() {
  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="px-6 pt-3 pb-4">
        <h1 className="text-xl font-semibold">{S.artifacts}</h1>
      </div>
      <FileBrowser root={ARTIFACTS_ROOT} />
    </section>
  );
}
