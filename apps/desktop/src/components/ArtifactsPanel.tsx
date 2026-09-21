// The artifacts view: what the agent has produced, as a tree.
//
// This is the same directory the workspace shows at `/artifacts` — the engine grafts the
// artifacts root in there so the user meets the agent's output inside their own files.
// Giving it its own entry in the sidebar is the other half of that: what the agent made
// is the thing a user comes back for, and having to remember which folder it is under is
// exactly the friction the graft was meant to remove.

import { FileBrowser } from "@/components/FileBrowser";
import { S } from "@/strings";

/**
 * Where the engine grafts the artifacts tree into the workspace. Must match
 * `ARTIFACTS_PATH` in `apps/desktop/core/src/workspace/manager.rs`: the engine also
 * refuses to detach this path, so the two are a pair and a rename has to move together.
 */
const ARTIFACTS_ROOT = "/artifacts";

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
