// The paths the engine grafts things into the workspace at.
//
// They are the engine's decisions, mirrored here because the webview has to name them to
// show them — or, in the root's case, to leave them out.

/** The whole workspace: the user's own files with everything else grafted into it. */
export const WORKSPACE_ROOT = "/";

/**
 * Where the agent's output lands. Must match `ARTIFACTS_PATH` in
 * `apps/desktop/core/src/workspace/manager.rs`, which also refuses to detach it.
 */
export const ARTIFACTS_ROOT = "/artifacts";
