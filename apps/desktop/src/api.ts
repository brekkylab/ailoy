// Every call into Rust, in one place and typed once.
//
// Components never `invoke` directly: a command name or argument that drifts should
// break here, at one typed edge, and not in a dozen call sites. Command names are
// snake_case, their arguments camelCase — Tauri converts to the Rust snake_case.
//
// Every one of these rejects with `T.EngineErrorPayload`; use `messageOf`/`kindOf` on
// whatever is caught.

import { Channel, invoke } from "@tauri-apps/api/core";

import type * as T from "./types";

export const sessionList = () => invoke<T.SessionSummary[]>("session_list");
export const sessionCreate = (model?: string) => invoke<T.SessionSummary>("session_create", { model: model ?? null });
export const sessionRename = (id: string, title: string) => invoke<void>("session_rename", { id, title });
export const sessionSetModel = (id: string, model: string) => invoke<void>("session_set_model", { id, model });
export const sessionDelete = (id: string) => invoke<void>("session_delete", { id });
export const messageList = (sessionId: string) => invoke<T.StoredMessage[]>("message_list", { sessionId });
export const sessionUsage = (sessionId: string) => invoke<T.SessionUsage>("session_usage", { sessionId });

/** Resolves to the new run's id. Rejects with `already_running` if one is in flight. */
export const runStart = (sessionId: string, text: string, onEvent: Channel<T.RunEvent>) =>
  invoke<string>("run_start", { sessionId, text, onEvent });
/**
 * Re-subscribes to a run still in flight, or resolves to `null` when there is none.
 * The channel receives a synthesized `started`, then the whole buffered text as one
 * `text_delta`, before live events resume.
 */
export const runAttach = (sessionId: string, onEvent: Channel<T.RunEvent>) =>
  invoke<string | null>("run_attach", { sessionId, onEvent });
/** `false` when nothing was running. */
export const runCancel = (sessionId: string) => invoke<boolean>("run_cancel", { sessionId });

export const workspaceInfo = () => invoke<T.WorkspaceInfo>("workspace_info");
/**
 * A file's bytes, for the viewers that open a format. The command answers with
 * `tauri::ipc::Response`, so this resolves to the buffer itself rather than to a number
 * per byte.
 */
export const fsReadBytes = (path: string) => invoke<ArrayBuffer>("fs_read_bytes", { path });

/** Repoint the workspace root at another directory on this machine. */
export const workspaceSetRoot = (path: string) =>
  invoke<T.WorkspaceInfo>("workspace_set_root", { path });
export const fsList = (path: string) => invoke<T.Entry[]>("fs_list", { path });
export const fsRead = (path: string) => invoke<T.FileContent>("fs_read", { path });
export const fsWrite = (path: string, text: string) => invoke<void>("fs_write", { path, text });
export const fsMkdir = (path: string) => invoke<void>("fs_mkdir", { path });
export const fsDelete = (path: string) => invoke<void>("fs_delete", { path });
export const fsRename = (from: string, to: string) => invoke<void>("fs_rename", { from, to });
export const fsImport = (dest: string, sources: string[]) => invoke<T.ImportReport>("fs_import", { dest, sources });
export const mountList = () => invoke<T.MountInfo[]>("mount_list");
export const mountAdd = (req: T.MountRequest) => invoke<T.MountInfo>("mount_add", { req });
export const mountRemove = (path: string) => invoke<void>("mount_remove", { path });

export const settingsGet = () => invoke<T.Settings>("settings_get");
export const settingsSet = (patch: T.SettingsPatch) => invoke<T.Settings>("settings_set", { patch });
export const modelsList = () => invoke<T.ModelInfo[]>("models_list");
export const openLogs = () => invoke<void>("open_logs");

export { kindOf, messageOf } from "./lib/errors";
