// The contexts the backend keeps under `~/.cache/ailoy/contexts` (see `src-tauri/src/context.rs`).
// Outside Tauri — a plain `vite dev` in the browser — there is no backend, and the list stays empty.

import { invoke } from "@tauri-apps/api/core";

export interface Context {
  id: string;
  name: string;
  /** The tree on disk, absolute. */
  dir: string;
  /** The context that is always there, made at startup, which nothing may delete. */
  default: boolean;
}

const inTauri = () => "__TAURI_INTERNALS__" in window;

class Contexts {
  list = $state<Context[]>([]);

  async refresh() {
    if (!inTauri()) return;
    try {
      this.list = await invoke<Context[]>("list_contexts");
    } catch (err) {
      console.warn("could not list contexts", err);
    }
  }

  /** Creates an empty context and returns it, already in `list`. */
  async create(name: string): Promise<Context | null> {
    if (!inTauri()) return null;
    try {
      const made = await invoke<Context>("create_context", { name });
      await this.refresh();
      return made;
    } catch (err) {
      console.warn("could not create a context", err);
      return null;
    }
  }
}

export const contexts = new Contexts();

/** One entry of a directory in a context's tree (see `FileEntry` in `context.rs`). */
export interface FileEntry {
  name: string;
  kind: "folder" | "file" | "link";
  size: number;
  modified: number | null;
}

/** A path under a context's tree, one name per segment; `[]` is the tree itself. */
export type Segments = string[];

export const listFiles = (id: string, path: Segments) => invoke<FileEntry[]>("list_context_files", { id, path });
export const makeDir = (id: string, path: Segments) => invoke<void>("make_context_dir", { id, path });
/** Copies files or folders from anywhere on this machine into `path`. */
export const addFiles = (id: string, path: Segments, sources: string[]) =>
  invoke<void>("add_context_files", { id, path, sources });
/** Writes the whole tree to `to`, a `.tar.gz` outside the cache. */
export const exportContext = (id: string, to: string) => invoke<void>("export_context", { id, to });
export const removeFile = (id: string, path: Segments) => invoke<void>("remove_context_file", { id, path });
