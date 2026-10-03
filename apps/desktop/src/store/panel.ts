// The file panel beside the thread: whether it is open, and what it is showing.
//
// A store rather than state in `App`, because the things that open it are deep in the
// thread — a tool call's row knows which file it wrote, and threading a callback down to
// every one of them would be most of the change. Whether it is open survives a relaunch;
// what it was showing does not, the way the workspace view's selection does not.

import { create } from "zustand";

import type { Place } from "@/lib/workspacePath";

const KEY = "ailoy.filePanel";

function storedOpen(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

function remember(open: boolean) {
  try {
    localStorage.setItem(KEY, open ? "1" : "0");
  } catch {
    /* storage may be unavailable */
  }
}

export interface PanelState {
  open: boolean;
  tab: "files" | "artifacts";
  /** The source under Files — a mount path — or `null` for My Computer. */
  source: string | null;
  /** The file open in the panel, as a workspace path. */
  file: string | null;
  toggle: () => void;
  close: () => void;
  setTab: (tab: PanelState["tab"]) => void;
  setSource: (source: string | null) => void;
  setFile: (file: string | null) => void;
  /** Open the panel on a file, wherever it lives. */
  show: (place: Place) => void;
}

export const usePanel = create<PanelState>((set) => ({
  open: storedOpen(),
  tab: "files",
  source: null,
  file: null,
  toggle: () =>
    set((s) => {
      remember(!s.open);
      return { open: !s.open };
    }),
  close: () => {
    remember(false);
    set({ open: false });
  },
  setTab: (tab) => set({ tab, file: null }),
  // Another source is another tree: what was open under the last one names nothing here.
  setSource: (source) => set({ source, file: null }),
  setFile: (file) => set({ file }),
  show: (place) => {
    remember(true);
    set({ open: true, tab: place.tab, source: place.source, file: place.path });
  },
}));
