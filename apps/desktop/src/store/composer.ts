// Text for the composer from somewhere else in the window.
//
// The composer owns what is typed; this is how a file's "Ask about this" puts a mention of
// the file into it without the file panel reaching into the composer's state. The composer
// subscribes and takes each insert once — see `Composer`.

import { create } from "zustand";

interface ComposerInbox {
  /** The latest text to add, numbered so the same text asked for twice is two inserts. */
  pending: { text: string; n: number } | null;
  insert: (text: string) => void;
  take: () => void;
}

export const useComposerInbox = create<ComposerInbox>((set) => ({
  pending: null,
  insert: (text) => set((s) => ({ pending: { text, n: (s.pending?.n ?? 0) + 1 } })),
  take: () => set({ pending: null }),
}));
