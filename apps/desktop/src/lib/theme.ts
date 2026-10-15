// Which palette the window is in.
//
// A window preference, like the sidebar's collapsed state and unlike anything in
// `settings_get`: it belongs to this machine's webview, it is wanted before the first IPC
// call resolves, and an engine round trip for it would only be a slower way to paint.
//
// `data-theme` on the root element is the whole mechanism, and `index.css` is written
// around it: absent means follow the OS, `light`/`dark` pin the `color-scheme` the
// `light-dark()` pairs resolve against. `useSystemTheme` in `App.tsx` watches the attribute
// and keeps Tailwind's `.dark` class on the same side, so setting it moves both halves.

export type Theme = "system" | "light" | "dark";

export const THEMES: Theme[] = ["system", "light", "dark"];

const KEY = "ailoy.theme";

/**
 * A stored value as a theme. Anything else — nothing stored yet, a key another build
 * wrote, a value hand-edited in devtools — reads as "follow the OS", which is the state
 * the app is in before anyone chooses.
 */
export function parseTheme(raw: string | null | undefined): Theme {
  return raw === "light" || raw === "dark" ? raw : "system";
}

export function readTheme(): Theme {
  try {
    return parseTheme(localStorage.getItem(KEY));
  } catch {
    return "system";
  }
}

/** Paint in `theme`, and remember it. */
export function setTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, theme);
  } catch {
    /* storage may be unavailable; the window is already in the right palette */
  }
}

/**
 * Put the window in the stored palette. Called before the first render — a theme applied
 * from inside a component is a frame of the wrong palette first.
 */
export function applyStoredTheme() {
  setTheme(readTheme());
}
