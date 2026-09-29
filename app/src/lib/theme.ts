export type Theme = "system" | "light" | "dark";

const KEY = "ailoy.theme";

export function readTheme(): Theme {
  try {
    const raw = localStorage.getItem(KEY);
    return raw === "light" || raw === "dark" ? raw : "system";
  } catch {
    return "system";
  }
}

/** Paint in `theme`, and remember it. `system` leaves `data-theme` unset so the OS decides. */
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

export function applyStoredTheme() {
  setTheme(readTheme());
}

/**
 * Keeps Tailwind's `.dark` class on the same side as the palette, for the few `dark:`
 * variants. `data-theme` wins where set; otherwise it follows the OS.
 */
export function syncDarkClass(): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const root = document.documentElement;
  const apply = () => {
    const forced = root.dataset.theme;
    root.classList.toggle("dark", forced ? forced === "dark" : mq.matches);
  };
  apply();
  mq.addEventListener("change", apply);
  const observer = new MutationObserver(apply);
  observer.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  return () => {
    mq.removeEventListener("change", apply);
    observer.disconnect();
  };
}
