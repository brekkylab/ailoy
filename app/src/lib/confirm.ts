import { confirm as tauriConfirm } from "@tauri-apps/plugin-dialog";

/**
 * WKWebView has no reliable `window.confirm`, so the Tauri dialog asks inside the app; a
 * plain browser tab (`npm run dev` without Tauri) falls back to the built-in one.
 */
export async function confirm(message: string, title: string): Promise<boolean> {
  if ("__TAURI_INTERNALS__" in window) return tauriConfirm(message, { title, kind: "warning" });
  return window.confirm(message);
}
