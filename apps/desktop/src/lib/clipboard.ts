// Putting text on the clipboard from the webview.

/**
 * Copy `text`, resolving to whether it worked.
 *
 * The async Clipboard API first — the webview is a secure context, so it is there — and the
 * old `execCommand` through a hidden textarea when it is refused, which a webview can still do
 * for a page it has not granted clipboard access to.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      area.remove();
    }
  }
}
