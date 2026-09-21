// A PDF, in the viewer the platform already has.
//
// `<object>` rather than a bundled renderer: WebKit displays PDFs natively, with selection,
// search and printing, and none of that is worth reimplementing. It is also why this is the
// one viewer with no size cap — the document streams from the scheme and is never resident
// in the webview's heap.
//
// The fallback inside the element is what shows when the platform declines to render it,
// which is the only failure this can detect: an `<object>` reports no error event.

import { wsfileUrl } from "@/lib/wsfile";
import { S } from "@/strings";

export function PdfViewer({ path }: { path: string }) {
  return (
    <object data={wsfileUrl(path)} type="application/pdf" className="h-[75vh] w-full rounded-md border">
      <p className="p-4 text-xs text-muted-foreground">{S.viewerFailed}</p>
    </object>
  );
}
