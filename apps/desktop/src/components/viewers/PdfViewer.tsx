// A PDF, in the viewer the platform already has.
//
// WebKit displays PDFs natively, with selection, search and printing, and none of that is
// worth reimplementing. It is also why this viewer takes the address rather than the bytes:
// the document streams, so a 200 MB report costs what its first page costs.
//
// The grey around a page is this element's own background showing through the letterboxing
// the viewer leaves when the page is narrower than the pane — not something the plugin
// paints, which is why setting it here reaches it. `bg-background` makes that band the
// window's own colour, so a document sits on the panel rather than in a tray on it.
//
// `#view=FitH` opens at page width, which is the reading width for a document. It is a
// hint: each platform's built-in viewer honours what it honours.
//
// An `<iframe>` rather than an `<object>`, which is what costs this viewer an inline
// message when a file cannot be fetched — an iframe reports no error for a cross-document
// load. The engine answers a missing or oversized file with a 404, and the native viewer
// shows its own empty state for it. No `sandbox`: the built-in viewer is itself a scripted
// document and an empty sandbox blanks it, while what it renders is a PDF rather than a
// page, so it cannot script this origin.

import { wsfileUrl } from "@/lib/wsfile";

export function PdfViewer({ path }: { path: string }) {
  return (
    <iframe
      title={path}
      src={`${wsfileUrl(path)}#view=FitH`}
      className="block h-full w-full border-0 bg-background"
    />
  );
}
