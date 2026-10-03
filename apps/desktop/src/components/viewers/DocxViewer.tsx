// A Word document, laid out by `docx-preview`.
//
// It renders into a DOM node it is handed rather than returning markup, so this is one of
// the few places the app holds a ref and lets a library write inside it. React never
// touches those children: the container is empty as far as the tree is concerned, and the
// effect clears it before each render so a second document does not land under the first.
//
// The module is imported lazily. It is the largest dependency in the app and only a `.docx`
// needs it, so it stays out of the bundle everything else pays for.

import { useEffect, useRef, useState } from "react";

import { useBytes } from "@/lib/useBytes";
import { report } from "@/lib/report";
import { S } from "@/strings";

export function DocxViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const host = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (bytes.state !== "ready" || !host.current) return;
    const into = host.current;
    let live = true;
    // Cleared rather than reset: this is the library's DOM, not React's, and `FileBrowser`
    // keys this component on the path so `failed` starts false on every new document.
    into.replaceChildren();
    void import("docx-preview")
      .then((docx) =>
        docx.renderAsync(bytes.bytes, into, undefined, {
          // The document's own page furniture is for paper. Inline what it says and drop
          // the rest, so it reads as a document in a pane rather than as a scan of one.
          inWrapper: false,
          ignoreWidth: true,
          ignoreHeight: true,
        }),
      )
      .catch((err: unknown) => {
        // What a reader says about a malformed container names parts of a file format,
        // which is not what belongs on screen. It goes to the console; the pane says the
        // one thing that is actionable.
        report(`${path} could not be read as a document`, err);
        if (live) setFailed(true);
      });
    return () => {
      live = false;
      into.replaceChildren();
    };
  }, [bytes, path]);

  if (bytes.state === "failed" || failed)
    return <p className="text-xs text-destructive">{S.viewerFailed}</p>;
  return (
    <>
      {bytes.state === "loading" && <p className="text-xs text-muted-foreground">{S.loading}</p>}
      {/* No `dark:prose-invert`: the page below is paper in either theme — see
          `.docx-host` in `index.css` — so inverting for the dark one would put light text
          on white for every element Word's own stylesheet leaves to us. */}
      <div ref={host} className="docx-host prose prose-sm max-w-none" />
    </>
  );
}
