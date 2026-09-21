// A picture, loaded by the webview from the app's own scheme.
//
// The bytes never pass through the app: the URL goes into an `<img>` and the webview's own
// loader does the fetching and decoding, which is what it is built for and what keeps a
// large photograph off the JavaScript heap.

import { useState } from "react";

import { wsfileUrl } from "@/lib/wsfile";
import { S } from "@/strings";

export function ImageViewer({ path }: { path: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <p className="text-xs text-destructive">{S.viewerFailed}</p>;
  return (
    <img
      src={wsfileUrl(path)}
      alt={path}
      onError={() => setFailed(true)}
      className="max-w-full rounded-md border bg-muted/20"
    />
  );
}
