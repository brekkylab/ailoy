// A picture.
//
// The bytes come over the bridge like every other viewer's, and the `<img>` is pointed at
// an object URL made from them.
//
// The URL is derived during render rather than set from an effect: it is a value computed
// from the bytes, and going through state would paint one frame without it for no reason.
// The effect is only the revoke — the URL has to outlive the render that made it and be
// released when the component goes, or the buffer is held for as long as the window lives.

import { useEffect, useMemo } from "react";

import { useBytes } from "@/lib/useBytes";
import { S } from "@/strings";

export function ImageViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const url = useMemo(
    () => (bytes.state === "ready" ? URL.createObjectURL(new Blob([bytes.bytes])) : null),
    [bytes],
  );

  useEffect(() => {
    if (!url) return;
    return () => URL.revokeObjectURL(url);
  }, [url]);

  if (bytes.state === "failed") return <p className="text-xs text-destructive">{S.viewerFailed}</p>;
  if (!url) return <p className="text-xs text-muted-foreground">{S.loading}</p>;
  return <img src={url} alt={path} className="max-w-full rounded-md border bg-muted/20" />;
}
