import { TriangleAlert } from "lucide-react";

/**
 * The one-line strip above the thread: a degraded workspace (warn) or a missing provider
 * key (error). `text` is always composed by the caller from `S`, never written here.
 */
export function Banner({ text, tone = "warn" }: { text: string; tone?: "warn" | "error" }) {
  return (
    <div
      className={
        tone === "error"
          ? "flex items-center gap-2 border-b bg-destructive/10 px-3 py-1.5 text-xs text-destructive"
          : "flex items-center gap-2 border-b bg-amber-500/10 px-3 py-1.5 text-xs text-amber-700 dark:text-amber-300"
      }
    >
      <TriangleAlert className="size-3.5 shrink-0" /> {text}
    </div>
  );
}
