// What a row in a file tree is drawn as: a folder, or a file by the kind of thing it is.
//
// The kind is `lib/viewers`' answer — the same registry that decides how the file opens —
// so a row's icon and what clicking it shows never disagree. Tinted, faintly, by family:
// enough to find the spreadsheet in a folder of documents at a glance, not enough to make
// a tree look like a legend.

import { cn } from "cn";
import {
  File,
  FileArchive,
  FileCode,
  FileImage,
  FileSpreadsheet,
  FileText,
  FileType,
  Folder,
  FolderOpen,
  Presentation,
  type LucideIcon,
} from "lucide-react";

import { viewerFor, type ViewerKind } from "@/lib/viewers";
import type { Entry } from "@/types";

const BY_KIND: Partial<Record<ViewerKind, [LucideIcon, string]>> = {
  markdown: [FileText, "text-muted-foreground"],
  text: [FileText, "text-muted-foreground"],
  code: [FileCode, "text-sky-500/80"],
  table: [FileSpreadsheet, "text-emerald-600/80 dark:text-emerald-500/80"],
  xlsx: [FileSpreadsheet, "text-emerald-600/80 dark:text-emerald-500/80"],
  image: [FileImage, "text-violet-500/80"],
  pdf: [FileType, "text-red-500/80"],
  docx: [FileText, "text-blue-500/80"],
  hwp: [FileText, "text-blue-500/80"],
  pptx: [Presentation, "text-orange-500/80"],
};

/** Archives are not something the browser opens, so the registry has no word for them. */
const ARCHIVE = /\.(zip|tar|tgz|gz|bz2|xz|7z|rar)$/i;

export function EntryIcon({ entry, open }: { entry: Entry; open?: boolean }) {
  const cls = "size-3.5 shrink-0";
  if (entry.kind === "dir") {
    const Icon = open ? FolderOpen : Folder;
    return <Icon className={cn(cls, "text-muted-foreground")} />;
  }
  if (ARCHIVE.test(entry.name)) return <FileArchive className={cn(cls, "text-amber-600/80")} />;
  const found = BY_KIND[viewerFor(entry.path)?.kind as ViewerKind];
  const [Icon, tint] = found ?? [File, "text-muted-foreground"];
  return <Icon className={cn(cls, tint)} />;
}
