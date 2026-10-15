// A delimited file, read the way a spreadsheet reads one.
//
// A row-number gutter that stays put while the sheet scrolls sideways, a header that
// stays put while it scrolls down, figures right-aligned in tabular figures, and an empty
// cell marked as empty rather than left looking like a rendering slip. The numbers in the
// gutter are the file's own, so a reader can point at a line in the source: the header
// record is line 1.
//
// The first record is treated as a header. That is a guess, and the right one often
// enough to be worth making: a file without one loses nothing but a bold row, while a file
// with one gains column names that stay in view.
//
// Rows are put in the DOM a page at a time. The engine already caps what it reads, but a
// megabyte of CSV is still tens of thousands of records and each one becomes DOM.

import { useMemo, useState } from "react";

import { cn } from "cn";

import { isNumericColumn, parseTable } from "@/lib/delimited";
import { S } from "@/strings";

/** How many body rows are added at a time. */
const PAGE = 500;

/** What to call the character a file turned out to be separated on. */
const DELIMITER_NAMES: Record<string, string> = {
  ",": "comma",
  ";": "semicolon",
  "\t": "tab",
  "|": "pipe",
};

export function TableViewer({
  text,
  delimiter,
  truncated,
}: {
  text: string;
  /** What the extension already knows, where it knows it. Sniffed when it does not. */
  delimiter?: string;
  /** The engine cut the file short, so the last row it handed over may be half of one. */
  truncated: boolean;
}) {
  const table = useMemo(() => {
    const parsed = parseTable(text, delimiter);
    // A cut file ends mid-record, and half a row rendered as a row is a row that lies
    // about what is in the file.
    if (!truncated || parsed.rows.length === 0) return parsed;
    return { ...parsed, rows: parsed.rows.slice(0, -1) };
  }, [text, delimiter, truncated]);

  const numeric = useMemo(
    () => Array.from({ length: table.columns }, (_, i) => isNumericColumn(table.rows, i)),
    [table],
  );

  const [shown, setShown] = useState(PAGE);
  const visible = table.rows.slice(0, shown);
  const remaining = table.rows.length - visible.length;

  if (table.columns === 0) return <p className="text-xs text-muted-foreground">{S.empty}</p>;

  const cell = "max-w-[42ch] truncate border-r border-b px-3 py-1 text-left align-top";
  const gutter =
    "sticky left-0 z-10 border-r border-b bg-muted px-2 py-1 text-right font-mono text-[11px] font-normal text-muted-foreground select-none";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="border-separate border-spacing-0 text-xs whitespace-pre">
          <thead>
            <tr>
              <th scope="col" className={cn(gutter, "top-0 z-20")}>
                <span className="sr-only">{S.row}</span>
              </th>
              {table.header.map((head, i) => (
                <th
                  key={i}
                  scope="col"
                  className={cn(
                    cell,
                    "sticky top-0 z-10 bg-muted font-medium",
                    numeric[i] && "text-right",
                  )}
                >
                  {head}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, r) => (
              <tr key={r} className="group">
                <th scope="row" className={cn(gutter, "group-hover:bg-accent")}>
                  {r + 2}
                </th>
                {row.map((value, c) => (
                  <td
                    key={c}
                    className={cn(
                      cell,
                      "group-hover:bg-accent",
                      numeric[c] && "text-right tabular-nums",
                      // An empty cell reads as an empty cell rather than as something
                      // that failed to render.
                      value === "" && "before:text-muted-foreground before:content-['–']",
                    )}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex shrink-0 items-center justify-between gap-3 border-t px-3 py-1.5 text-xs text-muted-foreground">
        <span>
          {table.rows.length.toLocaleString()} {S.rows} · {table.columns} {S.columns} ·{" "}
          {DELIMITER_NAMES[table.delimiter] ?? S.delimiter}
          {truncated && ` · ${S.fileTooLarge}`}
        </span>
        {remaining > 0 && (
          <button
            className="rounded border px-2 py-0.5 text-foreground hover:bg-accent"
            onClick={() => setShown((n) => n + PAGE)}
          >
            {S.showMore} ({remaining.toLocaleString()})
          </button>
        )}
      </div>
    </div>
  );
}
