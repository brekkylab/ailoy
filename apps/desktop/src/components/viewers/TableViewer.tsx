// A delimited file as the table it is.
//
// The first row is treated as a header. That is a guess, and it is the right one often
// enough to be worth making: a file without one loses nothing but a bold row, while a file
// with one gains column names that stay put while the body scrolls.
//
// Rows past `LIMIT` are not rendered. The engine already caps what it reads, but a
// megabyte of CSV is still tens of thousands of records and each one becomes DOM; the cap
// is on the laying out rather than on the reading, and the pane says how much it left.

import { useMemo } from "react";

import { parseDelimited } from "@/lib/delimited";
import { S } from "@/strings";

/** How many body rows are put in the DOM. */
const LIMIT = 500;

export function TableViewer({
  text,
  delimiter,
  truncated,
}: {
  text: string;
  delimiter: string;
  /** The engine cut the file short, so the last row it handed over may be half of one. */
  truncated: boolean;
}) {
  const rows = useMemo(() => {
    const parsed = parseDelimited(text, delimiter);
    // A cut file ends mid-record, and a half row rendered as a row is a row that lies
    // about what is in the file.
    return truncated && parsed.length > 1 ? parsed.slice(0, -1) : parsed;
  }, [text, delimiter, truncated]);

  if (rows.length === 0) return <p className="text-xs text-muted-foreground">{S.empty}</p>;

  const [head, ...body] = rows;
  const shown = body.slice(0, LIMIT);
  const hidden = body.length - shown.length;

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full border-collapse text-xs">
          <thead className="bg-muted/50">
            <tr>
              {head.map((cell, i) => (
                <th key={i} className="border-b px-2 py-1 text-left font-medium whitespace-nowrap">
                  {cell}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, r) => (
              <tr key={r} className="even:bg-muted/20">
                {/* Indexed against the header so a short row still lines up under the
                    right columns, and a long one does not silently lose its tail. */}
                {Array.from({ length: Math.max(head.length, row.length) }, (_, c) => (
                  <td key={c} className="border-b px-2 py-1 align-top font-mono">
                    {row[c] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(hidden > 0 || truncated) && (
        <p className="text-xs text-muted-foreground">
          {hidden > 0 ? `${S.rowsHidden} ${hidden}` : S.fileTooLarge}
        </p>
      )}
    </div>
  );
}
