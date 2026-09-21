// A spreadsheet, one sheet at a time.
//
// `exceljs` is imported lazily for the same reason `docx-preview` is: only a workbook needs
// it. It gives cell *values*, not the strings Excel would paint, so the formatting done
// here is deliberately shallow — a date reads as a date and a formula as its result, and
// anything else is shown as whatever it turned out to be rather than guessed at.
//
// Rows past the cap are not rendered: every cell becomes DOM, and a sheet is allowed to
// have a hundred thousand of them.

import { useEffect, useState } from "react";

import { useBytes } from "@/lib/useBytes";
import { S } from "@/strings";

/** Rows put in the DOM per sheet. */
const LIMIT = 500;

interface Sheet {
  name: string;
  rows: string[][];
  /** How many rows the sheet had beyond the ones kept. */
  hidden: number;
}

/** One cell as text. `exceljs` hands back whatever the cell holds, including objects. */
function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  if (typeof value === "object") {
    const v = value as Record<string, unknown>;
    // A formula cell carries its computed `result`; a rich-text cell carries runs; a
    // hyperlink carries the text it displays. Each is the readable half of the pair.
    if ("result" in v) return cellText(v.result);
    if ("text" in v) return cellText(v.text);
    if (Array.isArray(v.richText)) return v.richText.map((r) => cellText((r as { text?: unknown }).text)).join("");
    if ("error" in v) return String(v.error);
    return "";
  }
  return String(value);
}

export function XlsxViewer({ path }: { path: string }) {
  const bytes = useBytes(path);
  const [sheets, setSheets] = useState<Sheet[] | null>(null);
  const [active, setActive] = useState(0);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (bytes.state !== "ready") return;
    let live = true;
    // No reset here: `FileBrowser` keys this on the path, so another workbook is another
    // mount and the state below starts where it should.
    void import("exceljs")
      .then(async (mod) => {
        const wb = new mod.Workbook();
        await wb.xlsx.load(bytes.bytes);
        const read: Sheet[] = [];
        wb.eachSheet((ws) => {
          const rows: string[][] = [];
          ws.eachRow({ includeEmpty: true }, (row) => {
            if (rows.length >= LIMIT) return;
            const cells: string[] = [];
            row.eachCell({ includeEmpty: true }, (cell) => cells.push(cellText(cell.value)));
            rows.push(cells);
          });
          read.push({ name: ws.name, rows, hidden: Math.max(0, ws.rowCount - rows.length) });
        });
        if (live) setSheets(read);
      })
      .catch((err: unknown) => {
        console.warn(`${path} could not be read`, err);
        if (live) setFailed(true);
      });
    return () => {
      live = false;
    };
  }, [bytes, path]);

  if (bytes.state === "failed" || failed)
    return <p className="text-xs text-destructive">{S.viewerFailed}</p>;
  if (!sheets) return <p className="text-xs text-muted-foreground">{S.loading}</p>;
  if (sheets.length === 0) return <p className="text-xs text-muted-foreground">{S.empty}</p>;

  const sheet = sheets[Math.min(active, sheets.length - 1)];
  return (
    <div className="space-y-2">
      {sheets.length > 1 && (
        <div className="flex flex-wrap gap-1">
          {sheets.map((s, i) => (
            <button
              key={s.name}
              onClick={() => setActive(i)}
              aria-current={i === active ? "true" : undefined}
              className={`rounded px-2 py-0.5 text-xs ${i === active ? "bg-accent" : "hover:bg-accent"}`}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}
      <div className="overflow-x-auto rounded-md border">
        <table className="w-full border-collapse text-xs">
          <tbody>
            {sheet.rows.map((row, r) => (
              <tr key={r} className="even:bg-muted/20">
                {row.map((cell, c) => (
                  <td key={c} className="border-b px-2 py-1 align-top font-mono whitespace-nowrap">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sheet.hidden > 0 && (
        <p className="text-xs text-muted-foreground">
          {S.rowsHidden} {sheet.hidden}
        </p>
      )}
    </div>
  );
}
