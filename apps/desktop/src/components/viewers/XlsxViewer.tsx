// A spreadsheet, one sheet at a time.
//
// `exceljs` is imported lazily for the same reason `docx-preview` is: only a workbook needs
// it. It gives cell *values*, not the strings Excel would paint, so the formatting done
// here is deliberately shallow — a date reads as a date and a formula as its result, and
// anything else is shown as whatever it turned out to be rather than guessed at.
//
// Rows past the cap are not rendered: every cell becomes DOM, and a sheet is allowed to
// have a hundred thousand of them.
//
// A workbook exceljs refuses outright gets one more try with its drawings taken out. See
// `withoutDrawings` for why, and for why that loses nothing this viewer was showing.

import { useEffect, useState } from "react";

import { useBytes } from "@/lib/useBytes";
import { report } from "@/lib/report";
import { S } from "@/strings";

/** Rows put in the DOM per sheet. */
const LIMIT = 500;

/** A worksheet's pointer at its drawing. Empty by schema, but closed either way. */
const DRAWING_REF = /<drawing\b[^>]*\/>|<drawing\b[^>]*>[\s\S]*?<\/drawing>/g;

/**
 * The same workbook with every drawing removed — the parts and the worksheets' pointers at
 * them.
 *
 * For a workbook exceljs will not open. It matches a drawing's root element as the literal
 * tag `xdr:wsDr`, so a file that declares that namespace as the default and writes `<wsDr>`
 * — valid XML, and what more than one generator emits — parses to nothing, and then two
 * separate places read `.anchors` off the nothing without checking. One is reached by the
 * worksheet's `<drawing/>` element alone, which is why the parts have to go *and* the
 * pointers.
 *
 * Nothing is lost here: a drawing is a chart or an image, and this viewer has only ever
 * shown cells. It is done on the failure and not on every workbook, because rewriting the
 * archive costs a pass over the whole of it.
 */
async function withoutDrawings(bytes: ArrayBuffer): Promise<ArrayBuffer> {
  const JSZip = (await import("jszip")).default;
  const zip = await JSZip.loadAsync(bytes);
  for (const name of Object.keys(zip.files)) {
    if (name.startsWith("xl/drawings/")) zip.remove(name);
  }
  for (const name of Object.keys(zip.files)) {
    if (!/^xl\/worksheets\/[^/]+\.xml$/.test(name)) continue;
    const xml = await zip.files[name].async("string");
    zip.file(name, xml.replace(DRAWING_REF, ""));
  }
  return zip.generateAsync({ type: "arraybuffer" });
}

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
        let wb = new mod.Workbook();
        try {
          await wb.xlsx.load(bytes.bytes);
        } catch (err) {
          report(`${path} has a drawing exceljs cannot read; opening it without`, err);
          // A fresh workbook rather than the one that threw: it stopped part-way through
          // loading, and what it holds is whatever it had got to.
          wb = new mod.Workbook();
          await wb.xlsx.load(await withoutDrawings(bytes.bytes));
        }
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
        report(`${path} could not be read as a workbook`, err);
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
