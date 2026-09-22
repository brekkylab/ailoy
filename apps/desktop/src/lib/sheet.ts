// A worksheet, flattened into what a viewer draws.
//
// The reading is done elsewhere; this is the step after it. A sheet is a sparse map of
// cells addressed by row and column, and a table is a dense grid of rows containing
// cells. Merges are the whole difficulty: in the file a merge is a range with one cell
// holding the content, and in a table it is a `colspan` on that cell and the absence of
// every cell it covers.
//
// Nothing here produces markup. Text comes back as strings the viewer puts in text nodes,
// and the styles come back as a fixed set of fields — a colour, a weight, a border —
// rather than as CSS written here. A file off someone's disk cannot reach the DOM as
// anything but text.

import type { Cell as XCell, Row as XRow, Worksheet } from "exceljs";

/**
 * Past this the sheet is cut off and `truncated` says so. A form runs to tens of rows;
 * something with a hundred thousand is an export, and the whole of it as table rows is a
 * window that stops responding.
 */
const MAX_ROWS = 5000;
const MAX_COLUMNS = 128;

/**
 * Excel measures column widths in characters of the default font and row heights in
 * points. Neither is a CSS unit, and these are the conversions its own file-format notes
 * document: a character is 7 pixels plus 5 of cell padding, and a point is 96/72 of one.
 */
const PX_PER_CHAR = 7;
const CELL_PADDING_PX = 5;
const PX_PER_POINT = 96 / 72;

/** What a column is given when the file sets none. */
const DEFAULT_COLUMN_CHARS = 8.43;
/** What a row is given when neither it nor the sheet sets one, in points. */
const DEFAULT_ROW_POINTS = 15;

export type Align = "left" | "center" | "right";
export type VAlign = "top" | "middle" | "bottom";

/** One side of a cell's border, reduced to the three things CSS needs. */
export interface Border {
  width: number;
  style: "solid" | "dashed" | "dotted" | "double";
  color: string;
}

export interface Style {
  fill: string | null;
  color: string | null;
  bold: boolean;
  italic: boolean;
  underline: boolean;
  /** In pixels, or null to inherit the table's. */
  size: number | null;
  family: string | null;
  align: Align | null;
  valign: VAlign | null;
  wrap: boolean;
  /**
   * Absent sides are null, which is not a border of width zero: a neighbour's border
   * still shows through where this cell declares none.
   */
  borders: {
    top: Border | null;
    right: Border | null;
    bottom: Border | null;
    left: Border | null;
  };
}

export interface Cell {
  text: string;
  colSpan: number;
  rowSpan: number;
  style: Style;
}

export interface Row {
  /**
   * In pixels. A merged cell can make a row taller than this, which is why the viewer
   * treats it as a minimum rather than as a fixed height.
   */
  height: number;
  cells: Cell[];
}

export interface Sheet {
  name: string;
  /** One per column, in pixels. */
  widths: number[];
  rows: Row[];
  /** Set when the sheet was larger than the caps above. */
  truncated: boolean;
}

// -- Colour ------------------------------------------------------------------
//
// A colour in the file is `AARRGGBB`, and the alpha byte is not one: Excel writes `00` on
// colours that are plainly opaque, so reading it as alpha turns a grey header row
// invisible. The last six digits are the colour; the first two are dropped.
//
// The other two kinds a colour can be, `theme` and `indexed`, are palette references that
// need the workbook's theme to resolve. They are not resolved here, and a cell that uses
// one renders with no fill rather than with a guess.

interface ArgbColor {
  argb?: string;
  theme?: number;
  indexed?: number;
}

function cssColor(color: ArgbColor | undefined): string | null {
  const argb = color?.argb;
  if (typeof argb !== "string" || !/^[0-9a-f]{8}$/i.test(argb)) return null;
  return `#${argb.slice(2)}`;
}

// -- Borders -----------------------------------------------------------------

/** Excel names a border's weight and its pattern together. CSS wants them apart. */
const BORDER_STYLES: Record<string, { width: number; style: Border["style"] }> = {
  hair: { width: 1, style: "solid" },
  thin: { width: 1, style: "solid" },
  medium: { width: 2, style: "solid" },
  thick: { width: 3, style: "solid" },
  double: { width: 3, style: "double" },
  dotted: { width: 1, style: "dotted" },
  dashed: { width: 1, style: "dashed" },
  dashDot: { width: 1, style: "dashed" },
  dashDotDot: { width: 1, style: "dashed" },
  mediumDashed: { width: 2, style: "dashed" },
  mediumDashDot: { width: 2, style: "dashed" },
  mediumDashDotDot: { width: 2, style: "dashed" },
  slantDashDot: { width: 1, style: "dashed" },
};

/**
 * What a border falls back to. A line the file asked for is drawn even where it did not
 * say what colour, because losing it would lose the grid.
 */
const DEFAULT_BORDER_COLOR = "#9a9a9a";

function borderOf(side: { style?: string; color?: ArgbColor } | undefined): Border | null {
  if (!side?.style) return null;
  const shape = BORDER_STYLES[side.style] ?? { width: 1, style: "solid" as const };
  return { ...shape, color: cssColor(side.color) ?? DEFAULT_BORDER_COLOR };
}

// -- Values ------------------------------------------------------------------

/**
 * A number format, reduced to what it does to a number. Only the first section is read —
 * the `;` sections are the negative, zero and text cases, and a preview that renders the
 * positive form of every number is wrong in a way nobody misreads.
 */
function parseNumberFormat(fmt: string) {
  const positive = fmt.split(";")[0];
  const decimals = /\.(0+)/.exec(positive);
  return {
    decimals: decimals ? decimals[1].length : 0,
    grouped: positive.includes(","),
    percent: positive.includes("%"),
  };
}

/**
 * Whether a format is a date rather than a number. `m` is minutes in some contexts and
 * months in others, so the test is for the parts that are unambiguous.
 */
function isDateFormat(fmt: string): boolean {
  return /y{2,}|d{1,2}|h{1,2}|s{1,2}/i.test(fmt.replace(/\[[^\]]*\]/g, "").replace(/"[^"]*"/g, ""));
}

function formatNumber(value: number, fmt: string | undefined): string {
  // What a spreadsheet shows for an unformatted number, without the exponent notation a
  // very large one would otherwise get.
  if (!fmt || fmt === "General") return String(value);
  const { decimals, grouped, percent } = parseNumberFormat(fmt);
  const n = percent ? value * 100 : value;
  const text = n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    useGrouping: grouped,
  });
  return percent ? `${text}%` : text;
}

const pad = (n: number, width = 2) => String(n).padStart(width, "0");

/**
 * The date parts a format can name, substituted longest token first so `yyyy` is not read
 * as two `yy`. Anything else in the pattern — separators, literal text — stands.
 */
function formatDate(value: Date, fmt: string | undefined): string {
  if (!fmt) return value.toISOString().slice(0, 10);
  return fmt
    .replace(/\[[^\]]*\]/g, "")
    .replace(/yyyy/gi, String(value.getFullYear()))
    .replace(/yy/gi, pad(value.getFullYear() % 100))
    .replace(/mmmm/g, value.toLocaleString("en-US", { month: "long" }))
    .replace(/mmm/g, value.toLocaleString("en-US", { month: "short" }))
    .replace(/dddd/gi, value.toLocaleString("en-US", { weekday: "long" }))
    .replace(/ddd/gi, value.toLocaleString("en-US", { weekday: "short" }))
    .replace(/dd/gi, pad(value.getDate()))
    .replace(/d/gi, String(value.getDate()))
    .replace(/hh/g, pad(value.getHours()))
    .replace(/h/g, String(value.getHours()))
    .replace(/ss/g, pad(value.getSeconds()))
    .replace(/mm/g, pad(value.getMonth() + 1))
    .replace(/m/g, String(value.getMonth() + 1))
    .replace(/"/g, "")
    .trim();
}

/**
 * Anything a cell's value can be, as the text for it.
 *
 * A formula is shown as its cached result and never evaluated: the result is what the
 * spreadsheet that wrote the file computed. A blank one — which is what an empty form has
 * — stays blank, because the honest thing to show for a sum of nothing filled in is
 * nothing.
 */
export function cellText(value: unknown, numFmt?: string): string {
  if (value == null) return "";
  if (value instanceof Date) return formatDate(value, numFmt);
  if (typeof value === "number") {
    return numFmt && isDateFormat(numFmt) ? formatNumber(value, undefined) : formatNumber(value, numFmt);
  }
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
  if (typeof value !== "object") return String(value);

  const record = value as Record<string, unknown>;
  if ("error" in record) return String(record.error);
  if ("formula" in record || "sharedFormula" in record) return cellText(record.result, numFmt);
  if ("richText" in record) {
    const runs = record.richText as { text?: string }[] | undefined;
    return (runs ?? []).map((run) => run.text ?? "").join("");
  }
  // A hyperlink cell. The address is not rendered as a link: what is open here is a
  // document, and a click on a cell navigating out of the app is not what it means.
  if ("text" in record) return String(record.text);
  if ("hyperlink" in record) return String(record.hyperlink);
  return "";
}

// -- The grid ----------------------------------------------------------------

const ALIGNMENTS: Record<string, Align> = {
  left: "left",
  center: "center",
  centerContinuous: "center",
  right: "right",
  justify: "left",
  fill: "left",
  distributed: "center",
};

const VERTICAL: Record<string, VAlign> = {
  top: "top",
  middle: "middle",
  bottom: "bottom",
  distributed: "middle",
  justify: "middle",
};

function styleOf(cell: XCell, numeric: boolean): Style {
  const font = cell.font ?? {};
  const alignment = cell.alignment ?? {};
  const border = cell.border ?? {};
  // `fgColor` rather than `bgColor`: in a solid pattern fill it is the foreground that is
  // the block of colour, which is the one place Excel's naming catches everybody.
  const fill =
    cell.fill?.type === "pattern" && cell.fill.pattern === "solid" ? cssColor(cell.fill.fgColor) : null;

  return {
    fill,
    color: cssColor(font.color),
    bold: font.bold === true,
    italic: font.italic === true,
    underline: font.underline !== undefined && font.underline !== false,
    size: typeof font.size === "number" ? font.size * PX_PER_POINT : null,
    family: typeof font.name === "string" ? font.name : null,
    // A number with nothing said about it sits right, which is what a spreadsheet does
    // and what makes a column of figures readable.
    align: ALIGNMENTS[alignment.horizontal ?? ""] ?? (numeric ? "right" : null),
    valign: VERTICAL[alignment.vertical ?? ""] ?? null,
    wrap: alignment.wrapText === true,
    borders: {
      top: borderOf(border.top),
      right: borderOf(border.right),
      bottom: borderOf(border.bottom),
      left: borderOf(border.left),
    },
  };
}

/**
 * How far a merge starting at (row, column) reaches.
 *
 * Walked out from the master rather than read out of the worksheet's merge list, which is
 * not part of its public shape: every cell a merge covers reports that master as its own,
 * so the edge of the merge is the first cell that does not.
 */
function spanOf(
  sheet: Worksheet,
  master: XCell,
  row: number,
  column: number,
  rows: number,
  columns: number,
) {
  let colSpan = 1;
  while (column + colSpan <= columns && sheet.getCell(row, column + colSpan).master === master) {
    colSpan += 1;
  }
  let rowSpan = 1;
  while (row + rowSpan <= rows && sheet.getCell(row + rowSpan, column).master === master) {
    rowSpan += 1;
  }
  return { colSpan, rowSpan };
}

function heightOf(row: XRow, fallback: number): number {
  const points = typeof row.height === "number" ? row.height : fallback;
  return Math.round(points * PX_PER_POINT);
}

/** One worksheet as a grid of rows. */
export function readSheet(sheet: Worksheet): Sheet {
  const rowCount = Math.min(sheet.rowCount, MAX_ROWS);
  const columnCount = Math.min(sheet.columnCount, MAX_COLUMNS);
  const truncated = sheet.rowCount > rowCount || sheet.columnCount > columnCount;

  const widths: number[] = [];
  for (let c = 1; c <= columnCount; c += 1) {
    const chars = sheet.getColumn(c).width ?? DEFAULT_COLUMN_CHARS;
    widths.push(Math.round(chars * PX_PER_CHAR + CELL_PADDING_PX));
  }

  const defaultPoints = sheet.properties?.defaultRowHeight ?? DEFAULT_ROW_POINTS;
  const rows: Row[] = [];

  for (let r = 1; r <= rowCount; r += 1) {
    const source = sheet.getRow(r);
    const cells: Cell[] = [];

    for (let c = 1; c <= columnCount; c += 1) {
      const cell = sheet.getCell(r, c);
      // A cell a merge covers is not a cell in the table: the master carries it, and this
      // one would push the rest of the row out by a column.
      if (cell.master !== cell) continue;

      const { colSpan, rowSpan } = spanOf(sheet, cell, r, c, rowCount, columnCount);
      // Asked of the value rather than of the text, so a figure that formatted to
      // something with a currency symbol in it still counts as one.
      const numeric = typeof cell.value === "number" || cell.value instanceof Date;
      cells.push({ text: cellText(cell.value, cell.numFmt), colSpan, rowSpan, style: styleOf(cell, numeric) });
    }

    rows.push({ height: heightOf(source, defaultPoints), cells });
  }

  return { name: sheet.name, widths, rows, truncated };
}
