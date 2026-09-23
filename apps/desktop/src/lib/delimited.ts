// Splitting a delimited file into rows and fields.
//
// Hand-written rather than taken from a library, because the whole of the format that
// matters here is one state machine: a field is quoted or it is not, a quote inside a
// quoted field is doubled, and everything else is a character. A parser that split on the
// delimiter would break the first address with a comma in it, which is the case the
// format exists to handle.

/**
 * `text` as rows of fields.
 *
 * Both line endings are accepted, and a quoted field may contain either — which is the
 * reason this cannot be done by splitting on newlines first. A trailing newline does not
 * produce an empty last row; a trailing *delimiter* does produce an empty last field,
 * because that is a field the file actually declared.
 *
 * Nothing is rejected. A stray quote in an unquoted field is a character, and a quoted
 * field left open at end of input is closed there: this renders whatever a file turned out
 * to contain rather than refusing a file for being slightly wrong, which is what a viewer
 * is for.
 */
export function parseDelimited(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let i = 0;

  const endField = () => {
    row.push(field);
    field = "";
  };
  const endRow = () => {
    endField();
    rows.push(row);
    row = [];
  };

  while (i < text.length) {
    const c = text[i];

    if (quoted) {
      if (c === '"') {
        // A doubled quote is one literal quote; a single one ends the field.
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        quoted = false;
        i += 1;
        continue;
      }
      field += c;
      i += 1;
      continue;
    }

    if (c === '"' && field === "") {
      quoted = true;
      i += 1;
      continue;
    }
    if (c === delimiter) {
      endField();
      i += 1;
      continue;
    }
    if (c === "\r" || c === "\n") {
      endRow();
      // Consume CRLF as one ending rather than as an ending and an empty row.
      i += c === "\r" && text[i + 1] === "\n" ? 2 : 1;
      continue;
    }
    field += c;
    i += 1;
  }

  // Whatever is left is a final row — unless the file ended on its last newline, in which
  // case there is nothing in hand and there is no row to make.
  if (field !== "" || row.length > 0) endRow();
  return rows;
}

/**
 * The delimiters worth sniffing for.
 *
 * A file that separates on anything else reads as one column per record, which is the
 * honest answer rather than a wrong guess.
 */
const DELIMITERS = [",", ";", "\t", "|"];

/** How much of a file the sniffer looks at, and how many records it reads out of it. */
const SNIFF_BYTES = 64 << 10;
const SNIFF_ROWS = 40;

/**
 * The delimiter `text` is most likely written with.
 *
 * `.csv` names the file, not the character: an export from a spreadsheet set to a locale
 * that spells decimals with a comma separates on a semicolon, and the file is still
 * called `.csv`. Read with the wrong one it comes back as a single column of whole lines,
 * which is exactly what a reader sees and cannot explain.
 *
 * Each candidate is scored by how consistently it gives the same number of columns across
 * the sample, weighted by how many columns that is: a character that splits every record
 * into five fields is a delimiter, and one that splits a few records unevenly is a
 * character that happens to appear in the text. Agreement is squared so a clean 2-column
 * read beats a ragged 3-column one.
 */
export function sniffDelimiter(text: string): string {
  const sample = text.slice(0, SNIFF_BYTES);
  let best = DELIMITERS[0];
  let bestScore = 0;

  for (const delimiter of DELIMITERS) {
    const rows = parseDelimited(sample, delimiter)
      .filter((row) => row.length > 1 || row[0] !== "")
      .slice(0, SNIFF_ROWS);
    // The last record of a sliced sample may have been cut mid-line.
    const counts = (rows.length > 1 ? rows.slice(0, -1) : rows).map((row) => row.length);
    if (counts.length === 0) continue;

    const tally = new Map<number, number>();
    for (const n of counts) tally.set(n, (tally.get(n) ?? 0) + 1);

    let mode = 0;
    let hits = 0;
    for (const [n, count] of tally) {
      // Ties go to the wider read: four columns say more than two.
      if (count > hits || (count === hits && n > mode)) {
        mode = n;
        hits = count;
      }
    }
    if (mode < 2) continue;

    const agreement = hits / counts.length;
    const score = agreement * agreement * mode;
    if (score > bestScore) {
      bestScore = score;
      best = delimiter;
    }
  }

  return best;
}

/** A delimited file, read. */
export interface Table {
  /** What it was read with, sniffed unless the caller knew. */
  delimiter: string;
  /** The first record. A file with no header shows its first row of data up there. */
  header: string[];
  /** Every record after it, each padded out to `columns`. */
  rows: string[][];
  /** The width of the widest record. */
  columns: number;
}

/**
 * `text` as a table. `delimiter` overrides the sniffer — what a `.tsv` already knows.
 *
 * Ragged records are padded and never clipped: a trailing field only one row has is still
 * that row's data. Blank lines are dropped, because in a one-column file an empty record
 * and a blank line are the same bytes and skipping them is the reading that matches what
 * the file means.
 */
export function parseTable(text: string, delimiter?: string): Table {
  // A byte order mark is a byte order mark, not the first letter of the first column name.
  const body = text.replace(/^﻿/, "");
  const sep = delimiter ?? sniffDelimiter(body);
  const records = parseDelimited(body, sep).filter((row) => row.length > 1 || row[0] !== "");
  const columns = records.reduce((wide, row) => Math.max(wide, row.length), 0);
  const pad = (row: string[]) =>
    row.length === columns ? row : row.concat(Array<string>(columns - row.length).fill(""));

  return {
    delimiter: sep,
    header: pad(records[0] ?? []),
    rows: records.slice(1).map(pad),
    columns,
  };
}

/**
 * Whether a column holds numbers, and so should be read right-aligned.
 *
 * Thousands separators and a leading currency symbol still count; a column of dates or
 * ids does not, which is why this is asked per column and not per cell. Empty cells are
 * ignored, and a column that is entirely empty is not numeric.
 */
export function isNumericColumn(rows: readonly string[][], index: number): boolean {
  let seen = 0;
  for (const row of rows) {
    const cell = (row[index] ?? "").trim();
    if (cell === "") continue;
    if (!/^[-+(]?[$€£¥₩]?\s?\d[\d,\s]*(?:\.\d+)?\)?%?$/.test(cell)) return false;
    seen += 1;
  }
  return seen > 0;
}
