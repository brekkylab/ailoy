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
