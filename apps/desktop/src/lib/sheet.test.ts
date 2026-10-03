import { Workbook, type Worksheet } from "exceljs";
import { describe, expect, it } from "vitest";

import { cellText, readSheet } from "@/lib/sheet";

/** A sheet built in memory, so the tests read the same path a file does. */
function sheetWith(build: (ws: Worksheet) => void) {
  const ws = new Workbook().addWorksheet("Sheet1");
  build(ws);
  return readSheet(ws);
}

/** The cell holding `text`. An empty cell is a cell too, so position is not the way in. */
function cellNamed(sheet: ReturnType<typeof readSheet>, text: string) {
  const found = sheet.rows.flatMap((r) => r.cells).find((c) => c.text === text);
  if (!found) throw new Error(`no cell holding ${text}`);
  return found;
}

describe("readSheet", () => {
  it("takes the file's own column widths, in pixels", () => {
    const sheet = sheetWith((ws) => {
      ws.getColumn(1).width = 20;
      ws.getCell("A1").value = "wide";
      ws.getCell("B1").value = "default";
    });
    // 20 characters at 7px, plus the 5px of cell padding Excel's own notes document.
    expect(sheet.widths[0]).toBe(145);
    // A column the file says nothing about still gets a width, or the table would
    // rebalance it against its content.
    expect(sheet.widths[1]).toBeGreaterThan(0);
  });

  it("gives a merge to its master and drops the cells it covers", () => {
    // Left in, they would push the rest of the row out by a column each.
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = "title";
      ws.mergeCells("A1:C1");
      ws.getCell("D1").value = "after";
    });
    expect(sheet.rows[0].cells).toHaveLength(2);
    expect(sheet.rows[0].cells[0]).toMatchObject({ text: "title", colSpan: 3, rowSpan: 1 });
    expect(sheet.rows[0].cells[1].text).toBe("after");
  });

  it("reads a merge that reaches down as well as across", () => {
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = "tall";
      ws.mergeCells("A1:B3");
    });
    expect(sheet.rows[0].cells[0]).toMatchObject({ colSpan: 2, rowSpan: 3 });
  });

  it("drops the alpha byte off a colour instead of reading it as transparency", () => {
    // Excel writes `00` on colours that are plainly opaque, and a grey header row read
    // as alpha goes invisible.
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = "head";
      ws.getCell("A1").fill = { type: "pattern", pattern: "solid", fgColor: { argb: "00E8E8E8" } };
      ws.getCell("A1").font = { bold: true, color: { argb: "FF1A1A1A" } };
    });
    expect(sheet.rows[0].cells[0].style).toMatchObject({
      fill: "#E8E8E8",
      color: "#1A1A1A",
      bold: true,
    });
  });

  it("gives a border a colour even where the file named none", () => {
    // Dropping it would drop the line, and the grid with it.
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = "x";
      ws.getCell("A1").border = { top: { style: "thin" }, left: { style: "double" } };
    });
    const { borders } = sheet.rows[0].cells[0].style;
    expect(borders.top).toMatchObject({ width: 1, style: "solid" });
    expect(borders.top?.color).toMatch(/^#/);
    expect(borders.left).toMatchObject({ width: 3, style: "double" });
    // A side the file says nothing about stays null, so the neighbour's line still shows.
    expect(borders.right).toBeNull();
  });

  it("sits a bare number to the right, and leaves text alone", () => {
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = 42;
      ws.getCell("B1").value = "label";
    });
    expect(sheet.rows[0].cells[0].style.align).toBe("right");
    expect(sheet.rows[0].cells[1].style.align).toBeNull();
  });

  it("says when it cut the sheet off", () => {
    const small = sheetWith((ws) => {
      ws.getCell("A1").value = "x";
    });
    expect(small.truncated).toBe(false);
  });
});

describe("cellText", () => {
  it("shows a formula's cached result, never the formula", () => {
    expect(cellText({ formula: "SUM(A1:A9)", result: 12 })).toBe("12");
    // An empty form's sum of nothing filled in is nothing, not a zero.
    expect(cellText({ formula: "SUM(A1:A9)", result: null })).toBe("");
  });

  it("applies the number format the cell carries", () => {
    expect(cellText(1234.5, "#,##0.00")).toBe("1,234.50");
    expect(cellText(0.125, "0.0%")).toBe("12.5%");
    expect(cellText(1234.5)).toBe("1234.5");
  });

  it("does not read a date format as a number format", () => {
    // `yyyy-mm-dd` has a `,`-free pattern but `0` grouping rules would still mangle it.
    expect(cellText(45000, "yyyy-mm-dd")).toBe("45000");
  });

  it("joins the runs of a rich-text cell", () => {
    expect(cellText({ richText: [{ text: "가" }, { text: "나" }] })).toBe("가나");
  });

  it("shows an error as the error", () => {
    expect(cellText({ error: "#DIV/0!" })).toBe("#DIV/0!");
  });

  it("shows a hyperlink's text rather than making it a link", () => {
    expect(cellText({ text: "Anthropic", hyperlink: "https://example.invalid" })).toBe("Anthropic");
  });

  it("is empty for an empty cell", () => {
    expect(cellText(null)).toBe("");
    expect(cellText(undefined)).toBe("");
  });
});

describe("spill", () => {
  it("lets a label run over the empty cells after it", () => {
    // What a spreadsheet does and a table does not: the heading of a form sits in a
    // narrow column with a run of empty ones after it, and no merge anywhere.
    const sheet = sheetWith((ws) => {
      for (let c = 1; c <= 4; c += 1) ws.getColumn(c).width = 10;
      ws.getCell("A1").value = "◆ 판교테크노밸리 입주기업 임직원 거주지, 통근수단";
      ws.getCell("D1").value = "stop";
    });
    // B and C are free; D holds something, so the run ends there.
    expect(cellNamed(sheet, "◆ 판교테크노밸리 입주기업 임직원 거주지, 통근수단").spill).toEqual({ left: 0, right: sheet.widths[1] + sheet.widths[2] });
  });

  it("runs the other way for a label aligned right", () => {
    const sheet = sheetWith((ws) => {
      for (let c = 1; c <= 3; c += 1) ws.getColumn(c).width = 10;
      ws.getCell("C1").value = "a long label";
      ws.getCell("C1").alignment = { horizontal: "right" };
    });
    expect(cellNamed(sheet, "a long label").spill).toEqual({
      left: sheet.widths[0] + sheet.widths[1],
      right: 0,
    });
  });

  it("runs both ways for a centred one", () => {
    const sheet = sheetWith((ws) => {
      for (let c = 1; c <= 3; c += 1) ws.getColumn(c).width = 10;
      ws.getCell("B1").value = "centred";
      ws.getCell("B1").alignment = { horizontal: "center" };
      // A column with nothing in it anywhere is not a column of the sheet; this puts one
      // to the right of the label without putting anything beside it.
      ws.getCell("C2").value = "elsewhere";
    });
    expect(cellNamed(sheet, "centred").spill).toEqual({
      left: sheet.widths[0],
      right: sheet.widths[2],
    });
  });

  it("does not run out of a cell set to wrap", () => {
    // Wrapping is the file saying the text belongs inside the column.
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = "wrapped text";
      ws.getCell("A1").alignment = { wrapText: true };
    });
    expect(cellNamed(sheet, "wrapped text").spill).toEqual({ left: 0, right: 0 });
  });

  it("stops at a cell a merge covers, which is not empty", () => {
    const sheet = sheetWith((ws) => {
      ws.getCell("A1").value = "label";
      ws.getCell("B1").value = "merged";
      ws.mergeCells("B1:C1");
    });
    expect(cellNamed(sheet, "label").spill.right).toBe(0);
  });

  it("gives an empty cell nowhere to run", () => {
    const sheet = sheetWith((ws) => {
      ws.getCell("A2").value = "x";
    });
    expect(sheet.rows[0].cells.every((c) => c.spill.left === 0 && c.spill.right === 0)).toBe(true);
  });

  it("measures a cell as the columns it covers", () => {
    const sheet = sheetWith((ws) => {
      for (let c = 1; c <= 3; c += 1) ws.getColumn(c).width = 10;
      ws.getCell("A1").value = "merged";
      ws.mergeCells("A1:B1");
    });
    expect(cellNamed(sheet, "merged").width).toBe(sheet.widths[0] + sheet.widths[1]);
  });
});
