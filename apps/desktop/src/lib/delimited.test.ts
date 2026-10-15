import { describe, expect, it } from "vitest";

import {
  isNumericColumn,
  parseDelimited,
  parseTable,
  sniffDelimiter,
} from "@/lib/delimited";

const csv = (t: string) => parseDelimited(t, ",");

describe("parseDelimited", () => {
  it("splits plain rows and fields", () => {
    expect(csv("a,b\n1,2")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("keeps a delimiter that is inside quotes", () => {
    // The case the format exists for: splitting on the comma would make three fields.
    expect(csv('name,city\n"Doe, Jane",Seoul')).toEqual([
      ["name", "city"],
      ["Doe, Jane", "Seoul"],
    ]);
  });

  it("keeps a newline that is inside quotes", () => {
    expect(csv('a,"one\ntwo"\nb,c')).toEqual([
      ["a", "one\ntwo"],
      ["b", "c"],
    ]);
  });

  it("reads a doubled quote as one literal quote", () => {
    expect(csv('a,"say ""hi"""')).toEqual([["a", 'say "hi"']]);
  });

  it("treats CRLF as one ending, not as an ending and an empty row", () => {
    expect(csv("a,b\r\n1,2\r\n")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("drops a trailing newline but keeps a trailing delimiter's empty field", () => {
    expect(csv("a,b\n")).toEqual([["a", "b"]]);
    expect(csv("a,b,")).toEqual([["a", "b", ""]]);
  });

  it("keeps empty fields in the middle", () => {
    expect(csv("a,,c")).toEqual([["a", "", "c"]]);
  });

  it("takes a tab as the delimiter too", () => {
    expect(parseDelimited("a\tb\n1\t2", "\t")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });

  it("reads a quote in the middle of a bare field as a character", () => {
    // Not valid CSV, but a viewer renders what a file turned out to contain.
    expect(csv('a,b"c')).toEqual([["a", 'b"c']]);
  });

  it("closes a quoted field left open at end of input", () => {
    expect(csv('a,"unterminated')).toEqual([["a", "unterminated"]]);
  });

  it("is empty for empty input", () => {
    expect(csv("")).toEqual([]);
    expect(csv("\n")).toEqual([[""]]);
  });
});

describe("sniffDelimiter", () => {
  it("finds the character that splits the file evenly", () => {
    expect(sniffDelimiter("a,b,c\n1,2,3\n4,5,6")).toBe(",");
    expect(sniffDelimiter("a;b;c\n1;2;3\n4;5;6")).toBe(";");
    expect(sniffDelimiter("a\tb\tc\n1\t2\t3")).toBe("\t");
    expect(sniffDelimiter("a|b|c\n1|2|3")).toBe("|");
  });

  it("is not fooled by a comma inside semicolon-separated text", () => {
    // What a spreadsheet exports in a locale that spells decimals with a comma — and
    // still calls `.csv`.
    const text = "제품;단가;비고\n경유;1,632.50;주간\n휘발유;1,701.20;주간\n등유;1,410.00;주간";
    expect(sniffDelimiter(text)).toBe(";");
  });

  it("prefers the reading that is consistent, not the one that splits most", () => {
    // The pipe appears once; the comma splits every record into three.
    expect(sniffDelimiter("a,b,c\nd,e,f\ng|h,i,j")).toBe(",");
  });

  it("falls back to a comma when nothing splits anything", () => {
    expect(sniffDelimiter("one line of prose\nand another")).toBe(",");
  });
});

describe("parseTable", () => {
  it("takes the first record as the header and pads the ragged ones", () => {
    const table = parseTable("a,b,c\n1,2\n3,4,5,6");
    expect(table.columns).toBe(4);
    expect(table.header).toEqual(["a", "b", "c", ""]);
    expect(table.rows).toEqual([
      ["1", "2", "", ""],
      ["3", "4", "5", "6"],
    ]);
  });

  it("obeys a delimiter it was given rather than sniffing", () => {
    expect(parseTable("a;b\n1;2", ",").columns).toBe(1);
    expect(parseTable("a;b\n1;2").delimiter).toBe(";");
  });

  it("drops blank lines, which are not empty records", () => {
    expect(parseTable("a,b\n\n1,2\n\n").rows).toEqual([["1", "2"]]);
  });

  it("eats a byte order mark instead of naming a column after it", () => {
    expect(parseTable("﻿name,value\nx,1").header).toEqual(["name", "value"]);
  });

  it("is empty for an empty file", () => {
    expect(parseTable("")).toMatchObject({ header: [], rows: [], columns: 0 });
  });
});

describe("isNumericColumn", () => {
  const rows = [
    ["2026-01-02", "1,632.50", "경유", "", "-12"],
    ["2026-01-09", "₩1,701", "휘발유", "", "+3.5"],
    ["2026-01-16", "", "등유", "", "(7)"],
  ];

  it("counts figures, separators and currency and all", () => {
    expect(isNumericColumn(rows, 1)).toBe(true);
    expect(isNumericColumn(rows, 4)).toBe(true);
  });

  it("does not count dates or words", () => {
    expect(isNumericColumn(rows, 0)).toBe(false);
    expect(isNumericColumn(rows, 2)).toBe(false);
  });

  it("does not count a column with nothing in it", () => {
    expect(isNumericColumn(rows, 3)).toBe(false);
  });
});
