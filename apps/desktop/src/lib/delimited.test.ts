import { describe, expect, it } from "vitest";

import { parseDelimited } from "@/lib/delimited";

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
