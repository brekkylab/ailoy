import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import { BYTES_KEY, trimBytes } from "./bytes";

const MiB = 1 << 20;

function client(files: [path: string, mib: number, at: number][]) {
  const qc = new QueryClient();
  for (const [path, mib, at] of files) {
    qc.setQueryData([BYTES_KEY, path], new ArrayBuffer(mib * MiB), { updatedAt: at });
  }
  return qc;
}

const held = (qc: QueryClient) =>
  qc
    .getQueryCache()
    .findAll({ queryKey: [BYTES_KEY] })
    .map((q) => q.queryKey[1])
    .sort();

describe("trimBytes", () => {
  it("keeps what fits and drops the oldest reads", () => {
    const qc = client([
      ["/old.pdf", 4, 1_000],
      ["/newer.pdf", 4, 2_000],
      ["/newest.pdf", 4, 3_000],
    ]);
    trimBytes(qc, 9 * MiB);
    expect(held(qc)).toEqual(["/newer.pdf", "/newest.pdf"]);
  });

  it("does nothing while everything fits", () => {
    const qc = client([
      ["/a.pdf", 4, 1_000],
      ["/b.pdf", 4, 2_000],
    ]);
    trimBytes(qc, 64 * MiB);
    expect(held(qc)).toEqual(["/a.pdf", "/b.pdf"]);
  });

  it("never drops the file on screen, even when it is the whole budget", () => {
    // The newest read is what a viewer is drawing from; evicting it would re-read the file
    // the reader is looking at, which is the one read the cache exists to prevent.
    const qc = client([
      ["/small.pdf", 1, 1_000],
      ["/huge.pdf", 80, 2_000],
    ]);
    trimBytes(qc, 8 * MiB);
    expect(held(qc)).toEqual(["/huge.pdf"]);
  });

  it("ignores entries that hold no bytes", () => {
    const qc = new QueryClient();
    qc.setQueryData([BYTES_KEY, "/pending.pdf"], undefined);
    qc.setQueryData(["file", "/other.txt"], { text: "not bytes" });
    trimBytes(qc, 1);
    expect(held(qc)).toEqual([]);
    expect(qc.getQueryData(["file", "/other.txt"])).toBeTruthy();
  });
});
