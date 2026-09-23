import { describe, expect, it } from "vitest";

import { formatTokens, trimZeros } from "./tokens";

describe("a token count", () => {
  it("is exact below 1k and a magnitude above it", () => {
    expect([formatTokens(0), formatTokens(512), formatTokens(999)]).toEqual(["0", "512", "999"]);
    expect([formatTokens(1000), formatTokens(3782), formatTokens(81_008), formatTokens(84_790)]).toEqual([
      "1k",
      "3.8k",
      "81k",
      "84.8k",
    ]);
    expect([formatTokens(1_000_000), formatTokens(1_050_000), formatTokens(2_345_678)]).toEqual([
      "1M",
      "1.05M",
      "2.35M",
    ]);
  });

  it("drops the zeros a fixed-point format leaves", () => {
    expect([trimZeros("1.50"), trimZeros("2.00"), trimZeros("0.15"), trimZeros("10")]).toEqual(["1.5", "2", "0.15", "10"]);
  });
});
