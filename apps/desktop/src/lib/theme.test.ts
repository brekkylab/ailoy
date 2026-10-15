import { describe, expect, it } from "vitest";

import { parseTheme, THEMES } from "./theme";

describe("parseTheme", () => {
  it("takes the two values that pin a palette", () => {
    expect(parseTheme("light")).toBe("light");
    expect(parseTheme("dark")).toBe("dark");
  });

  it("reads anything else as following the system", () => {
    // Nothing stored yet, a key some other build wrote, a value edited by hand.
    expect(parseTheme(null)).toBe("system");
    expect(parseTheme(undefined)).toBe("system");
    expect(parseTheme("")).toBe("system");
    expect(parseTheme("system")).toBe("system");
    expect(parseTheme("Dark")).toBe("system");
    expect(parseTheme("solarized")).toBe("system");
  });

  it("parses each theme it offers", () => {
    for (const t of THEMES) expect(THEMES).toContain(parseTheme(t === "system" ? null : t));
  });
});
