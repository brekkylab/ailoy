import { describe, expect, it } from "vitest";

import { formatMessageTime, formatRelativeTime } from "@/lib/time";

const NOW = new Date("2026-09-14T12:00:00").getTime();
const ago = (ms: number) => formatRelativeTime(NOW - ms, NOW);

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

describe("formatRelativeTime", () => {
  it("calls anything under a minute just now", () => {
    expect(ago(0)).toBe("just now");
    expect(ago(59_999)).toBe("just now");
  });

  it("counts down in the largest unit that still fits", () => {
    expect(ago(MINUTE)).toBe("1m ago");
    expect(ago(59 * MINUTE)).toBe("59m ago");
    expect(ago(HOUR)).toBe("1h ago");
    expect(ago(23 * HOUR + 59 * MINUTE)).toBe("23h ago");
    expect(ago(DAY)).toBe("1d ago");
    expect(ago(6 * DAY)).toBe("6d ago");
  });

  it("floors rather than rounds, so a row never claims more time than has passed", () => {
    expect(ago(119_000)).toBe("1m ago");
  });

  it("switches to a local calendar date at a week", () => {
    expect(ago(7 * DAY)).toBe("2026-09-07");
    // Zero-padded, and read in local time: the UTC date here is the 2nd in Seoul's zone.
    expect(formatRelativeTime(new Date("2026-01-02T09:00:00").getTime(), NOW)).toBe("2026-01-02");
  });

  it("does not count forwards when a clock runs ahead", () => {
    expect(ago(-5 * MINUTE)).toBe("just now");
  });

  it("says nothing for a timestamp that is not one", () => {
    expect(formatRelativeTime(Number.NaN, NOW)).toBe("");
  });
});

describe("formatMessageTime", () => {
  const now = new Date("2026-09-23T15:30:00").getTime();
  it("is a clock today, and carries the date otherwise", () => {
    expect(formatMessageTime(new Date("2026-09-23T09:05:00").getTime(), now)).toBe("09:05");
    expect(formatMessageTime(new Date("2026-09-22T23:59:00").getTime(), now)).toBe("Sep 22, 23:59");
    expect(formatMessageTime(new Date("2025-12-31T08:00:00").getTime(), now)).toBe("Dec 31, 2025, 08:00");
    expect(formatMessageTime(Number.NaN, now)).toBe("");
  });
});
