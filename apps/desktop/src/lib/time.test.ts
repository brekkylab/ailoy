import { describe, expect, it } from "vitest";

import { formatElapsed, formatRelativeTime } from "@/lib/time";

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

describe("formatElapsed", () => {
  const now = new Date("2026-09-23T15:30:00").getTime();
  const back = (ms: number) => formatElapsed(now - ms, now);
  it("spells out how long ago, in the largest unit that fits", () => {
    expect(back(0)).toBe("just now");
    expect(back(59_000)).toBe("just now");
    expect(back(MINUTE)).toBe("1 minute ago");
    expect(back(5 * MINUTE)).toBe("5 minutes ago");
    expect(back(HOUR)).toBe("1 hour ago");
    expect(back(23 * HOUR)).toBe("23 hours ago");
    expect(back(DAY)).toBe("1 day ago");
    expect(back(29 * DAY)).toBe("29 days ago");
    expect(back(60 * DAY)).toBe("2 months ago");
    expect(back(400 * DAY)).toBe("1 year ago");
    expect(back(-5 * MINUTE)).toBe("just now");
    expect(formatElapsed(Number.NaN, now)).toBe("");
  });
});
