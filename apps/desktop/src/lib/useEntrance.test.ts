import { describe, expect, it } from "vitest";

import { isEntering, recordEntrance, type EntranceRow } from "./useEntrance";

const start = { session: "s", primed: false, seen: new Set<string>(), entering: new Set<string>(), streamed: false };
const rows = (...keys: string[]): EntranceRow[] => keys.map((key) => ({ key, answer: key.startsWith("a") }));

describe("which rows fade in", () => {
  it("takes the first list a session shows as history", () => {
    const s = recordEntrance(start, true, rows("u1", "a1"), false);
    expect(isEntering(s, { key: "u1" })).toBe(false);
    expect(isEntering(s, { key: "a1", answer: true })).toBe(false);
    // Not before the list has arrived, either.
    expect(recordEntrance(start, false, rows("u1"), false)).toBe(start);
  });

  it("fades in what arrives afterwards, and keeps it entering", () => {
    let s = recordEntrance(start, true, rows("u1"), false);
    expect(isEntering(s, { key: "u2" })).toBe(true);
    s = recordEntrance(s, true, rows("u1", "u2"), false);
    expect(isEntering(s, { key: "u2" })).toBe(true);
    // Nothing new, nothing changed — the same state, so no update.
    expect(recordEntrance(s, true, rows("u1", "u2"), false)).toBe(s);
  });

  it("lets a stored answer take over from the stream without a fade", () => {
    let s = recordEntrance(start, true, rows("u1"), false);
    // The block the run streams into is new, and fades in; then text streams into it.
    s = recordEntrance(s, true, rows("u1", "live:1"), true);
    expect(isEntering(s, { key: "live:1" })).toBe(true);
    // The stored answer replaces it: already read, so no fade.
    expect(isEntering(s, { key: "a2", answer: true })).toBe(false);
    s = recordEntrance(s, true, rows("u1", "a2"), false);
    expect(isEntering(s, { key: "a2", answer: true })).toBe(false);
    // An answer that was never streamed does fade in.
    expect(isEntering(s, { key: "a3", answer: true })).toBe(true);
  });
});
