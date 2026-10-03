import { describe, expect, it } from "vitest";

import type { BootstrapStatus } from "@/types";

import { elapsed, failedStep, isReady, progress, shownSteps } from "./bootstrap";

const status = (steps: BootstrapStatus["steps"], ready = false): BootstrapStatus => ({ steps, ready });

describe("bootstrap", () => {
  it("is not ready before the status has been read", () => {
    expect(isReady(undefined)).toBe(false);
    expect(isReady(status([], true))).toBe(true);
  });

  it("leaves a skipped step out of what is shown and counted", () => {
    const s = status([
      { id: "console_server", state: "skipped" },
      { id: "console_image", state: "skipped" },
      { id: "catalog", state: "running", started_at: 0 },
    ]);
    expect(shownSteps(s).map((x) => x.id)).toEqual(["catalog"]);
    expect(progress(s)).toEqual({ done: 0, total: 1 });
  });

  it("counts what is done, and names the step that failed", () => {
    const s = status([
      { id: "console_server", state: "failed", message: "offline" },
      { id: "console_image", state: "pending" },
      { id: "catalog", state: "done" },
    ]);
    expect(progress(s)).toEqual({ done: 1, total: 3 });
    expect(failedStep(s)).toEqual({ id: "console_server", state: "failed", message: "offline" });
    expect(failedStep(status([{ id: "catalog", state: "done" }], true))).toBeUndefined();
  });

  it("reads a running step's time as a clock does", () => {
    expect(elapsed(0, 7_400)).toBe("0:07");
    expect(elapsed(0, 102_000)).toBe("1:42");
    expect(elapsed(0, 725_000)).toBe("12:05");
    // A start stamped a moment ahead of this clock is not negative time.
    expect(elapsed(5_000, 0)).toBe("0:00");
  });
});
