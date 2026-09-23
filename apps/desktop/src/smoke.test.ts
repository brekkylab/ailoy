import { describe, expect, it } from "vitest";

import dialogSrc from "@/components/ui/dialog.tsx?raw";
import { S } from "@/strings";

describe("strings", () => {
  it("has an app name", () => {
    expect(S.appName).toBe("Ailoy");
  });
});

describe("generated ui", () => {
  it("dialog close labels come from strings.ts (RC11), not the shadcn English default", () => {
    expect(dialogSrc).not.toMatch(/>\s*Close\s*</);
    expect(dialogSrc).toContain("S.close");
  });
});
