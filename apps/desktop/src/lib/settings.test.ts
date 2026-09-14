import { describe, expect, it } from "vitest";

import type { ProviderSetting, Settings } from "@/types";

import { hasAnyKey } from "./settings";

const provider = (key: string, has_key: boolean): ProviderSetting => ({
  key,
  label: key,
  has_key,
  key_hint: has_key ? "abcd" : "",
  region: null,
});

const settings = (providers: ProviderSetting[]): Settings => ({
  providers,
  default_model: "",
  max_tokens: 4096,
  max_turns: 20,
  catalog_refresh: true,
});

describe("hasAnyKey", () => {
  it("is true when any provider has a key, false when none does", () => {
    expect(hasAnyKey(settings([provider("anthropic", false), provider("openai", true)]))).toBe(true);
    expect(hasAnyKey(settings([provider("anthropic", false), provider("openai", false)]))).toBe(false);
  });

  it("reads unloaded settings as 'yes', so the no-key warning never flashes on a cold start", () => {
    expect(hasAnyKey(undefined)).toBe(true);
    expect(hasAnyKey(null)).toBe(true);
  });

  it("is false for a build with no providers at all", () => {
    expect(hasAnyKey(settings([]))).toBe(false);
  });
});
