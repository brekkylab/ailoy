import { describe, expect, it } from "vitest";

import { modelsOf, monogram } from "./providers";
import type { ModelInfo } from "@/types";

const model = (provider: string, name: string, available: boolean): ModelInfo => ({
  id: `${provider}/${name}`,
  provider,
  name,
  context: null,
  output: null,
  cost: null,
  reasoning: false,
  tool_call: true,
  available,
});

describe("monogram", () => {
  it("takes the distinctive word of a vendor-plus-product name", () => {
    expect(monogram("Amazon Bedrock")).toBe("B");
    expect(monogram("Moonshot Kimi")).toBe("K");
  });

  it("takes the first letter of a one-word name, whatever its case", () => {
    expect(monogram("OpenAI")).toBe("O");
    expect(monogram("xAI")).toBe("X");
  });

  it("has something to draw for a label the engine adds later", () => {
    expect(monogram("")).toBe("");
    expect(monogram("  spaced  out  ")).toBe("O");
  });
});

describe("modelsOf", () => {
  const models = [
    model("Anthropic", "Claude Opus 5", true),
    model("Anthropic", "Claude Sonnet 5", true),
    model("xAI", "Grok 4", false),
  ];

  it("counts a provider's models and how many its key unlocks", () => {
    expect(modelsOf(models, "Anthropic")).toEqual({
      total: 2,
      available: 2,
      names: ["Claude Opus 5", "Claude Sonnet 5"],
    });
    expect(modelsOf(models, "xAI")).toEqual({ total: 1, available: 0, names: ["Grok 4"] });
  });

  it("is empty for a provider with no catalog entries, and before the catalog loads", () => {
    expect(modelsOf(models, "DeepSeek")).toEqual({ total: 0, available: 0, names: [] });
    expect(modelsOf(undefined, "Anthropic")).toEqual({ total: 0, available: 0, names: [] });
  });
});
