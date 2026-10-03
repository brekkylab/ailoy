import { describe, expect, it } from "vitest";

import type { ModelInfo, ProviderSetting } from "@/types";

import { formatPrice, initialVendor, searchModels, vendorsOf } from "./modelPicker";

const model = (provider: string, id: string, name: string, available = true): ModelInfo => ({
  id,
  provider,
  name,
  context: null,
  output: null,
  cost: null,
  reasoning: false,
  tool_call: true,
  available,
});

const provider = (key: string, label: string, has_key: boolean): ProviderSetting => ({
  key,
  label,
  has_key,
  key_hint: "",
  region: null,
  regions: [],
  routing: null,
  routings: [],
});

const MODELS = [
  model("Anthropic", "anthropic/claude-opus-5", "Claude Opus 5"),
  model("Anthropic", "anthropic/claude-sonnet-5", "Claude Sonnet 5"),
  model("OpenAI", "openai/gpt-6-sol", "GPT-6 Sol", false),
  model("Amazon Bedrock", "bedrock/global.anthropic.claude-sonnet-5", "Claude Sonnet 5"),
  model("Amazon Bedrock", "bedrock/global.openai.gpt-5.6-luna", "GPT-5.6 Luna"),
];

describe("the model picker", () => {
  it("groups the catalog under the providers the settings list, in their order", () => {
    const vendors = vendorsOf(
      [provider("openai", "OpenAI", false), provider("anthropic", "Anthropic", true), provider("xai", "xAI", false)],
      MODELS,
    );
    expect(vendors.map((v) => [v.key, v.models.length])).toEqual([
      ["openai", 1],
      ["anthropic", 2],
      // No models yet is still a vendor, so the list does not shift when the catalog lands.
      ["xai", 0],
    ]);
  });

  it("finds a model by every word, across names, ids and vendors", () => {
    expect(searchModels(MODELS, "opus 5").map((m) => m.id)).toEqual(["anthropic/claude-opus-5"]);
    expect(searchModels(MODELS, "bedrock sonnet").map((m) => m.id)).toEqual([
      "bedrock/global.anthropic.claude-sonnet-5",
    ]);
    expect(searchModels(MODELS, "  ")).toEqual([]);
  });

  it("puts a model the user can call first, then a name that starts with the query", () => {
    // Both names start with "GPT"; the one without a key goes second.
    expect(searchModels(MODELS, "gpt").map((m) => m.id)).toEqual([
      "bedrock/global.openai.gpt-5.6-luna",
      "openai/gpt-6-sol",
    ]);
    // Callable and leading beats callable and merely matching.
    expect(searchModels(MODELS, "claude").map((m) => m.name)[0]).toBe("Claude Opus 5");
  });

  it("leads with the name that holds the query as typed", () => {
    // "5" is in "4.5" as well, which is why the older model is found at all.
    const models = [model("Anthropic", "anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5"), ...MODELS];
    expect(searchModels(models, "sonnet 5").map((m) => m.id)).toEqual([
      "anthropic/claude-sonnet-5",
      "bedrock/global.anthropic.claude-sonnet-5",
      "anthropic/claude-sonnet-4-5",
    ]);
  });

  it("opens on the current model's vendor, else the first one with a key", () => {
    const vendors = vendorsOf([provider("openai", "OpenAI", false), provider("anthropic", "Anthropic", true)], MODELS);
    expect(initialVendor(vendors, "openai/gpt-6-sol")).toBe("openai");
    expect(initialVendor(vendors, null)).toBe("anthropic");
    expect(initialVendor(vendors, "gone/model")).toBe("anthropic");
    expect(initialVendor([], null)).toBeNull();
  });

  it("writes a price the way a row can spare it", () => {
    expect(formatPrice({ input: 5, output: 25, cache_read: null, cache_write: null })).toBe("$5 / $25");
    expect(formatPrice({ input: 0.15, output: 0.6, cache_read: null, cache_write: null })).toBe("$0.15 / $0.6");
    expect(formatPrice({ input: 3, output: null, cache_read: null, cache_write: null })).toBeNull();
    expect(formatPrice(null)).toBeNull();
  });
});
