import { describe, expect, it } from "vitest";

import { providerOf } from "./settingsPatch";

describe("providerOf", () => {
  it("names the provider whose key is being written", () => {
    expect(providerOf({ provider_keys: { openai: "sk-1" } })).toBe("openai");
    expect(providerOf({ provider_keys: { anthropic: null } })).toBe("anthropic");
  });

  it("gives Bedrock its region and its routing", () => {
    expect(providerOf({ bedrock_region: "eu-central-1" })).toBe("bedrock");
    expect(providerOf({ bedrock_routing: "eu" })).toBe("bedrock");
    // A key and a region in one patch is still one pane's.
    expect(providerOf({ provider_keys: { bedrock: "k" }, bedrock_region: "us-east-1" })).toBe("bedrock");
  });

  it("is nothing for a patch no pane sent", () => {
    expect(providerOf(undefined)).toBeNull();
    expect(providerOf({})).toBeNull();
    expect(providerOf({ default_model: "anthropic/claude-opus-5" })).toBeNull();
    expect(providerOf({ max_turns: 10 })).toBeNull();
    expect(providerOf({ provider_keys: {} })).toBeNull();
  });
});
