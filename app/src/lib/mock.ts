// Placeholder data for the UI. Nothing here talks to ailoy; it is replaced wholesale once
// the backend is wired in.

import type { ChatMessage, Model, Provider, SessionSummary } from "@/types";

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;

export const PROVIDERS: Provider[] = [
  { key: "anthropic", label: "Anthropic" },
  { key: "openai", label: "OpenAI" },
  { key: "google", label: "Google Gemini" },
  { key: "bedrock", label: "Amazon Bedrock" },
];

// Ids spelled the way ailoy resolves them: `<provider>/<model>`.
export const MODELS: Model[] = [
  { id: "anthropic/claude-sonnet-5", name: "Claude Sonnet 5", provider: "anthropic" },
  { id: "anthropic/claude-opus-5-5", name: "Claude Opus 5.5", provider: "anthropic" },
  { id: "openai/gpt-5", name: "GPT-5", provider: "openai" },
  { id: "google/gemini-3-pro", name: "Gemini 3 Pro", provider: "google" },
  { id: "bedrock/anthropic.claude-sonnet-5", name: "Claude Sonnet 5", provider: "bedrock" },
  { id: "bedrock/anthropic.claude-opus-5-5", name: "Claude Opus 5.5", provider: "bedrock" },
  // GPT-6 Luna and Sol have no plain id on Bedrock, only inference profiles, hence `global.`.
  { id: "bedrock/openai.gpt-6-astra", name: "GPT-6 Astra", provider: "bedrock" },
  { id: "bedrock/global.openai.gpt-6-luna", name: "GPT-6 Luna", provider: "bedrock" },
  { id: "bedrock/global.openai.gpt-6-sol", name: "GPT-6 Sol", provider: "bedrock" },
  { id: "bedrock/openai.gpt-5.6-luna", name: "GPT-5.6 Luna", provider: "bedrock" },
  { id: "bedrock/openai.gpt-5.6-sol", name: "GPT-5.6 Sol", provider: "bedrock" },
  { id: "bedrock/openai.gpt-5.6-terra", name: "GPT-5.6 Terra", provider: "bedrock" },
  { id: "bedrock/amazon.nova-pro-v1:0", name: "Nova Pro", provider: "bedrock" },
];

/** The regions ailoy's `BedrockRegion` accepts. `us-east-1` is the default there too. */
export const BEDROCK_REGIONS = [
  "us-east-1", "us-east-2", "us-west-1", "us-west-2", "us-gov-east-1", "us-gov-west-1",
  "ca-central-1", "ca-west-1", "sa-east-1", "mx-central-1",
  "eu-central-1", "eu-central-2", "eu-west-1", "eu-west-2", "eu-west-3", "eu-north-1", "eu-south-1", "eu-south-2",
  "il-central-1", "me-central-1", "me-south-1", "af-south-1",
  "ap-south-1", "ap-south-2", "ap-east-2", "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
  "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-4", "ap-southeast-5", "ap-southeast-6", "ap-southeast-7",
];
export const DEFAULT_BEDROCK_REGION = "us-east-1";

export function seedSessions(now = Date.now()): SessionSummary[] {
  return [
    { id: "s1", title: "Quarterly report outline", model: MODELS[0].id, updated_at: now - 20 * 60 * 1000 },
    { id: "s2", title: "Rust borrow checker question", model: MODELS[1].id, updated_at: now - 3 * HOUR },
    { id: "s3", title: "Trip itinerary to Jeju", model: MODELS[2].id, updated_at: now - DAY - HOUR },
    { id: "s4", title: "Summarize meeting notes", model: MODELS[0].id, updated_at: now - 4 * DAY },
    { id: "s5", title: "Regex for Korean phone numbers", model: MODELS[4].id, updated_at: now - 12 * DAY },
  ];
}

export function seedMessages(now = Date.now()): Record<string, ChatMessage[]> {
  return {
    s1: [
      { id: "m1", role: "user", text: "Can you outline a quarterly report for our team?", created_at: now - 22 * 60 * 1000 },
      {
        id: "m2",
        role: "assistant",
        text: "Sure. A simple structure:\n\n1. Summary of the quarter\n2. Key metrics and how they moved\n3. What shipped\n4. Risks and open issues\n5. Plan for next quarter",
        created_at: now - 20 * 60 * 1000,
      },
    ],
  };
}
