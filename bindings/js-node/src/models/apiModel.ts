const openAIModelIds = [
  "o4-mini",
  "o3",
  "o3-pro",
  "o3-mini",
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
] as const;
const geminiModelIds = [
  "gemini-2.5-flash",
  "gemini-2.5-pro",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-1.5-pro",
] as const;
const claudeModelIds = [
  "claude-sonnet-4-20250514",
  "claude-3-7-sonnet-20250219",
  "claude-3-5-sonnet-20241022",
  "claude-3-5-sonnet-20240620",
  "claude-opus-4-20250514",
  "claude-3-opus-20240229",
  "claude-3-5-haiku-20241022",
  "claude-3-haiku-20240307",
] as const;

export type APIModelId =
  | (typeof openAIModelIds)[number]
  | (typeof geminiModelIds)[number]
  | (typeof claudeModelIds)[number]
  | (string & {});

export type APIModelProvider = "openai" | "gemini" | "claude";

interface APIModelArgs {
  id: APIModelId;
  provider?: APIModelProvider;
  apiKey: string;
}

export class APIModel {
  id: APIModelId;
  provider: APIModelProvider;
  apiKey: string;
  readonly componentType: string;

  constructor(args: APIModelArgs) {
    this.id = args.id;
    this.apiKey = args.apiKey;
    if (args.provider === undefined) {
      if (openAIModelIds.includes(this.id as any)) {
        args.provider = "openai";
      } else if (geminiModelIds.includes(this.id as any)) {
        args.provider = "gemini";
      } else if (claudeModelIds.includes(this.id as any)) {
        args.provider = "claude";
      } else {
        throw Error(
          `Failed to infer the model provider. Please provide an explicit model provider.`
        );
      }
    }
    this.provider = args.provider;

    this.componentType = args.provider;
  }

  toAttrs() {
    return {
      model: this.id,
      api_key: this.apiKey,
    };
  }
}

export default function (args: APIModelArgs): APIModel {
  return new APIModel(args);
}
