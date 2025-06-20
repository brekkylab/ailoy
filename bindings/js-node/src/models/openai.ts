const knownOpenAIModelIds = [
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

export type OpenAIModelId =
  | (typeof knownOpenAIModelIds)[number]
  | (string & {});

interface OpenAIModelArgs {
  id: OpenAIModelId;
  apiKey: string;
}

export class OpenAIModel {
  id: OpenAIModelId;
  apiKey: string;
  readonly componentType: string = "openai";

  constructor(args: OpenAIModelArgs) {
    this.id = args.id;
    this.apiKey = args.apiKey;
  }

  toAttrs() {
    return {
      model: this.id,
      api_key: this.apiKey,
    };
  }
}

export default function (args: OpenAIModelArgs): OpenAIModel {
  return new OpenAIModel(args);
}
