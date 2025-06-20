const knownGeminiModelIds = [
  "gemini-2.5-flash",
  "gemini-2.5-pro",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-1.5-pro",
] as const;

export type GeminiModelId =
  | (typeof knownGeminiModelIds)[number]
  | (string & {});

interface GeminiModelArgs {
  id: GeminiModelId;
  apiKey: string;
}

export class GeminiModel {
  id: GeminiModelId;
  apiKey: string;
  readonly componentType: string = "gemini";

  constructor(args: GeminiModelArgs) {
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

export default function (args: GeminiModelArgs): GeminiModel {
  return new GeminiModel(args);
}
