const knownClaudeModelIds = [
  "claude-sonnet-4-20250514",
  "claude-3-7-sonnet-20250219",
  "claude-3-5-sonnet-20241022",
  "claude-3-5-sonnet-20240620",
  "claude-opus-4-20250514",
  "claude-3-opus-20240229",
  "claude-3-5-haiku-20241022",
  "claude-3-haiku-20240307",
] as const;

export type ClaudeModelId =
  | (typeof knownClaudeModelIds)[number]
  | (string & {});

interface ClaudeModelArgs {
  id: ClaudeModelId;
  apiKey: string;
}

export class ClaudeModel {
  id: ClaudeModelId;
  apiKey: string;
  readonly componentType: string = "claude";

  constructor(args: ClaudeModelArgs) {
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

export default function (args: ClaudeModelArgs): ClaudeModel {
  return new ClaudeModel(args);
}
