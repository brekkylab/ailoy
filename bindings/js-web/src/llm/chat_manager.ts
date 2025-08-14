import { Message, ToolDescription } from "../agent";
import { Runtime } from "../runtime";
import { joinPath, readOPFSFile } from "../utils/opfs";

export class ChatManager {
  private name: string;
  private runtime: Runtime;
  private model: string;
  private quantization: string;
  private cacheScope: string = "ailoy";

  private bosToken?: string;
  private eosToken?: string;
  private botcToken?: string;
  private eotcToken?: string;

  constructor(runtime: Runtime, model: string, quantization: string) {
    this.name = runtime.generateUUID();
    this.runtime = runtime;
    this.model = model;
    this.quantization = quantization;
  }

  async init(modelPath: string) {
    // Call runtime to define chat manager component
    const result = await this.runtime.define("chat_manager", this.name, {
      model: this.model,
      quantiation: this.quantization,
    });

    const configFilePath = joinPath(
      this.cacheScope,
      modelPath,
      "chat-template-config.json"
    );

    const chatTemplateConfig = (await readOPFSFile(configFilePath, "json")) as {
      template_file: string;
      bos_token: string;
      eos_token: string;
      botc_token: string;
      eotc_token: string;
    };

    this.bosToken = chatTemplateConfig.bos_token;
    this.eosToken = chatTemplateConfig.eos_token;
    this.botcToken = chatTemplateConfig.botc_token || undefined;
    this.eotcToken = chatTemplateConfig.eotc_token || undefined;

    if (!result) throw Error(`chat manager component define failed`);
  }

  setCacheScope(cacheScope: string) {
    this.cacheScope = cacheScope;
  }

  async applyChatTemplate(
    /** The user message to send to the model */
    messages: Message[],
    tools: { type: "function"; function: ToolDescription }[],
    /** If True, enables reasoning capabilities (default: False) */
    reasoning?: boolean
  ): Promise<string> {
    const res = await this.runtime.callMethod(
      this.name,
      "apply_chat_template",
      {
        messages,
        tools,
        reasoning,
      }
    );
    return res.result;
  }

  getBosToken(): string | undefined {
    return this.bosToken;
  }

  getEosToken(): string | undefined {
    return this.eosToken;
  }

  getBotcToken(): string | undefined {
    return this.botcToken;
  }

  getEotcToken(): string | undefined {
    return this.eotcToken;
  }

  isBosToken(token: string): boolean {
    return this.bosToken !== undefined && token === this.bosToken;
  }

  isEosToken(token: string): boolean {
    return this.eosToken !== undefined && token === this.eosToken;
  }

  isBotcToken(token: string): boolean {
    return this.botcToken !== undefined && token === this.botcToken;
  }

  isEotcToken(token: string): boolean {
    return this.eotcToken !== undefined && token === this.eotcToken;
  }

  async dispose() {
    // Call runtime to delete tokenizer component
    const result = await this.runtime.delete(this.name);
    if (!result) throw Error(`tokenizer component delete failed`);
  }
}
