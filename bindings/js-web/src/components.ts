import { Image } from "wasm-vips";

import { Runtime } from "./runtime";
import { Message, ToolDescription } from "./agent";

export class ChatManager {
  private name: string;
  private runtime: Runtime;
  private model: string;
  private quantization: string;

  constructor(runtime: Runtime, model: string, quantization: string) {
    this.name = runtime.generateUUID();
    this.runtime = runtime;
    this.model = model;
    this.quantization = quantization;
  }

  async init() {
    // Call runtime to define chat manager component
    const result = await this.runtime.define("chat_manager", this.name, {
      model: this.model,
      quantiation: this.quantization,
    });
    if (!result) throw Error(`chat manager component define failed`);
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
}

export class Tokenizer {
  private name: string;
  private runtime: Runtime;
  private model: string;
  private quantization: string;

  constructor(runtime: Runtime, model: string, quantization: string) {
    this.name = runtime.generateUUID();
    this.runtime = runtime;
    this.model = model;
    this.quantization = quantization;
  }

  async init() {
    // Call runtime to define tokenizer component
    const result = await this.runtime.define("tokenizer", this.name, {
      model: this.model,
      quantiation: this.quantization,
    });
    if (!result) throw Error(`tokenizer component define failed`);
  }

  async encode(text: string): Promise<Int32Array> {
    const res = await this.runtime.callMethod(this.name, "encode", {
      text,
    });
    return Int32Array.from(res.tokens ?? []);
  }

  async decode(ids: Int32Array): Promise<string> {
    const res = await this.runtime.callMethod(this.name, "decode", {
      tokens: Array.from(ids),
    });
    return res.text ?? "";
  }

  async dispose() {
    // Call runtime to delete tokenizer component
    const result = await this.runtime.delete(this.name);
    if (!result) throw Error(`tokenizer component delete failed`);
  }
}
