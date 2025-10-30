import { Runtime } from "../runtime";

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
