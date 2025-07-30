import init_minijinja, { Environment } from "minijinja-js/dist/web";
import * as tvmjs from "@mlc-ai/web-runtime";
import { Tokenizer } from "@mlc-ai/web-tokenizers";
import log from "loglevel";
import { LLMChatPipeline } from "./llm_chat";
import { EmbeddingPipeline } from "./embedding";
import { CustomLock, findModelRecord } from "./support";
import {
  ChatConfig,
  GenerationConfig,
  ModelType,
  postInitAndCheckGenerationConfigValues,
  prebuiltAppConfig,
} from "./config";
import {
  DeviceLostError,
  FeatureSupportError,
  FetchFileFromURLError,
  MissingModelWasmError,
  ShaderF16SupportError,
  UnsupportedTokenizerFilesError,
  WebGPUNotAvailableError,
} from "./error";

const appConfig = prebuiltAppConfig;

const logger: (msg: string) => void = log.info;

const baseUrl = "https://models.download.ailoy.co/";

const fetchFromUrl = async (
  url: string,
  to: "arraybuffer" | "text" | "json" = "arraybuffer",
  base: string = baseUrl
) => {
  let url_inst = url.startsWith("http") ? new URL(url) : new URL(url, base);
  const body = await fetch(url_inst.href);
  if (to === "json") return body.json();
  else if (to === "text") return body.text();
  else return body.arrayBuffer();
};

export interface Embedding {
  embedding: Array<number>;
  index: number;
  object: "embedding";
}

export interface ChatCompletionChunk {
  id: string;
  choices: Array<any>;
  created: number;
  model: string;
  object: "chat.completion.chunk";
}

export class TextContent {
  constructor(public type: "text", public text: string) {}
}

export interface FunctionData {
  type: "function";
  id?: string;
  function: {
    name: string;
    arguments: any;
  };
}

export interface SystemMessage {
  role: "system";
  content: string | Array<TextContent>;
}

export interface UserMessage {
  role: "user";
  content: string | Array<TextContent>;
}

export interface AssistantMessage {
  role: "assistant";
  content?: string | Array<TextContent>;
  name?: string;
  tool_calls?: Array<FunctionData>;

  // Non-OpenAI fields
  reasoning?: Array<{ type: "text"; text: string }>;
}

export interface ToolMessage {
  role: "tool";
  content: string | Array<{ type: "text"; text: string }>;
  tool_call_id?: string;
}

export type Message =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolMessage;

export class Engine {
  private templateEnv: Environment | undefined = undefined;
  private modelId: string;
  private modelUrl: string | undefined = undefined;
  private modelType: ModelType | undefined = undefined;
  private chatConfig: ChatConfig | undefined = undefined;
  private pipeline: LLMChatPipeline | EmbeddingPipeline | undefined = undefined;
  private lock: CustomLock | undefined = undefined;

  constructor(modelId: string = "Qwen/Qwen3-0.6B") {
    this.modelId = modelId;
  }

  async getTokenizer(config: ChatConfig) {
    if (config.tokenizer_files.includes("tokenizer.json")) {
      const url = new URL("tokenizer.json", this.modelUrl).href;
      const tokenizer_json = await fetchFromUrl(url);
      if (tokenizer_json === undefined) {
        throw new FetchFileFromURLError(url);
      }
      console.log(
        `tokenizer.json(size: ${tokenizer_json.byteLength}) downloaded. `
      );
      return Tokenizer.fromJSON(tokenizer_json);
    }
    throw new UnsupportedTokenizerFilesError(config.tokenizer_files);
  }

  async loadModel() {
    console.log("START: Model Load");
    await init_minijinja();
    this.templateEnv = new Environment();
    this.templateEnv.enablePyCompat();

    const modelRecord = findModelRecord(this.modelId, appConfig);
    this.modelUrl = new URL(modelRecord.model, baseUrl).href;
    this.modelType =
      modelRecord.model_type === undefined || modelRecord.model_type === null
        ? ModelType.LLM
        : modelRecord.model_type;

    const template_filename = modelRecord.model_id.replace("/", "--") + ".j2";
    console.log("Download template file:", template_filename);
    const templateContent = await fetchFromUrl(
      template_filename,
      "text",
      this.modelUrl
    );

    console.log("Template file:", templateContent.slice(0, 100));

    this.templateEnv.addTemplate(this.modelId, templateContent);

    this.chatConfig = {
      ...(await fetchFromUrl("mlc-chat-config.json", "json", this.modelUrl)),
      ...modelRecord.overrides,
    } as ChatConfig;

    const wasmUrl = modelRecord.model_lib;
    // const wasmUrl =
    //   "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_48/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm";
    if (wasmUrl === undefined) {
      throw new MissingModelWasmError(modelRecord.model_id);
    }
    const wasmSource = await fetchFromUrl(
      wasmUrl,
      "arraybuffer",
      this.modelUrl
    );
    if (wasmSource === undefined) {
      throw new FetchFileFromURLError(wasmUrl);
    }

    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      tvmjs.createPolyfillWASI(),
      logger
    );

    // setting progress callback?

    // detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
    }
    let gpuLabel = "WebGPU";
    if (gpuDetectOutput.adapterInfo.description.length != 0) {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.description;
    } else {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.vendor;
    }
    if (modelRecord.required_features !== undefined) {
      for (const feature of modelRecord.required_features) {
        if (!gpuDetectOutput.device.features.has(feature)) {
          if (feature == "shader-f16") {
            throw new ShaderF16SupportError();
          }
          throw new FeatureSupportError(feature);
        }
      }
    }
    gpuDetectOutput.device.lost.then((info: any) => {
      throw new DeviceLostError();
    });
    tvm.initWebGPU(gpuDetectOutput.device);

    console.log("chatConfing:", this.chatConfig);
    // be replaced with VM function
    const tokenizer = await this.getTokenizer(this.chatConfig);

    await tvm.fetchNDArrayCache(
      this.modelUrl,
      tvm.webgpu(),
      "webllm/model",
      "cache"
    );

    if (modelRecord.model_type === ModelType.embedding) {
      this.pipeline = new EmbeddingPipeline(tvm, tokenizer, this.chatConfig);
    } else {
      this.pipeline = new LLMChatPipeline(
        tvm,
        tokenizer,
        this.chatConfig
        // logitProcessor
      );
    }
    await this.pipeline.asyncLoadWebGPUPipelines();

    this.lock = new CustomLock();

    console.log("DONE: Model Load");
  }

  async inferEM(prompt: string) {
    try {
      const pipeline = this.pipeline as EmbeddingPipeline;
      // 1. Call EmbeddingPipeline to get embeddings
      const embedResult: Array<Array<number>> = await pipeline.embedStep(
        prompt
      );

      // 2. Prepare response
      const batchSize = embedResult.length;
      const data: Array<Embedding> = [];
      for (let i = 0; i < batchSize; i++) {
        const curEmbedding: Embedding = {
          embedding: embedResult[i],
          index: i,
          object: "embedding",
        };
        data.push(curEmbedding);
      }
      return {
        data: data,
        model: this.modelId,
        object: "list",
      };
    } finally {
      await this.lock!.release();
    }
  }

  private async *asyncGenerate(
    messages: Array<Message>,
    reasoning: boolean = false,
    // request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number
  ): AsyncGenerator<any, void, void> {
    // 0. Pre-processing
    try {
      postInitAndCheckGenerationConfigValues(genConfig);
    } catch (err) {
      await this.lock!.release();
      throw err;
    }

    // 1. Helper function that generates the chunk
    const created = Date.now();
    const id = crypto.randomUUID();
    // this.interruptSignal = false;
    let prevMessageLength = 0; // to know where to start slicing the delta; does not count �

    function _countTrailingReplacementChar(curMessage: string): number {
      let cntr = 0;
      for (let i = curMessage.length - 1; i >= 0; i--) {
        if (curMessage.charAt(i) === "�") {
          cntr += 1;
        } else {
          return cntr;
        }
      }
      return cntr;
    }

    async function _getChunk(
      selectedPipeline: LLMChatPipeline
    ): Promise<ChatCompletionChunk | undefined> {
      // Remove the replacement character (U+FFFD) from the response to handle emojis.
      // Each emoji is made up of multiples of 4 tokens; when truncated, it is displayed as �, so
      // we skip this delta until a full emoji is rendered
      // TODO(Charlie): This does not consider cases of � not being emoji, need to fix with Streamer
      const curMessage = selectedPipeline.getMessage();
      const numTrailingReplacementChar =
        _countTrailingReplacementChar(curMessage);
      if (numTrailingReplacementChar % 4 !== 0) {
        return undefined;
      }

      const deltaMessage = curMessage.slice(prevMessageLength);
      prevMessageLength = curMessage.length;
      const chunk: ChatCompletionChunk = {
        id: id,
        choices: [
          {
            delta: { content: deltaMessage, role: "assistant" },
            finish_reason: null, // not finished yet
            index: 0,
            // logprobs: logprobs,
          },
        ],
        model: model,
        object: "chat.completion.chunk",
        created: created,
      };
      return chunk;
    }

    // 2. Auto-regressive loop
    let curChunk;
    try {
      await this.prefill(messages, reasoning, pipeline, chatConfig, genConfig);
      curChunk = await _getChunk(pipeline); // prefill produces a chunk
    } catch (err) {
      await this.lock!.release();
      throw err;
    }
    if (curChunk) {
      yield curChunk;
    }

    while (!pipeline.stopped()) {
      try {
        await this.decode(pipeline, genConfig);
        curChunk = await _getChunk(pipeline);
      } catch (err) {
        await this.lock!.release();
        throw err;
      }
      if (curChunk) {
        yield curChunk;
      }
    }

    pipeline.setSeed(Date.now());

    // 3. Last chunk empty marking the end
    // If function calling, use the last chunk to return tool_calls
    let finish_reason = pipeline.getFinishReason()!;

    const lastChunk = {
      id: id,
      choices: [
        {
          //   delta: isFunctionCalling
          //     ? {
          //         role: "assistant",
          //         tool_calls: tool_calls,
          //       }
          //     : {},
          delta: {},
          finish_reason: finish_reason,
          index: 0,
        },
      ],
      model: model,
      object: "chat.completion.chunk",
      created: created,
    };
    yield lastChunk;
    await this.lock!.release();
  }

  async inferLM(
    messages: Array<Message>,
    reasoning: boolean = false,
    genConfig: GenerationConfig
  ) {
    if (
      this.pipeline === undefined ||
      this.chatConfig === undefined ||
      this.lock === undefined
    )
      return;

    const timeReceived = Date.now();
    await this.lock.acquire();

    return this.asyncGenerate(
      messages,
      reasoning,
      this.modelId,
      this.pipeline as LLMChatPipeline,
      this.chatConfig!,
      genConfig,
      timeReceived
    );

    // no scenario for `stream=false`
  }

  async prefill(
    messages: Array<Message>,
    reasoning: boolean = false,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig
  ) {
    const input_str = this.applyChatTemplate(messages, reasoning);

    console.log(input_str);

    return await pipeline.prefillStep(input_str, genConfig);
  }

  async decode(pipeline: LLMChatPipeline, genConfig?: GenerationConfig) {
    return await pipeline.decodeStep(genConfig);
  }

  applyChatTemplate(
    messages: Array<Message>,
    reasoning: boolean = false
  ): string {
    const TOOLS = [
      {
        type: "function",
        function: {
          name: "multiply",
          description: "A function that multiplies two numbers",
          parameters: {
            type: "object",
            properties: {
              a: {
                type: "number",
                description: "The first number to multiply",
              },
              b: {
                type: "number",
                description: "The second number to multiply",
              },
            },
            required: ["a", "b"],
          },
        },
      },
    ];

    if (this.templateEnv === undefined) return "";
    const rendered = this.templateEnv.renderTemplate(this.modelId, {
      messages,
      add_generation_prompt: true,
      enable_thinking: reasoning,
      tools: TOOLS,
    });
    return rendered;
  }
}
