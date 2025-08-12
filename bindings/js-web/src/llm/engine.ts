import { MessageOutput } from "../agent";
import * as tvmjs from "../tvmjs";
import { joinPath, readOPFSFile } from "../utils/opfs";
import {
  ChatConfig,
  GenerationConfig,
  ModelType,
  postInitAndCheckGenerationConfigValues,
  prebuiltAppConfig,
} from "./config";
import { EmbeddingPipeline } from "./embedding";
import {
  DeviceLostError,
  FeatureSupportError,
  FetchFileFromURLError,
  MissingModelWasmError,
  ShaderF16SupportError,
  WebGPUNotAvailableError,
} from "./error";
import { LLMChatPipeline } from "./llm_chat";
import { CustomLock, findModelRecord } from "./support";
import { Tokenizer } from "./tokenizer";

const appConfig = prebuiltAppConfig;

// const logger: (msg: string) => void = log.info;

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

export class Engine {
  private modelId: string;
  private modelPath: string | undefined = undefined;
  private cacheScope: string = "ailoy";

  private chatConfig: ChatConfig | undefined = undefined;
  private pipeline: LLMChatPipeline | EmbeddingPipeline | undefined = undefined;
  private lock: CustomLock | undefined = undefined;

  // ailoy related
  private tokenizer: Tokenizer;

  constructor(modelId: string, tokenizer: Tokenizer) {
    this.modelId = modelId;
    this.tokenizer = tokenizer;
  }

  async loadModel() {
    const modelRecord = findModelRecord(this.modelId, appConfig);
    this.modelPath = modelRecord.model;

    this.chatConfig = {
      ...((await readOPFSFile(
        joinPath(this.cacheScope, this.modelPath, "mlc-chat-config.json"),
        "json"
      )) as any),
      ...modelRecord.overrides,
    } as ChatConfig;

    const wasmUrl = modelRecord.model_lib;
    if (wasmUrl === undefined) {
      throw new MissingModelWasmError(modelRecord.model_id);
    }
    const wasmSource = (await readOPFSFile(
      joinPath(this.cacheScope, this.modelPath, wasmUrl),
      "arraybuffer"
    )) as ArrayBuffer;
    if (wasmSource === undefined) {
      throw new FetchFileFromURLError(wasmUrl);
    }

    const tvm = await tvmjs.instantiate(
      wasmSource,
      tvmjs.createPolyfillWASI(),
      // logger
      console.log
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
    // gpuDetectOutput.device.lost.then((info: any) => {
    //   throw new DeviceLostError();
    // });
    tvm.initWebGPU(gpuDetectOutput.device);

    await tvm.fetchNDArrayCache(
      this.modelPath,
      tvm.webgpu(),
      this.cacheScope,
      "opfs"
    );

    if (modelRecord.model_type === ModelType.embedding) {
      this.pipeline = new EmbeddingPipeline(
        tvm,
        this.tokenizer,
        this.chatConfig
      );
    } else {
      this.pipeline = new LLMChatPipeline(
        tvm,
        this.tokenizer,
        this.chatConfig
        // logitProcessor
      );
    }
    await this.pipeline.asyncLoadWebGPUPipelines();

    this.lock = new CustomLock();
  }

  async inferEM(prompt: string) {
    try {
      const pipeline = this.pipeline as EmbeddingPipeline;

      await this.lock!.acquire();

      // 1. Call EmbeddingPipeline to get embeddings
      const embedResult: Array<Array<number>> =
        await pipeline.embedStep(prompt);

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
    input: string,
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

    function processChunk(chunk: ChatCompletionChunk): MessageOutput {
      const choice = chunk.choices.at(0);
      return {
        message: {
          role: "assistant",
          content: choice.delta.content,
          reasoning: [],
        },
        finish_reason: choice.finish_reason,
      };
    }

    // 2. Auto-regressive loop
    let curChunk;
    try {
      await this.prefill(input, pipeline, chatConfig, genConfig);
      curChunk = await _getChunk(pipeline); // prefill produces a chunk
    } catch (err) {
      await this.lock!.release();
      throw err;
    }
    if (curChunk) {
      yield processChunk(curChunk);
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
        yield processChunk(curChunk);
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
    } as ChatCompletionChunk;
    await this.lock?.release();
    yield processChunk(lastChunk);
  }

  async inferLM(input: string, genConfig: GenerationConfig) {
    if (
      this.pipeline === undefined ||
      this.chatConfig === undefined ||
      this.lock === undefined
    )
      return;

    const timeReceived = Date.now();
    await this.lock.acquire();

    return this.asyncGenerate(
      input,
      this.modelId,
      this.pipeline as LLMChatPipeline,
      this.chatConfig!,
      genConfig,
      timeReceived
    );

    // no scenario for `stream=false`
  }

  async prefill(
    input: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig
  ) {
    return await pipeline.prefillStep(input, genConfig);
  }

  async decode(pipeline: LLMChatPipeline, genConfig?: GenerationConfig) {
    return await pipeline.decodeStep(genConfig);
  }

  async dispose() {
    this.pipeline?.dispose();
    await this.tokenizer.dispose();
  }
}
