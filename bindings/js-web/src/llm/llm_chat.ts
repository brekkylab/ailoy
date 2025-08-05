/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable no-prototype-builtins */
// import { Tokenizer } from "@mlc-ai/web-tokenizers";
// import * as tvmjs from "@mlc-ai/web-runtime";
import { Tokenizer } from "../agent";
import * as tvmjs from "../tvmjs";
import { ChatConfig, GenerationConfig } from "./config";
import { getChunkedPrefillInputData } from "./support";
import {
  AttentionSinkSizeError,
  ContextWindowSizeExceededError,
  MinValueError,
  RangeError,
  WindowSizeConfigurationError,
  WindowSizeSpecificationError,
} from "./error";

export type FinishReason = "stop" | "length" | "tool_calls" | "abort";

export class LLMChatPipeline {
  private config: ChatConfig;
  private tokenizer: Tokenizer;

  // TVM functions
  private tvm: tvmjs.Instance;
  private device: tvmjs.DLDevice;
  private vm: tvmjs.VirtualMachine;
  private prefill: tvmjs.PackedFunc;
  private decoding: tvmjs.PackedFunc;
  private image_embed: tvmjs.PackedFunc | undefined;
  private embed: tvmjs.PackedFunc;
  //   private fapplyBitmask: tvmjs.PackedFunc;
  // Functions related to PagedKVCache
  private fclearKVCaches: tvmjs.PackedFunc;
  private fKVCacheAddSequence: tvmjs.PackedFunc;
  private fKVCacheRemoveSequence: tvmjs.PackedFunc;
  private fKVCacheBeginForward: tvmjs.PackedFunc;
  private fKVCacheEndForward: tvmjs.PackedFunc;
  private fKVCacheEnableSlidingWindowForSeq: tvmjs.PackedFunc;

  // parameter states
  private params: tvmjs.TVMObject;
  private kvCache: tvmjs.TVMObject;
  private logitsOnCPU?: tvmjs.NDArray = undefined;
  private filledKVCacheLength = 0;

  // meta data
  private contextWindowSize = -1;
  private slidingWindowSize = -1;
  private attentionSinkSize = -1;
  private prefillChunkSize = -1;
  private stopStr: string[];
  private stopTokens: Array<number>;

  // states
  private outputMessage = "";
  private outputIds: Array<number> = [];
  private stopTriggered = false;
  // private finishReason: ChatCompletionFinishReason | undefined = undefined;
  private finishReason: FinishReason | undefined = undefined;

  constructor(tvm: tvmjs.Instance, tokenizer: Tokenizer, config: ChatConfig) {
    // 0. Setting attributes
    this.tvm = tvm;
    this.tokenizer = tokenizer;
    this.config = config;

    this.stopStr = this.config.conv_template.stop_str;
    this.stopTokens = this.config.conv_template.stop_token_ids;

    this.device = this.tvm.webgpu();

    // 1. Create VM and get the core functions
    tvm.beginScope();
    this.vm = this.tvm.detachFromCurrentScope(
      this.tvm.createVirtualMachine(this.device)
    );
    this.prefill = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("prefill")
    );
    this.embed = this.tvm.detachFromCurrentScope(this.vm.getFunction("embed"));
    this.decoding = this.tvm.detachFromCurrentScope(
      this.vm.getFunction("decode")
    );
    try {
      this.image_embed = this.tvm.detachFromCurrentScope(
        this.vm.getFunction("image_embed")
      );
    } catch {
      // log.info("Cannot find function image_embed.");
    }

    // 2. Get json stored in the vm's metadata function
    const fgetMetadata = this.vm.getFunction("_metadata");
    const ret_value = fgetMetadata();
    // const metadataStr = this.tvm.detachFromCurrentScope(ret_value).toString();
    const metadataStr = ret_value.toString();
    const metadata = JSON.parse(metadataStr);

    console.log("lib metadata: ", metadata);

    // 3. Load parameters by name
    const paramNames: string[] = [];
    metadata.params.forEach((param: any) => {
      paramNames.push(param.name);
    });
    this.params = this.tvm.detachFromCurrentScope(
      this.tvm.getParamsFromCacheByName(paramNames)
    );

    // 4. Read in compilation configurations from metadata
    this.prefillChunkSize = metadata.prefill_chunk_size;
    // log.info("Using prefillChunkSize: ", this.prefillChunkSize);
    if (this.prefillChunkSize <= 0) {
      throw new MinValueError("prefill_chunk_size", 0);
    }

    // 5. Consolidate KVCache settings: context window, sliding window, attention sink
    this.slidingWindowSize = config.sliding_window_size;
    this.contextWindowSize = config.context_window_size;
    this.attentionSinkSize = config.attention_sink_size;
    if (this.contextWindowSize !== -1 && this.slidingWindowSize !== -1) {
      throw new WindowSizeConfigurationError(
        this.contextWindowSize,
        this.slidingWindowSize
      );
    } else if (this.slidingWindowSize != -1) {
      // Use sliding window and attention sink
      // log.info("Using slidingWindowSize: ", this.slidingWindowSize);
      if (this.attentionSinkSize >= 0) {
        // log.info("Using attentionSinkSize: ", this.attentionSinkSize);
      } else {
        throw new AttentionSinkSizeError();
      }
    } else if (this.contextWindowSize != -1) {
      // Use default kv cache without sliding window
      // log.info("Using contextWindowSize: ", this.contextWindowSize);
    } else {
      throw new WindowSizeSpecificationError();
    }

    // 5. Create cache
    // Load cache functions and instantiate KVCache
    this.fclearKVCaches = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_clear")
    );
    this.fKVCacheAddSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_add_sequence")
    );
    this.fKVCacheRemoveSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_remove_sequence")
    );
    this.fKVCacheBeginForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_begin_forward")
    );
    this.fKVCacheEndForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_end_forward")
    );
    this.fKVCacheEnableSlidingWindowForSeq = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc(
        "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq"
      )
    );

    // Create PagedKVCache; we do not expose KVCache config for now
    const fcreateCache = this.vm.getFunction("create_tir_paged_kv_cache");
    const defaultPageSize = 16;
    const defaultMaxNumSequence = 1;
    const maxTotalSeqLen =
      this.slidingWindowSize != -1
        ? this.slidingWindowSize
        : this.contextWindowSize;
    this.kvCache = this.tvm.detachFromCurrentScope(
      fcreateCache(
        this.tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
        this.tvm.makeShapeTuple([maxTotalSeqLen]), // max_total_sequence_length
        this.tvm.makeShapeTuple([this.prefillChunkSize]), // prefill_chunk_size
        this.tvm.makeShapeTuple([defaultPageSize]), // page_size, hard coded for now
        this.tvm.makeShapeTuple([this.slidingWindowSize != -1 ? 1 : 0])
      )
    );

    this.filledKVCacheLength = 0;
    this.resetChat(); // especially needed for PagedKVCache as we need to call fKVCacheAddSequence
    tvm.endScope();
  }

  dispose() {
    this.params.dispose();
    this.decoding.dispose();
    this.prefill.dispose();
    this.embed.dispose();
    this.image_embed?.dispose();
    this.vm.dispose();
    this.kvCache.dispose();
    this.fclearKVCaches.dispose();
    this.logitsOnCPU?.dispose();
    this.tvm.dispose();
    this.tokenizer.dispose();
  }

  /**
   * Get the current message.
   */
  getMessage() {
    return this.outputMessage;
  }

  /**
   * Reset the chat history
   */
  //   resetChat(keepStats = false) {
  resetChat() {
    this.tvm.beginScope();
    this.resetKVCache();
    this.filledKVCacheLength = 0;
    this.tvm.endScope();
  }

  /**
   * Reset KV Cache
   */
  resetKVCache() {
    this.fclearKVCaches(this.kvCache);
    this.fKVCacheAddSequence!(this.kvCache, new tvmjs.Scalar(0, "int64"));
    if (this.slidingWindowSize != -1) {
      this.fKVCacheEnableSlidingWindowForSeq(
        this.kvCache,
        new tvmjs.Scalar(0, "int64"),
        new tvmjs.Scalar(this.slidingWindowSize, "int32"),
        new tvmjs.Scalar(this.attentionSinkSize, "int32")
      );
    }
  }

  /**
   * @returns Whether stop is triggered.
   */
  stopped(): boolean {
    return this.stopTriggered;
  }

  /**
   * @returns Finish reason; undefined if generation not started/stopped yet.
   */
  // getFinishReason(): ChatCompletionFinishReason | undefined {
  getFinishReason(): FinishReason | undefined {
    return this.finishReason;
  }

  /**
   * Set the seed for the RNG `this.tvm.rng`.
   */
  setSeed(seed: number): void {
    this.tvm.setSeed(seed);
  }

  async asyncLoadWebGPUPipelines() {
    await this.tvm.asyncLoadWebGPUPipelines(this.vm.getInternalModule());
  }

  /**
   * Generate the first token given input prompt
   */
  async prefillStep(
    input_str: string,
    genConfig?: GenerationConfig
  ): Promise<void> {
    // cleanup the per convo states
    this.outputIds = [];
    this.outputMessage = "";
    this.stopTriggered = false;

    const retGetInputData = await this.getInputData(input_str);
    const inputData: Array<Array<number>> = retGetInputData[0];
    const promptLen: number = retGetInputData[1];

    // 1. Chunk inputData to embed and forward in one shot for each, minimize intermediate data
    const retGetChunks = getChunkedPrefillInputData(
      inputData,
      this.prefillChunkSize
    );
    const chunks: Array<Array<number>>[] = retGetChunks[0];
    const chunkLens: Array<number> = retGetChunks[1];

    // 2. Prefill each chunk
    this.tvm.beginScope();
    let logits: tvmjs.NDArray;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkLen = chunkLens[i];
      const prevFilledLen = this.filledKVCacheLength;
      logits = this.tvm.detachFromCurrentScope(
        await this.embedAndForward(chunk, chunkLen)
      );
      if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
        throw new Error(
          "Internal Error: filledKVCacheLength does not match expected value."
        );
      }
    }
    this.tvm.endScope();

    // 4. Sample, stats, post process token sampled.
    // We wait for prefill and grammar matcher init to finish
    await Promise.all([this.device.sync()]);
    const nextToken = await this.sampleTokenFromLogits(logits!, genConfig);
    logits!.dispose();

    await this.processNextToken(nextToken, genConfig);
  }

  async decodeStep(genConfig?: GenerationConfig): Promise<void> {
    if (this.stopTriggered) {
      throw Error("Cannot run decode when stopped");
    }

    this.tvm.beginScope();
    const chunk: Array<Array<number>> = [
      this.outputIds.slice(this.outputIds.length - 1),
    ];
    const chunkLen = chunk.length;
    const prevFilledLen = this.filledKVCacheLength;
    const logits = this.tvm.detachFromCurrentScope(
      await this.embedAndForward(chunk, chunkLen)
    );
    if (this.filledKVCacheLength !== prevFilledLen + chunkLen) {
      throw new Error(
        "Internal Error: filledKVCacheLength does not match expected value."
      );
    }
    this.tvm.endScope();

    // sample from logits
    const nextToken = await this.sampleTokenFromLogits(logits, genConfig);
    logits.dispose();

    await this.processNextToken(nextToken, genConfig);
  }

  /**
   * Manually trigger stop if it is not stopped.
   */
  triggerStop() {
    if (this.stopTriggered) {
      return;
    }
    this.stopTriggered = true;
    this.finishReason = "abort";
  }

  /**
   * Add a generated token and check for stop.
   *
   * @param nextToken The next token.
   * @param genConfig Configs that override `this.config` for this round of generation.
   */
  private async processNextToken(
    nextToken: number,
    genConfig?: GenerationConfig
  ): Promise<void> {
    if (this.stopTriggered) {
      throw Error("Cannot call process when it is stoppped");
    }

    // Get max_tokens from generationConfig (specified by user in completion request)
    // If not specified, do not set a limit
    let max_tokens = Infinity;
    // if (genConfig !== undefined && genConfig.max_tokens) {
    //   max_tokens = genConfig.max_tokens;
    // }
    if (max_tokens <= 0) {
      throw new MinValueError("max_tokens", 0);
    }

    // Get ignore_eos from generationConfig (specified by user in completion request)
    let ignore_eos = false;
    if (
      genConfig !== undefined &&
      genConfig.ignore_eos !== undefined &&
      genConfig.ignore_eos !== null
    ) {
      ignore_eos = genConfig.ignore_eos;
    }

    // Get stopStrs, possibly overridden by genConfig for this round
    let stopStrs = this.stopStr;
    if (genConfig !== undefined && genConfig.stop) {
      stopStrs = stopStrs.concat(genConfig.stop);
    }

    let stopTokens = this.stopTokens;
    if (ignore_eos) {
      stopTokens = [];
      stopStrs = [];
    }

    // Stop condition 1: stop token; otherwise, append to `this.outputIds`
    if (stopTokens.includes(nextToken)) {
      this.stopTriggered = true;
      this.finishReason = "stop";
    }
    if (!this.stopTriggered) {
      this.outputIds.push(nextToken);
    }

    // Stop condition 2: stop string; update `this.outputMessage` subsequently
    let outputMessage = await this.tokenizer.decode(
      new Int32Array(this.outputIds)
    );
    let stopPos = -1;
    for (const stopStr of stopStrs) {
      // Stop at the first stopStr we find
      stopPos = outputMessage.lastIndexOf(stopStr);
      if (stopPos != -1) {
        outputMessage = outputMessage.substring(0, stopPos);
        this.stopTriggered = true;
        this.finishReason = "stop";
        break;
      }
    }
    this.outputMessage = outputMessage;

    // Stop condition 3: exceed max_tokens
    if (this.outputIds.length >= max_tokens) {
      this.stopTriggered = true;
      this.finishReason = "length";
      // log.info("Generation stopped due to exceeding max_tokens.");
    }

    // Stop condition 4: exceed KVCache's context window size
    if (
      this.slidingWindowSize == -1 &&
      this.filledKVCacheLength == this.contextWindowSize
    ) {
      this.stopTriggered = true;
      this.finishReason = "length";
      // log.info("Generation stopped due to exceeding context_window_size.");
    }
  }

  /**
   * Given input tokens, return embeddings of them by calling embed kernel.
   *
   * @note precondition: inputTokens.length <= prefillChunkSize, since we take care of
   * chunking in `getChunkedPrefillInputData()`.
   */
  private getTokensEmbeddings(inputTokens: number[]): tvmjs.NDArray {
    this.tvm.beginScope();
    if (inputTokens.length > this.prefillChunkSize) {
      throw new Error(
        "Internal Error: getTokensEmbeddings input should be <= prefillChunkSize."
      );
    }
    const inputData = this.tvm.empty(
      [inputTokens.length],
      "int32",
      this.device
    );
    inputData.copyFrom(inputTokens);
    const embed: tvmjs.NDArray = this.tvm.detachFromCurrentScope(
      this.embed!(inputData, this.params)
    );
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(embed); // tracked by scope of embedAndForward
    return embed;
  }

  /**
   * Embed and forward input data, that can be either array of tokens, or an image.
   * This will increment `this.filledKVCacheLength`.
   *
   * @param inputData data to embed and forward
   * @param inputDataLen length of this inputData, should smaller than prefill chunk size.
   * @returns The logits returned by this forward as tvmjs.NDArray on GPU.
   *
   * @note Precondition: inputData's data length is smaller than prefill chunk size
   */
  private async embedAndForward(
    // inputData: Array<Array<number> | ImageURL>,
    inputData: Array<Array<number>>,
    inputDataLen: number
  ): Promise<tvmjs.NDArray> {
    if (inputDataLen > this.prefillChunkSize) {
      throw new Error(
        "InternalError: expect inputDataLen <= this.prefillChunkSize."
      );
    }
    // TODO: we should combine string data to embed once, then rearrange the embeddings; currently
    // ["hi", imageUrl, "hi"] would call embed kernels 3 times, while 2 would suffice.

    // 1. Embed all inputData
    this.tvm.beginScope();
    const embeddings: tvmjs.NDArray[] = [];
    for (let i = 0; i < inputData.length; i++) {
      const data = inputData[i];
      //   if (Array.isArray(data)) {
      embeddings.push(this.getTokensEmbeddings(data));
      // } else // no image treatment
    }

    // 2. Concatenate embeddings
    let allEmbeddings: tvmjs.NDArray;
    if (embeddings.length === 1) {
      allEmbeddings = embeddings[0];
    } else {
      allEmbeddings = this.tvm.concatEmbeddings(embeddings);
    }
    if (inputDataLen !== allEmbeddings.shape[0]) {
      throw new Error("InternalError: expect seqLen == allEmbeddings.shape[0]");
    }
    allEmbeddings = allEmbeddings.view([1].concat(allEmbeddings.shape));
    // TODO: Should we end this scope here and begin another scope? Will this dispose embeddings to
    // save RAM? We will detach allEmbeddings from this scope and attach to the next scope.

    // 3. Forward the concatenated embeddings
    const inputLenShape = this.tvm.makeShapeTuple([inputDataLen]);
    const seqIdsTuple = this.tvm.makeShapeTuple([0]);
    this.fKVCacheBeginForward!(this.kvCache, seqIdsTuple, inputLenShape);
    let retValue;
    if (inputDataLen > 1) {
      retValue = this.prefill(allEmbeddings, this.kvCache, this.params);
    } else {
      retValue = this.decoding(allEmbeddings, this.kvCache, this.params);
    }

    // Epilogue
    this.fKVCacheEndForward!(this.kvCache);
    this.filledKVCacheLength += inputDataLen;
    const logits = this.tvm.detachFromCurrentScope(retValue.get(0));
    this.tvm.endScope();
    this.tvm.attachToCurrentScope(logits);
    return logits;
  }

  // NOTE: caller must call device.sync()
  private updateLogitsOnCPU(logits: tvmjs.NDArray): tvmjs.NDArray {
    if (this.logitsOnCPU == undefined) {
      this.logitsOnCPU = this.tvm.detachFromCurrentScope(
        this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu())
      );
    } else {
      if (logits.shape[0] != this.logitsOnCPU.shape[0]) {
        throw Error("We expect the size of logits to remain unchanged");
      }
    }
    this.logitsOnCPU.copyFrom(logits);
    return this.logitsOnCPU;
  }

  private async sampleTokenFromLogits(
    logitsOnGPU: tvmjs.NDArray,
    genConfig?: GenerationConfig
  ) {
    function _hasValue(value: any): boolean {
      // if we use `if value` directly, `value` being 0 evaluates to false, violating semantics
      return value !== undefined && value !== null;
    }
    let temperature: number = this.config.temperature;
    let top_p: number = this.config.top_p;
    let repetition_penalty: number = this.config.repetition_penalty;
    let frequency_penalty: number = this.config.frequency_penalty;
    let presence_penalty: number = this.config.presence_penalty;

    if (genConfig !== undefined) {
      if (_hasValue(genConfig.temperature)) {
        temperature = genConfig.temperature!;
      }
      if (_hasValue(genConfig.top_p)) {
        top_p = genConfig.top_p!;
      }
      if (_hasValue(genConfig.repetition_penalty)) {
        repetition_penalty = genConfig.repetition_penalty!;
      }
      if (_hasValue(genConfig.frequency_penalty)) {
        frequency_penalty = genConfig.frequency_penalty!;
      }
      if (_hasValue(genConfig.presence_penalty)) {
        presence_penalty = genConfig.presence_penalty!;
      }
      // If only one of frequency or presence penatly is set, make the other one 0.0
      if (_hasValue(frequency_penalty) && !_hasValue(presence_penalty)) {
        presence_penalty = 0.0;
      }
      if (_hasValue(presence_penalty) && !_hasValue(frequency_penalty)) {
        frequency_penalty = 0.0;
      }
    }
    // Check range validity
    if (top_p <= 0 || top_p > 1) {
      throw new RangeError("top_p", 0, 1);
    }
    if (temperature < 0) {
      throw new MinValueError("temperature", 0);
    }
    if (repetition_penalty <= 0) {
      throw new MinValueError("repetition_penalty", 0);
    }
    if (
      frequency_penalty &&
      (frequency_penalty < -2.0 || frequency_penalty > 2.0)
    ) {
      throw new RangeError("frequency_penalty", -2.0, 2.0);
    }
    if (
      presence_penalty &&
      (presence_penalty < -2.0 || presence_penalty > 2.0)
    ) {
      throw new RangeError("presence_penalty", -2.0, 2.0);
    }
    this.tvm.beginScope();
    this.updateLogitsOnCPU(logitsOnGPU);
    this.tvm.endScope();
    await this.device.sync();

    if (this.logitsOnCPU == undefined) {
      throw Error("logits should be assigned");
    }

    // If logprobs, need the actual distribution via softmax, otherwise directly sample from logits
    let sampledToken: number;
    sampledToken = this.tvm.sampleTopPFromLogits(
      this.logitsOnCPU,
      temperature,
      top_p
    );

    return sampledToken;
  }

  // private getInputData(): [Array<Array<number> | ImageURL>, number] {
  private async getInputData(
    input_str: string
  ): Promise<[Array<Array<number>>, number]> {
    const ret: Array<Array<number>> = [];
    let curTokens: Array<number> = [];
    let prompts: Array<string | Array<string>>;

    prompts = [input_str];

    let numPromptTokens = 0;
    for (let i = 0; i < prompts.length; i++) {
      const curPrompt = prompts[i];
      if (typeof curPrompt === "string") {
        const encoded = await this.tokenizer.encode(curPrompt);
        numPromptTokens += encoded.length;
        curTokens.push(...encoded);
      } else {
        for (let j = 0; j < curPrompt.length; j++) {
          const curPromptContent: string = curPrompt[j];
          // if (typeof curPromptContent === "string") {
          const encoded = await this.tokenizer.encode(curPromptContent);
          numPromptTokens += encoded.length;
          curTokens.push(...encoded);
          // }  else  // no image treatment
        }
      }
    }
    // Deal with last curTokens
    if (curTokens.length !== 0) {
      ret.push([...curTokens]);
    }

    // Check if input tokens exceed context window size
    if (
      this.slidingWindowSize == -1 && // There is no limit on contextWindowSize for sliding window
      numPromptTokens + this.filledKVCacheLength > this.contextWindowSize
    ) {
      throw new ContextWindowSizeExceededError(
        numPromptTokens,
        this.contextWindowSize
      );
    }
    return [ret, numPromptTokens];
  }

  /**
   * Synchronize the device.
   */
  async sync(): Promise<void> {
    // Is it equivalent to this.tvm.sync()?
    await this.device.sync();
  }
}
