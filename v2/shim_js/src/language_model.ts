import {
  Instance,
  NDArray,
  PackedFunc,
  Scalar,
  TVMObject,
  VirtualMachine,
} from "./tvmjs";
import { init as init_tvm_runtime, TVMRuntime } from "./tvm_runtime";

const PAGE_SIZE = 16;

class KVCache {
  private tvm: Instance;
  private inner: TVMObject;
  private fKVStateClear: PackedFunc;
  private fKVStateAddSequence: PackedFunc;
  private fKVStateRemoveSequence: PackedFunc;
  private fKVStateForkSequence: PackedFunc;
  private fKVStateBeginForward: PackedFunc;
  private fKVStateEndForward: PackedFunc;
  private fKVStatePopn: PackedFunc;
  private fKVCacheGetNumAvailablePages: PackedFunc;
  private fKVCacheGetTotalSequenceLength: PackedFunc;

  constructor(rt: TVMRuntime) {
    this.tvm = rt.tvm;

    this.tvm.beginScope();
    // Determine prefill chunk size
    const prefillChunkSize = rt.metadata.prefill_chunk_size;
    const contextWindowSize = rt.metadata.context_window_size;
    const slidingWindowSize = rt.metadata.sliding_window_size;
    // const defaultPageSize = 16;
    const defaultMaxNumSequence = 1;
    const maxTotalSeqLen =
      slidingWindowSize != -1 ? slidingWindowSize : contextWindowSize;

    this.inner = this.tvm.detachFromCurrentScope(
      rt.vm.getFunction("create_tir_paged_kv_cache")(
        this.tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
        this.tvm.makeShapeTuple([maxTotalSeqLen]), // max_total_sequence_length
        this.tvm.makeShapeTuple([prefillChunkSize]), // prefill_chunk_size
        this.tvm.makeShapeTuple([PAGE_SIZE]), // page_size, hard coded for now
        this.tvm.makeShapeTuple([slidingWindowSize != -1 ? 1 : 0])
      )
    );

    this.fKVStateClear = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_clear")
    );
    this.fKVStateAddSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_add_sequence")
    );
    this.fKVStateRemoveSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_remove_sequence")
    );
    this.fKVStateForkSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_fork_sequence")
    );
    this.fKVStateForkSequence = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_fork_sequence")
    );
    this.fKVStateBeginForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_begin_forward")
    );
    this.fKVStateEndForward = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_end_forward")
    );
    this.fKVStatePopn = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.kv_state_popn")
    );
    this.fKVCacheGetNumAvailablePages = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc(
        "vm.builtin.attention_kv_cache_get_num_available_pages"
      )
    );
    this.fKVCacheGetTotalSequenceLength = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc(
        "vm.builtin.attention_kv_cache_get_total_sequence_length"
      )
    );
    this.addSequence();
    this.tvm.endScope();
  }

  clear() {
    this.fKVStateClear(this.inner);
    this.addSequence();
  }

  addSequence() {
    this.fKVStateAddSequence(
      this.inner,
      new Scalar(0, "int") /* Sequence ID */
    );
  }

  removeSequence() {
    this.fKVStateRemoveSequence(
      this.inner,
      new Scalar(0, "int") /* Sequence ID */
    );
  }

  beginForward(sequence_length: number) {
    this.fKVStateBeginForward(
      this.inner,
      this.tvm.makeShapeTuple([0]) /* Sequence ID */,
      this.tvm.makeShapeTuple([sequence_length])
    );
  }

  endForward() {
    this.fKVStateEndForward(this.inner);
  }

  popn(num_tokens: number) {
    this.fKVStatePopn(
      this.inner,
      new Scalar(0, "int") /* Sequence ID */,
      new Scalar(num_tokens, "int")
    );
  }

  getNumAvailablePages(): number {
    const rv = this.fKVCacheGetNumAvailablePages(this.inner);
    return rv;
  }

  getTotalSequenceLength(): number {
    const rv = this.fKVCacheGetTotalSequenceLength(this.inner);
    return rv;
  }

  getCache(): TVMObject {
    return this.inner;
  }
}

export class LanguageModel {
  private rt: TVMRuntime;
  private tvm: Instance;
  private kvcache: KVCache;
  private prefillChunkSize: number;
  private history: number[] = [];

  private fEmbed: PackedFunc;
  private fPrefill: PackedFunc;
  private fDecode: PackedFunc;
  private fSampleTopPfromLogits: PackedFunc;
  private config: {
    temperature: number;
    top_p: number;
  };

  constructor(rt: TVMRuntime) {
    this.rt = rt;
    this.tvm = this.rt.tvm;

    this.tvm.beginScope();

    this.prefillChunkSize = this.rt.metadata.prefill_chunk_size;

    this.kvcache = new KVCache(this.rt);

    this.fEmbed = this.rt.get_function("embed");
    this.fPrefill = this.rt.get_function("prefill");
    this.fDecode = this.rt.get_function("decode");
    this.fSampleTopPfromLogits = this.tvm.detachFromCurrentScope(
      this.tvm.getGlobalFunc("vm.builtin.sample_top_p_from_logits")
    );

    this.config = {
      temperature: 0.6,
      top_p: 0.9,
    };

    this.tvm.endScope();
  }

  private clear() {
    this.kvcache.clear();
    this.history = [];
  }

  async prefill(tokens: Uint32Array) {
    if (!tokens || tokens.length === 0) {
      throw new Error("Token must not be empty");
    }

    this.tvm.beginScope();

    // Make sure that kv-cache and history is sync
    if (this.kvcache.getTotalSequenceLength() != this.history.length)
      this.clear();

    // The longest common prefix (LCP) between inputs & previous conversations
    let lcp_index = 0;
    while (lcp_index < this.history.length && lcp_index < tokens.length) {
      if (this.history[lcp_index] != tokens[lcp_index]) break;
      ++lcp_index;
    }

    // Rewind the head of kv-cache to the LCP
    if (lcp_index < this.history.length) {
      this.kvcache.popn(this.history.length - lcp_index);
    }

    // Tokens to be added (wihout common prefixes)
    const new_tokens = tokens.slice(lcp_index);
    if (new_tokens.length == 0) return;

    // Calculate remaining space in KV cache
    if (new_tokens.length >= this.kvcache.getNumAvailablePages() * PAGE_SIZE)
      throw Error("Context length limit exceed");

    // Chunk size to be split
    const prefillChunkSize = this.prefillChunkSize;
    for (let i = 0; i < new_tokens.length; i += prefillChunkSize) {
      // Prefill i to j
      const j = Math.min(i + prefillChunkSize, new_tokens.length);
      const length = j - i;
      const current_new_tokens = new Int32Array(
        new_tokens.buffer,
        new_tokens.byteOffset + i * 4,
        length
      );

      // Input NDArray<int32>[length]
      const input_cpu = this.tvm.withNewScope(() => {
        return this.tvm.detachFromCurrentScope(
          this.tvm.empty([length], "int32", this.tvm.cpu())
        );
      });
      input_cpu.copyFrom(current_new_tokens);
      const input = this.tvm.withNewScope(() => {
        return this.tvm.detachFromCurrentScope(
          this.tvm.empty([length], "int32", this.rt.device)
        );
      });
      // Copy from new_tokens[i..j) – reinterpret as Int32Array
      input.copyFrom(input_cpu);
      await this.rt.device.sync();

      // Embedding of the input: [T, D]
      const embedding = this.fEmbed(input, this.rt.params) as NDArray;

      // Reshape to [1, T, D]
      const embedding_reshaped = embedding.view([
        1,
        embedding.shape[0],
        embedding.shape[1],
      ]);

      // Forward prefill
      this.kvcache.beginForward(length);
      this.fPrefill(
        embedding_reshaped,
        this.kvcache.getCache(),
        this.rt.params
      );
      this.kvcache.endForward();
    }

    // Update history
    this.history = Array.from(tokens);

    this.tvm.endScope();
  }

  async decode(last_token: number): Promise<Float32Array> {
    if (this.kvcache.getNumAvailablePages() < 1) {
      throw new Error("Context length limit exceed");
    }

    this.tvm.beginScope();

    // Input NDArray<int32>[1]
    const input = this.tvm.empty([1], "int32", this.rt.device);
    const buf = new Int32Array(1);
    buf[0] = last_token;
    input.copyFrom(buf);

    // Embed
    const embed = this.fEmbed(input, this.rt.params) as NDArray;
    const embReshaped = embed.view([1, 1, embed.shape[1]]);

    // In decode, the sequence length of new tokens are always 1
    this.kvcache.beginForward(1);
    // Forward decode (output: [logits, kv_caches])
    const out = this.fDecode(
      embReshaped,
      this.kvcache.getCache(),
      this.rt.params
    );
    // Extract logits (1 x seq_len x vocab_size)
    // Note that the seq_len is the ID of the seqence, used for decoding multiple
    // context in parallel. In our cases, it always set to 1.
    const logits: NDArray = this.tvm.detachFromCurrentScope(out.get(0));
    const logitsCPU = this.tvm.detachFromCurrentScope(
      this.tvm.empty(logits.shape, logits.dtype, this.tvm.cpu())
    );
    logitsCPU.copyFrom(logits);
    await this.rt.device.sync();
    const rv = logitsCPU.toArray() as Float32Array;
    logits.dispose();
    logitsCPU.dispose();
    this.kvcache.endForward();
    this.tvm.endScope();
    return rv;
  }

  sample(logits: Float32Array): number {
    this.tvm.beginScope();
    const logitsNDArray = this.tvm.empty(
      [logits.length],
      "float32",
      this.tvm.cpu()
    );
    logitsNDArray.copyFrom(logits);
    const sampled_token = this.fSampleTopPfromLogits(
      logitsNDArray,
      new Scalar(this.config.temperature, "int"),
      new Scalar(this.config.top_p, "int"),
      new Scalar(Math.random(), "int")
    );
    this.tvm.endScope();
    return sampled_token;
  }
}

export async function init(
  cache_contents: Record<string, ArrayBuffer>
): Promise<LanguageModel> {
  const rt = await init_tvm_runtime(cache_contents);
  const rv = new LanguageModel(rt);
  return rv;
}
