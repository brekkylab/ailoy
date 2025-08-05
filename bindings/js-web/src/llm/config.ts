/* eslint-disable @typescript-eslint/no-non-null-assertion */
import { MinValueError, NonNegativeError, RangeError } from "./error";

/**
 * Conversation template config
 */
export interface ConvTemplateConfig {
  system_template: string;
  system_message: string;
  // roles: Record<Role, string>;
  // role_templates?: Partial<Record<Role, string>>;
  seps: Array<string>;
  role_content_sep?: string;
  role_empty_sep?: string;
  stop_str: Array<string>;
  system_prefix_token_ids?: Array<number>;
  stop_token_ids: Array<number>;
  add_role_after_system_message?: boolean;
}

/**
 * Place holders that can be used in role templates.
 * For example, a role template of
 * `<<question>> ${MessagePlaceholders.USER} <<function>> ${MessagePlaceholders.FUNCTION}`
 * will insert the user message to ${MessagePlaceholders.USER}
 * and insert the function message to ${MessagePlaceholders.FUNCTION}
 * at run time.
 */
export enum MessagePlaceholders {
  system = "{system_message}",
  user = "{user_message}",
  assistant = "{assistant_message}",
  tool = "{tool_message}",
  function = "{function_string}",
  hermes_tools = "{hermes_tools}",
}

/**
 * Information about the tokenizer. Currently, only `token_postproc_method` is used to
 * post process the token table when using grammar.
 */
export interface TokenizerInfo {
  token_postproc_method: string;
  prepend_space_in_encode: boolean;
  strip_space_in_decode: boolean;
}

/**
 * Config of one chat model, a data structure representing `mlc-chat-config.json`.
 * This only corresponds to the chat-related fields and `tokenizer_files` of `mlc-chat-config.json`.
 * Only these fields affect the conversation in runtime.
 * i.e. The third part in https://llm.mlc.ai/docs/get_started/mlc_chat_config.html.
 *
 * This is initialized in `MLCEngine.reload()` with the model's `mlc-chat-config.json`.
 */
export interface ChatConfig {
  // First three fields affect the entire conversation, i.e. used in `MLCEngine.reload()`
  tokenizer_files: Array<string>;
  tokenizer_info?: TokenizerInfo;
  token_table_postproc_method?: string; // TODO: backward compatibility, remove soon
  vocab_size: number;
  // conv_config?: Partial<ConvTemplateConfig>;
  conv_template: ConvTemplateConfig;
  // KVCache settings
  context_window_size: number;
  sliding_window_size: number;
  attention_sink_size: number;
  // Fields below can be swapped per-generation via `GenerationConfig`
  // Fields only used in MLC
  repetition_penalty: number;
  // Fields shared by MLC and OpenAI APIs
  frequency_penalty: number;
  presence_penalty: number;
  top_p: number;
  temperature: number;
  bos_token_id?: number;
}

/**
 * Custom options that can be used to override known config values.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface ChatOptions extends Partial<ChatConfig> {}

/**
 * Config for a single generation.
 * Essentially `ChatConfig` without `tokenizer_files`, `conv_config`, or `conv_template`.
 * We also support additional fields not present in `mlc-chat-config.json` due to OpenAI-like APIs.
 *
 * Note that all values are optional. If unspecified, we use whatever values in `ChatConfig`
 * initialized during `MLCEngine.reload()`.
 */
export interface GenerationConfig {
  // Only used in MLC
  repetition_penalty?: number;
  ignore_eos?: boolean;
  // Shared by MLC and OpenAI APIs
  top_p?: number | null;
  temperature?: number | null;
  // Only in OpenAI APIs
  max_tokens?: number | null;
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
  stop?: string | null | Array<string>;
  n?: number | null;
  logit_bias?: Record<string, number> | null;
  // extra_body in ChatCompletionsRequest
  // enable_thinking?: boolean | null;
}

export function postInitAndCheckGenerationConfigValues(
  config: GenerationConfig
): void {
  function _hasValue(value: any): boolean {
    // if we use `if value` directly, `value` being 0 evaluates to false, violating semantics
    return value !== undefined && value !== null;
  }
  if (
    config.frequency_penalty &&
    (config.frequency_penalty < -2.0 || config.frequency_penalty > 2.0)
  ) {
    throw new RangeError("frequency_penalty", -2.0, 2.0);
  }
  if (
    config.presence_penalty &&
    (config.presence_penalty < -2.0 || config.presence_penalty > 2.0)
  ) {
    throw new RangeError("presence_penalty", -2.0, 2.0);
  }
  if (_hasValue(config.repetition_penalty) && config.repetition_penalty! <= 0) {
    throw new MinValueError("repetition_penalty", 0);
  }
  if (_hasValue(config.max_tokens) && config.max_tokens! <= 0) {
    throw new MinValueError("max_tokens", 0);
  }
  if ((_hasValue(config.top_p) && config.top_p! <= 0) || config.top_p! > 1) {
    throw new RangeError("top_p", 0, 1);
  }
  if (_hasValue(config.temperature) && config.temperature! < 0) {
    throw new NonNegativeError("temperature");
  }
}

export enum ModelType {
  "LLM",
  "embedding",
  "VLM", // vision-language model
}

/**
 * Information for a model.
 * @param model: the huggingface link to download the model weights, accepting four formats:
 *    - https://huggingface.co/{USERNAME}/{MODEL}, which we automatically use the main branch
 *    - https://huggingface.co/{USERNAME}/{MODEL}/, which we automatically use the main branch
 *    - https://huggingface.co/{USERNAME}/{MODEL}/resolve/{BRANCH}
 *    - https://huggingface.co/{USERNAME}/{MODEL}/resolve/{BRANCH}/
 * @param model_id: what we call the model.
 * @param model_lib: link to the model library (wasm file) the model uses.
 * @param overrides: partial ChatConfig to override mlc-chat-config.json; can be used to change KVCache settings.
 * @param vram_required_MB: amount of vram in MB required to run the model (can use
 *    `utils/vram_requirements` to calculate).
 * @param low_resource_required: whether the model can run on limited devices (e.g. Android phone).
 * @param buffer_size_required_bytes: required `maxStorageBufferBindingSize`, different for each device.
 * @param required_features: feature needed to run this model (e.g. shader-f16).
 * @param model_type: the intended usecase for the model, if unspecified, default to LLM.
 */
export interface ModelRecord {
  model: string;
  model_id: string;
  model_lib: string;
  overrides?: ChatOptions;
  vram_required_MB?: number;
  low_resource_required?: boolean;
  buffer_size_required_bytes?: number;
  required_features?: Array<string>;
  model_type?: ModelType;
}

/**
 * Extra configuration that can be
 * passed to the load.
 *
 * @param model_list: models to be used.
 * @param useIndexedDBCache: if true, will use IndexedDBCache to cache models and other artifacts.
 * If false or unspecified, will use the Cache API. For more information of the two, see:
 * https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria#what_technologies_store_data_in_the_browser
 *
 * @note Note that the Cache API is more well-tested in WebLLM as of now.
 */
export interface AppConfig {
  model_list: Array<ModelRecord>;
  useIndexedDBCache?: boolean;
}

/**
 * Default models and model library mapping to be used if unspecified.
 *
 * @note This is the only source of truth of which prebuilt model libraries are compatible with the
 * current WebLLM npm version.
 */
export const prebuiltAppConfig: AppConfig = {
  useIndexedDBCache: false,
  model_list: [
    // Qwen-3
    {
      model: "tvm-models/Qwen--Qwen3-0.6B/q4f16_1/",
      // model: "Qwen3-0.6B-q4f16_1-MLC/resolve/main/",
      model_id: "Qwen/Qwen3-0.6B",
      model_lib: "lib-wasm32-Emscripten-webgpu.wasm",
      // model_lib:
      //   "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_48/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
      vram_required_MB: 1403.34,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "tvm-models/Qwen--Qwen3-1.7B/q4f16_1",
      model_id: "Qwen/Qwen3-1.7B",
      model_lib: "lib-wasm32-Emscripten-webgpu.wasm",
      vram_required_MB: 2036.66,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "tvm-models/Qwen--Qwen3-4B/q4f16_1",
      model_id: "Qwen/Qwen3-4B",
      model_lib: "lib-wasm32-Emscripten-webgpu.wasm",
      vram_required_MB: 3431.59,
      low_resource_required: true,
      overrides: {
        context_window_size: 4096,
      },
    },
    {
      model: "tvm-models/Qwen--Qwen3-8B/q4f16_1",
      model_id: "Qwen/Qwen3-8B",
      model_lib: "lib-wasm32-Emscripten-webgpu.wasm",
      vram_required_MB: 5695.78,
      low_resource_required: false,
      overrides: {
        context_window_size: 4096,
      },
    },
    // Embedding models
    // -b means max_batch_size this model allows. The smaller it is, the less memory the model consumes.
    {
      model: "tvm-models/BAAI--bge-m3/q4f16_1",
      model_id: "BAAI/bge-m3",
      model_lib: "lib-wasm32-Emscripten-webgpu.wasm",
      vram_required_MB: 1407.51,
      model_type: ModelType.embedding,
    },
  ],
};
