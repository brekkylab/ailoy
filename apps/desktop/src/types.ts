// The engine's IPC surface, mirrored in TypeScript.
//
// Every type here is the serde shape of a Rust type in `src-tauri`/`core`: renaming a
// field on either side is a silent break that shows up as a UI that stops painting, so
// these names are copied verbatim (snake_case) rather than idiomatized. Only the
// wrappers the UI owns — `LiveRun` in `store/runs.ts` — use camelCase.

export type Role = "system" | "user" | "assistant" | "tool";

export type Part =
  | { type: "text"; text: string }
  | { type: "function"; id: string; function: { name: string; arguments: unknown } }
  | { type: "value"; value: unknown }
  // `data` is `ailoy::datatype::Bytes`, a `serde_bytes::ByteBuf`. Its schema says
  // `format: base64`, but that is for the models' wire formats; serde_json — which is
  // what Tauri's IPC uses — has no byte-string type and writes it as an array of numbers.
  // The UI does not render images yet; whatever does must read it as bytes, not decode it.
  | { type: "image"; image: { type: "embedded"; mime_type: string; data: number[] } | { type: "url"; url: string } };

export interface Message {
  role: Role;
  contents: Part[];
  thinking?: string;
  tool_calls?: Part[];
  id?: string;
  signature?: string;
}

/**
 * Token counts for one model response. The prompt-side fields are disjoint components,
 * never overlapping totals: the full prompt is
 * `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.
 */
export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}

export interface RateLimitWindow { limit: number | null; remaining: number | null; reset_at_ms: number | null }
export interface RateLimitInfo {
  requests?: RateLimitWindow | null;
  tokens?: RateLimitWindow | null;
  input_tokens?: RateLimitWindow | null;
  output_tokens?: RateLimitWindow | null;
}

/**
 * The machine-readable tag on every command rejection. Mirrors `EngineError::kind` in
 * `apps/desktop/core/src/error.rs`; the webview branches on it (offer a retry, open the
 * settings pane, refetch a stale list) and shows `message` verbatim.
 */
export type EngineErrorKind =
  | "not_found"
  | "already_running"
  | "invalid"
  | "console_unavailable"
  | "workspace"
  | "storage"
  | "io"
  | "other";

/** What a rejected `invoke` throws: always this object, never a bare string. */
export interface EngineErrorPayload { kind: EngineErrorKind; message: string }

export interface SessionSummary { id: string; title: string; model: string; created_at: number; updated_at: number; running: boolean }
export interface StoredMessage { seq: number; depth: number; source_agent: string | null; message: Message; usage: TokenUsage | null; created_at: number }
export interface SessionUsage {
  input_tokens: number; output_tokens: number; cache_read_tokens: number; cache_write_tokens: number;
  estimated_cost_usd: number | null; context_used: number | null; context_limit: number | null;
}

export type RunEvent =
  | { type: "started"; run_id: string }
  | { type: "text_delta"; text: string }
  | { type: "thinking_delta"; text: string }
  | { type: "tool_call_started"; id: string; name: string; arguments: unknown }
  | { type: "message"; seq: number; depth: number; source_agent: string | null; message: Message; usage: TokenUsage | null }
  | { type: "usage"; usage: TokenUsage | null; rate_limit: RateLimitInfo | null; context_used: number | null; context_limit: number | null }
  | { type: "awaiting_approval"; id: string; name: string; arguments: unknown }
  | { type: "done" }
  | { type: "cancelled" }
  /**
   * A run failed. `kind` is the engine's own classification — `model`, `tool`, `stream`,
   * `storage`, `console_unavailable`, `max_turns`, `internal` — and is a plain string,
   * not the `EngineErrorKind` of a rejected command. `status` and `retryable` are only
   * ever filled for `kind: "model"`: `status` is the HTTP status when a response arrived
   * at all, and `retryable` is what a retry button should be enabled by.
   */
  | { type: "error"; kind: string; message: string; status: number | null; retryable: boolean };

/** The root is a `local` mount too; what makes it the root is its path being `/`. */
export type MountKind = "local" | "notion" | "s3";
export type MountStatus = { status: "ok" } | { status: "error"; message: string };
export interface MountInfo { id: string; path: string; kind: MountKind; label: string; detail: string; writable: boolean; status: MountStatus }
/** The request side of a mount. The engine also has a `root` variant, which only it creates. */
export type MountConfig =
  | { kind: "local"; host_root: string }
  | { kind: "notion"; api_key: string }
  | { kind: "s3"; bucket: string; region: string; access_key_id: string; secret_access_key: string; endpoint: string | null; key_prefix: string | null };
export interface MountRequest { path: string; label: string | null; config: MountConfig }

export type WorkspaceStatus = { status: "mounted" } | { status: "degraded"; reason: string };
export interface WorkspaceInfo { mountpoint: string; files_root: string; status: WorkspaceStatus }

export interface Entry { name: string; path: string; kind: "dir" | "file"; size: number | null; mtime_ms: number | null }
export interface FileContent {
  path: string;
  text: string | null;
  /** What the bytes were decoded as. `null` when there was no text to read. */
  encoding: string | null;
  size: number;
  truncated: boolean;
}
export interface ImportReport { files: number; bytes: number; skipped: string[] }

/** One way a provider will route a call: what goes into a model id, and its own name for it. */
export interface RegionRouting { id: string; label: string }
export interface ProviderSetting {
  key: string;
  label: string;
  has_key: boolean;
  key_hint: string;
  /** The region a call goes to, already defaulted by the engine. `null` for a provider with one endpoint. */
  region: string | null;
  /** The regions on offer; empty unless there are several. */
  regions: string[];
  /** Which inference profile the provider's models are reached through, `null` where there is one. */
  routing: string | null;
  /** The profiles on offer, from the catalog; empty unless there are several. */
  routings: RegionRouting[];
}
export interface Settings { providers: ProviderSetting[]; default_model: string; max_tokens: number; max_turns: number; catalog_refresh: boolean }
export interface SettingsPatch {
  provider_keys?: Record<string, string | null>;
  bedrock_region?: string;
  bedrock_routing?: string;
  default_model?: string;
  max_tokens?: number;
  max_turns?: number;
  catalog_refresh?: boolean;
}
export interface ModelCost { input: number | null; output: number | null; cache_read: number | null; cache_write: number | null }
/** Where the model list came from, and whether a newer one is on its way. See `lib/catalog`. */
export interface CatalogStatus {
  /** When the list in use was fetched from models.dev, Unix ms; `null` when there is none yet. */
  fetched_at: number | null;
  /** Models in the list, every provider counted. Zero is a first start that has not fetched. */
  models: number;
  refreshing: boolean;
  /** Why the last fetch failed, until one succeeds. */
  error: string | null;
}
export interface ModelInfo { id: string; provider: string; name: string; context: number | null; output: number | null; cost: ModelCost | null; reasoning: boolean; tool_call: boolean; available: boolean }
