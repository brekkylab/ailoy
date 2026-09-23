import { invoke } from "@tauri-apps/api/core";

import { store } from "@/lib/store.svelte";

interface DevEnvKeys {
  keys: Record<string, string>;
  bedrock_region: string | null;
}

/**
 * Seeds the provider keys from the `.env` the Rust side read at startup (see
 * `src-tauri/src/dev_env.rs`). Dev builds inside Tauri only; a key typed in Settings wins.
 */
export async function loadDevEnvKeys() {
  if (!import.meta.env.DEV || !("__TAURI_INTERNALS__" in window)) return;
  try {
    const env = await invoke<DevEnvKeys>("dev_env_keys");
    for (const [provider, key] of Object.entries(env.keys)) store.keys[provider] ??= key;
    if (env.bedrock_region) store.bedrockRegion = env.bedrock_region;
  } catch (err) {
    console.warn("could not read dev .env keys", err);
  }
}
