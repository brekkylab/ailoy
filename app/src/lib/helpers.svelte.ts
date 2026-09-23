// The helper pane's agents: one per tab that has the pane, each asked to change what that
// tab shows — `context` a context's files, `agentmaker` the agents.
//
// Each is `agents/{id}/agent.json` in the cache (see `HELPERS` in `src-tauri/src/agent.rs`):
// kept beside the Agent tab's collection and out of it, and edited in its file. Outside
// Tauri there is no file, and a blank agent on the first model stands in.
//
// A helper keeps one thread per thing it works on — a context, an agent — in memory, like
// `store`: the reply is a placeholder until ailoy is wired in. When it is, a turn that
// changed what is on screen bumps `revision[key]`, and the tab re-reads on it.

import { invoke } from "@tauri-apps/api/core";

import { type Agent, blankAgent, fromStored } from "@/lib/agent";
import { MODELS } from "@/lib/mock";
import { S } from "@/strings";
import type { ChatMessage } from "@/types";

let counter = 0;
const uid = () => `hm${Date.now().toString(36)}${(counter++).toString(36)}`;

export type HelperId = "context" | "agentmaker";

class Helper {
  /** By thread key. */
  messages = $state<Record<string, ChatMessage[]>>({});
  running = $state<Record<string, boolean>>({});
  /** Goes up whenever the helper may have changed what the thread is about. */
  revision = $state<Record<string, number>>({});
  agent = $state<Agent | null>(null);
  /** The agent's model unless one was picked in the pane, which lasts until the app quits. */
  picked = $state<string | null>(null);
  model = $derived(this.picked ?? (this.agent?.model || MODELS[0].id));

  constructor(readonly id: HelperId) {}

  /** Reads the agent once; the pane calls it whenever it opens. */
  async load() {
    if (this.agent) return;
    if (!("__TAURI_INTERNALS__" in window)) {
      this.agent = blankAgent(this.id, MODELS[0].id);
      return;
    }
    try {
      this.agent = fromStored(await invoke<Record<string, unknown>>("get_helper_agent", { id: this.id }));
    } catch (err) {
      console.warn(`could not read the helper ${this.id}`, err);
    }
  }

  send(key: string, text: string) {
    this.messages[key] = [...(this.messages[key] ?? []), { id: uid(), role: "user", text, created_at: Date.now() }];
    this.running[key] = true;
    setTimeout(() => {
      if (!this.running[key]) return;
      this.messages[key] = [
        ...(this.messages[key] ?? []),
        { id: uid(), role: "assistant", text: S.notConnected, created_at: Date.now() },
      ];
      this.running[key] = false;
      this.revision[key] = (this.revision[key] ?? 0) + 1;
    }, 700);
  }

  stop(key: string) {
    this.running[key] = false;
  }

  clear(key: string) {
    this.running[key] = false;
    delete this.messages[key];
  }
}

export const helpers: Record<HelperId, Helper> = {
  context: new Helper("context"),
  agentmaker: new Helper("agentmaker"),
};
