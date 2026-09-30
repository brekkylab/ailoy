// The agents the backend keeps under `~/.cache/ailoy/agents` (see `src-tauri/src/agent.rs`).
//
// Edits autosave instead of waiting for a Save button, so leaving the tab never loses work.
//
// Outside Tauri (plain `vite dev`) there is no backend: the tab opens one blank in-memory
// agent and nothing is written.

import { invoke } from "@tauri-apps/api/core";

import { type Agent, blankAgent, duplicateOf, fromStored } from "@/lib/agent";
import { confirm } from "@/lib/confirm";
import { MODELS } from "@/lib/mock";
import { S } from "@/strings";

// Debounces keystrokes.
const SAVE_DELAY = 600;

const inTauri = () => "__TAURI_INTERNALS__" in window;
const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

export type SaveState = "clean" | "saving" | "saved" | "failed";

class Agents {
  list = $state<Agent[]>([]);
  /** The agent a chat starts with; stored on disk as a flag on one agent. */
  defaultId = $state<string | null>(null);
  /** Nothing is saved until the first list has come back. */
  loaded = $state(false);
  loadError = $state<string | null>(null);
  saveState = $state<SaveState>("clean");
  saveError = $state<string | null>(null);

  /**
   * Last-written body per agent id, so a no-op edit writes nothing. Not `$state`: the effect
   * must not depend on it, or recording a save would schedule another.
   */
  private written = new Map<string, string>();

  constructor() {
    // One effect over the whole collection: making an agent the default also clears the flag
    // on another, so one click dirties two documents.
    $effect.root(() => {
      $effect(() => {
        if (!this.loaded) return;
        const dirty = this.list
          .map((agent) => this.document(agent))
          .map((doc) => ({ doc, body: JSON.stringify(doc) }))
          .filter(({ doc, body }) => this.written.get(doc.id as string) !== body);
        if (!dirty.length) return;
        const timer = setTimeout(() => void this.flush(dirty), SAVE_DELAY);
        return () => clearTimeout(timer);
      });
    });
  }

  /** What an export writes: the on-screen document, without this install's default flag. */
  exported(agent: Agent): Record<string, unknown> {
    const { default: _, ...doc } = this.document(agent);
    return doc;
  }

  /** The document a save would send. */
  private document(agent: Agent): Record<string, unknown> {
    return { ...$state.snapshot(agent), default: agent.id === this.defaultId };
  }

  /**
   * Reads the collection once. Later visits keep what is on screen: the backend holds only
   * what this tab wrote, and re-reading mid-edit would drop a pending save.
   */
  async load() {
    if (this.loaded) return;
    if (!inTauri()) {
      this.list = [blankAgent("Default", MODELS[0].id)];
      this.defaultId = this.list[0].id;
      this.loaded = true;
      return;
    }
    try {
      const docs = await invoke<Record<string, unknown>[]>("list_agents");
      this.list = docs.map(fromStored);
      this.defaultId = (docs.find((d) => d.default)?.id as string | undefined) ?? null;
      // Snapshot while `defaultId` still matches the backend…
      for (const agent of this.list) this.written.set(agent.id, JSON.stringify(this.document(agent)));
      // …then fill in a missing default, leaving that agent dirty so the choice is saved.
      if (!this.list.length) this.list = [blankAgent("Default", MODELS[0].id)];
      this.defaultId ??= this.list[0].id;
      this.loadError = null;
      this.loaded = true;
    } catch (e) {
      this.loadError = messageOf(e);
    }
  }

  /**
   * Sends the bodies captured when found dirty, not re-read from `list`, so an edit made
   * mid-flight is not recorded under the older body.
   */
  private async flush(dirty: { doc: Record<string, unknown>; body: string }[]) {
    if (!inTauri()) {
      for (const { doc, body } of dirty) this.written.set(doc.id as string, body);
      return;
    }
    this.saveState = "saving";
    this.saveError = null;
    try {
      // In series: the backend clears the old default as a side effect of saving the new one,
      // and concurrent saves would race on it.
      for (const { doc, body } of dirty) {
        await invoke("save_agent", { agent: doc });
        this.written.set(doc.id as string, body);
      }
      this.saveState = "saved";
    } catch (e) {
      // Left out of `written`, so the next edit retries it.
      this.saveState = "failed";
      this.saveError = messageOf(e);
    }
  }

  /** A new agent on the given model, already in `list`. */
  create(name: string, model: string): Agent {
    const agent = blankAgent(name, model);
    this.list = [...this.list, agent];
    return agent;
  }

  duplicate(agent: Agent, name: string): Agent {
    const copy = duplicateOf($state.snapshot(agent), name);
    this.list = [...this.list, copy];
    return copy;
  }

  /**
   * Deletes `id` on disk first: if the file delete failed after the list drop, it would
   * return on next start with references to it already removed from other agents. The last
   * agent is kept so a chat always has one.
   */
  async remove(id: string): Promise<boolean> {
    if (this.list.length <= 1) return false;
    if (inTauri()) {
      try {
        await invoke("remove_agent", { id });
      } catch (e) {
        this.saveState = "failed";
        this.saveError = messageOf(e);
        return false;
      }
    }
    this.written.delete(id);
    this.list = this.list.filter((a) => a.id !== id);
    // Remove it as a sub-agent; the effect saves the agents that named it.
    for (const agent of this.list) {
      if (agent.subagents.includes(id)) agent.subagents = agent.subagents.filter((s) => s !== id);
    }
    if (this.defaultId === id) this.defaultId = this.list[0].id;
    return true;
  }

  /** Confirms, then deletes. */
  async confirmRemove(agent: Agent): Promise<boolean> {
    if (this.list.length <= 1) return false;
    if (!(await confirm(S.confirmDeleteAgent(agent.name || S.untitled), S.deleteAgent))) return false;
    return this.remove(agent.id);
  }
}

export const agents = new Agents();
