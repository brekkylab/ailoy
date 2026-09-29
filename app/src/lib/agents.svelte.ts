// The agents the backend keeps under `~/.cache/ailoy/agents` (see `src-tauri/src/agent.rs`).
//
// Edits are written back as they are made rather than behind a Save button: every control
// in the Agent tab is a field of a document, and a settings tab that can be left with
// unsaved work in it is one that loses work. `SAVE_DELAY` keeps that from being a write
// per keystroke.
//
// Outside Tauri — a plain `vite dev` in the browser — there is no backend: the tab opens
// on one blank agent held in memory, and nothing is written.

import { invoke } from "@tauri-apps/api/core";

import { type Agent, blankAgent, duplicateOf, fromStored } from "@/lib/agent";
import { confirm } from "@/lib/confirm";
import { MODELS } from "@/lib/mock";
import { S } from "@/strings";

const SAVE_DELAY = 600;

const inTauri = () => "__TAURI_INTERNALS__" in window;
const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

export type SaveState = "clean" | "saving" | "saved" | "failed";

class Agents {
  list = $state<Agent[]>([]);
  /** The agent a chat starts with. On disk it is a flag on one agent; one value here. */
  defaultId = $state<string | null>(null);
  /** Nothing is saved until the first list has come back. */
  loaded = $state(false);
  loadError = $state<string | null>(null);
  saveState = $state<SaveState>("clean");
  saveError = $state<string | null>(null);

  /**
   * What was last written for each agent, by id, so an edit that changes nothing writes
   * nothing. Not `$state`: the effect below must not depend on it, or recording a save
   * would schedule another one.
   */
  private written = new Map<string, string>();

  constructor() {
    // One effect over the whole collection rather than one per agent: making an agent the
    // default clears the flag on another, and that is two documents from one click.
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

  /** What an export writes: the document as it is on screen, without this install's default. */
  exported(agent: Agent): Record<string, unknown> {
    const { default: _, ...doc } = this.document(agent);
    return doc;
  }

  /** The document a save would send. */
  private document(agent: Agent): Record<string, unknown> {
    return { ...$state.snapshot(agent), default: agent.id === this.defaultId };
  }

  /**
   * Reads the collection once. Later visits keep what is on screen: the backend only ever
   * holds what this tab wrote, and re-reading mid-edit would throw away a pending save.
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
      // Everything just read is what is on disk, so the snapshot is taken while
      // `defaultId` is still exactly what the backend said…
      for (const agent of this.list) this.written.set(agent.id, JSON.stringify(this.document(agent)));
      // …and only then is a missing one filled in, which leaves that agent dirty and so
      // gets the decision written back.
      if (!this.list.length) this.list = [blankAgent("Default", MODELS[0].id)];
      this.defaultId ??= this.list[0].id;
      this.loadError = null;
      this.loaded = true;
    } catch (e) {
      this.loadError = messageOf(e);
    }
  }

  /**
   * Sends the documents as they were when found to differ — not re-read from `list`, or an
   * edit made while this is in flight would be recorded under the older body.
   */
  private async flush(dirty: { doc: Record<string, unknown>; body: string }[]) {
    if (!inTauri()) {
      for (const { doc, body } of dirty) this.written.set(doc.id as string, body);
      return;
    }
    this.saveState = "saving";
    this.saveError = null;
    try {
      // In series: the backend clears the old default as a side effect of writing the new
      // one, and two saves racing that would be two answers to the same question.
      for (const { doc, body } of dirty) {
        await invoke("save_agent", { agent: doc });
        this.written.set(doc.id as string, body);
      }
      this.saveState = "saved";
    } catch (e) {
      // Left out of `written`, so the next edit tries it again.
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
   * Deletes `id`, on disk first: dropping it from the list and then failing to delete the
   * file would bring it back on the next start, with the references to it cleaned out of
   * everyone else. The last agent stays, so a chat always has one to start with.
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
    // No longer anyone's sub-agent. The effect writes the ones that named it.
    for (const agent of this.list) {
      if (agent.subagents.includes(id)) agent.subagents = agent.subagents.filter((s) => s !== id);
    }
    if (this.defaultId === id) this.defaultId = this.list[0].id;
    return true;
  }

  /** Asks first, then deletes: what the sidebar row and the General page both do. */
  async confirmRemove(agent: Agent): Promise<boolean> {
    if (this.list.length <= 1) return false;
    if (!(await confirm(S.confirmDeleteAgent(agent.name || S.untitled), S.deleteAgent))) return false;
    return this.remove(agent.id);
  }
}

export const agents = new Agents();
