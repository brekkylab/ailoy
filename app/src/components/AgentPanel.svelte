<!--
  Configures an agent; chatting with one is the Chats tab. Every control is a field of the
  agent's document and is saved as it is edited (see `lib/agents.svelte.ts`).
-->
<script lang="ts">
  import { ArrowDown, ArrowUp, Bot, Check, Copy, Plus, Star, Trash, X } from "@lucide/svelte";

  import EmptyState from "@/components/EmptyState.svelte";
  import Section from "@/components/settings/Section.svelte";
  import {
    type Agent,
    SEARCH_ENGINES,
    STEP_KINDS,
    type StepKind,
    badNumber,
    entry,
    newId,
    stepLine,
  } from "@/lib/agent";
  import { agents } from "@/lib/agents.svelte";
  import { contexts } from "@/lib/contexts.svelte";
  import { MODELS, PROVIDERS } from "@/lib/mock";
  import { btn, input } from "@/lib/ui";
  import { S } from "@/strings";
  import type { AgentSection } from "@/views";

  let {
    agentId,
    section,
    onSelectAgent,
  }: {
    /** Already resolved by `App`, which falls back to the default agent. */
    agentId: string | null;
    section: AgentSection;
    onSelectAgent: (id: string) => void;
  } = $props();

  // Looked up here rather than passed in, so the edits below are to the collection's own
  // state and not to a prop.
  const agent = $derived(agents.list.find((a) => a.id === agentId) ?? null);

  $effect(() => {
    void agents.load();
    // The general page attaches a context, so it needs the current list.
    void contexts.refresh();
  });

  const inTauri = "__TAURI_INTERNALS__" in window;
  const others = $derived(agent ? agents.list.filter((a) => a.id !== agent.id) : []);
  const json = $derived(agent ? JSON.stringify(entry(agent, agents.list), null, 2) : "");
  const provider = $derived.by(() => {
    const slash = agent?.model.indexOf("/") ?? -1;
    return slash > 0 ? (PROVIDERS.find((p) => p.key === agent!.model.slice(0, slash)) ?? null) : null;
  });
  /** An attached id the context list does not have: kept visible, since the document still names it. */
  const strayContext = $derived(
    agent?.context && !contexts.list.some((c) => c.id === agent.context) ? agent.context : null,
  );
  let copied = $state(false);

  function toggle(list: string[], value: string): string[] {
    return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
  }

  function duplicate(a: Agent) {
    onSelectAgent(agents.duplicate(a, S.copyOf(a.name || S.untitled)).id);
  }
  function addStep(a: Agent, kind: StepKind) {
    a.sandbox.steps = [...a.sandbox.steps, { id: newId("step"), kind, first: "", second: "" }];
  }
  function moveStep(a: Agent, index: number, by: number) {
    const next = index + by;
    if (next < 0 || next >= a.sandbox.steps.length) return;
    const steps = [...a.sandbox.steps];
    [steps[index], steps[next]] = [steps[next], steps[index]];
    a.sandbox.steps = steps;
  }
  const fields = (kind: StepKind) => STEP_KINDS.find((k) => k.id === kind) ?? STEP_KINDS[0];

  async function copySpec() {
    try {
      await navigator.clipboard.writeText(json);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      /* no clipboard here; the text is selectable */
    }
  }

  const lede = "max-w-[66ch] text-sm leading-relaxed text-muted-foreground";
  const label = "text-xs font-medium text-muted-foreground";
  const hint = "text-xs text-muted-foreground";
  const code = "font-mono text-[0.9em]";
  const mono = "font-mono text-xs";
  /** `input` without its `w-full`, for the controls that share a row and set their own width. */
  const cell = input.replace("w-full", "");
  const pre = "rounded-lg border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-muted-foreground";
  const option = (on: boolean) => [
    "flex w-full items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors hover:bg-accent",
    on && "border-primary/40 bg-primary/5",
  ];
  const box = (on: boolean, one: boolean) => [
    "grid size-4 shrink-0 place-items-center border",
    one ? "rounded-full" : "rounded",
    on ? "border-primary bg-primary text-primary-foreground" : "text-transparent",
  ];
  const chip = (on: boolean) => [
    "rounded-full border px-2.5 py-0.5 font-mono text-xs transition-colors hover:bg-accent",
    on ? "border-primary/50 bg-primary/10 text-foreground" : "text-muted-foreground",
  ];
</script>

<!-- `one` rounds the box: a radio, where only one of the list can be on. -->
{#snippet check(on: boolean, name: string, note: string, pick: () => void, one = false)}
  <button class={option(on)} aria-pressed={on} onclick={pick}>
    <span class={box(on, one)}><Check class="size-3" strokeWidth={3} /></span>
    <span class="min-w-0 flex-1">
      <span class="block truncate text-sm font-medium">{name}</span>
      <span class="block text-xs text-muted-foreground">{note}</span>
    </span>
  </button>
{/snippet}

<section class="flex min-h-0 min-w-0 flex-1 flex-col border-t">
  {#if !agents.loaded}
    {#if agents.loadError}
      <div class="grid flex-1 place-items-center">
        <div class="space-y-3 text-center">
          <p class="text-sm text-destructive">{S.agentsUnreadable(agents.loadError)}</p>
          <button class={btn.outline} onclick={() => void agents.load()}>{S.retry}</button>
        </div>
      </div>
    {:else}
      <EmptyState text={S.loadingAgents} />
    {/if}
  {:else if !agent}
    <EmptyState text={S.pickAgent} />
  {:else}
    <div class="min-h-0 flex-1 overflow-y-auto px-6 pt-5 pb-10">
      <div class="mx-auto max-w-3xl space-y-8">
        {#if section === "general"}
          <Section title="Identity">
            <p class={lede}>
              Also the agent's <code class={code}>card</code>, which is what a calling agent reads when this one is
              delegated to.
            </p>
            <div class="grid gap-3 sm:grid-cols-2">
              <label class="space-y-1.5">
                <span class={label}>{S.name}</span>
                <input class={input} bind:value={agent.name} placeholder="Default" />
              </label>
              <label class="space-y-1.5">
                <span class={label}>Description</span>
                <input class={input} bind:value={agent.description} placeholder="What this agent is for" />
              </label>
            </div>
          </Section>

          <Section title={S.model}>
            <p class={lede}>
              ailoy resolves <code class={code}>provider/model</code> against its provider registry, with the key saved
              for that provider in Settings.
            </p>
            <label class="block max-w-md space-y-1.5">
              <span class={label}>Model name</span>
              <input
                class={[input, mono]}
                list="agent-models"
                bind:value={agent.model}
                placeholder="anthropic/claude-sonnet-5"
                spellcheck="false"
              />
              <datalist id="agent-models">
                {#each MODELS as m (m.id)}
                  <option value={m.id}>{m.name}</option>
                {/each}
              </datalist>
              {#if provider}
                <span class={hint}>{provider.label}</span>
              {:else}
                <span class="text-xs text-destructive">
                  ailoy reads the part before the <code class={code}>/</code> as the provider; without a known one it has
                  nothing to resolve against.
                </span>
              {/if}
            </label>
            <div class="flex flex-wrap gap-1.5">
              {#each MODELS as m (m.id)}
                <button class={chip(agent.model === m.id)} onclick={() => (agent.model = m.id)}>{m.id}</button>
              {/each}
            </div>
          </Section>

          <Section title={S.context}>
            <p class={lede}>
              The context mounted read-only into this agent's sandbox — one at most for now. Click the chosen one again
              to run without any.
            </p>
            {#if !contexts.list.length && !strayContext}
              <p class={hint}>No context yet — the Context tab is where they are made.</p>
            {:else}
              <div class="space-y-1.5">
                {#each contexts.list as c (c.id)}
                  {@render check(
                    agent.context === c.id,
                    c.name,
                    c.id,
                    () => (agent.context = agent.context === c.id ? null : c.id),
                    true,
                  )}
                {/each}
                {#if strayContext}
                  {@render check(
                    true,
                    strayContext,
                    "Attached, but no longer there — deleted, or never made.",
                    () => (agent.context = null),
                    true,
                  )}
                {/if}
              </div>
            {/if}
          </Section>

          <Section title="Sampling">
            <p class={lede}>
              Every one is optional; an empty field leaves the provider's own default in place. Top-k is honoured by
              Anthropic and Gemini and ignored by OpenAI.
            </p>
            <div class="grid grid-cols-2 gap-3 md:grid-cols-4">
              {#each [["temperature", "Temperature"], ["topP", "Top P"], ["topK", "Top K"], ["maxTokens", "Max tokens"]] as const as [key, name] (key)}
                <label class="space-y-1.5">
                  <span class={label}>{name}</span>
                  <input
                    class={[input, mono, badNumber(agent.options[key]) && "border-destructive"]}
                    bind:value={agent.options[key]}
                    placeholder="default"
                    inputmode="decimal"
                  />
                </label>
              {/each}
            </div>
          </Section>

          <Section title="This agent">
            <div class="flex flex-wrap items-center gap-2">
              <button
                class={btn.outline}
                disabled={agent.id === agents.defaultId}
                onclick={() => (agents.defaultId = agent.id)}
              >
                <Star class="size-3.5" />
                {agent.id === agents.defaultId ? S.isDefault : S.makeDefault}
              </button>
              <button class={btn.outline} onclick={() => duplicate(agent)}>
                <Copy class="size-3.5" />
                {S.duplicate}
              </button>
              <div class="flex-1"></div>
              <button
                class={[btn.outline, "text-destructive"]}
                disabled={agents.list.length <= 1}
                title={agents.list.length <= 1 ? S.lastAgent : undefined}
                onclick={() => void agents.confirmRemove(agent)}
              >
                <Trash class="size-3.5" />
                {S.deleteAgent}
              </button>
            </div>
          </Section>
        {:else if section === "prompt"}
          <Section title={S.agentPrompt}>
            <p class={lede}>
              ailoy's <code class={code}>instruction</code>: private guidance the model works from. The public
              description a caller reads is the card, under General.
            </p>
            <textarea
              class={[input, "h-auto min-h-[420px] resize-y py-2.5 font-mono text-sm leading-relaxed"]}
              bind:value={agent.instruction}
              placeholder="You are the agent for…"
              spellcheck="false"
            ></textarea>
            <p class={hint}>
              {agent.instruction.length.toLocaleString()} characters · left empty, the model runs with no instruction at
              all.
            </p>
          </Section>
        {:else if section === "tools"}
          <Section title="Search engines">
            <p class={lede}>
              <code class={code}>web_search</code> aggregates whichever of these answer, then deduplicates and ranks by
              how many returned the same page. Selecting none means all of them.
            </p>
            <div class="flex flex-wrap gap-1.5">
              {#each SEARCH_ENGINES as engine (engine)}
                <button
                  class={chip(agent.engines.includes(engine))}
                  aria-pressed={agent.engines.includes(engine)}
                  onclick={() => (agent.engines = toggle(agent.engines, engine))}
                >
                  {engine}
                </button>
              {/each}
            </div>
            <p class={hint}>
              {agent.engines.length === 0
                ? "All engines."
                : `${agent.engines.length} of ${SEARCH_ENGINES.length} engines.`}
            </p>
          </Section>

          <Section title="MCP servers">
            <div class="flex items-start gap-3">
              <p class={[lede, "flex-1"]}>
                A tool source rather than a tool: the server is asked what it has, and what it answers joins the
                built-in tools every agent gets. ailoy registers the stdio transport but does not implement it yet.
              </p>
              <button
                class={btn.outline}
                onclick={() =>
                  (agent.mcp = [...agent.mcp, { id: newId("mcp"), name: "", transport: "http", target: "" }])}
              >
                <Plus class="size-3.5" /> Add server
              </button>
            </div>
            {#if !agent.mcp.length}
              <p class={hint}>No MCP server on this agent.</p>
            {:else}
              <div class="space-y-1.5">
                {#each agent.mcp as server (server.id)}
                  <div class="flex items-center gap-1.5">
                    <input class={[cell, "w-36 shrink-0"]} bind:value={server.name} placeholder={S.name} />
                    <select class={[cell, "w-40 shrink-0"]} bind:value={server.transport}>
                      <option value="http">Streamable HTTP</option>
                      <option value="stdio">stdio</option>
                    </select>
                    <input
                      class={[cell, mono, "flex-1"]}
                      bind:value={server.target}
                      placeholder={server.transport === "http" ? "https://host/mcp" : "my-server --stdio"}
                      spellcheck="false"
                    />
                    <button
                      class={btn.ghostIcon}
                      aria-label="Remove server"
                      onclick={() => (agent.mcp = agent.mcp.filter((s) => s.id !== server.id))}
                    >
                      <X class="size-4" />
                    </button>
                  </div>
                {/each}
              </div>
            {/if}
          </Section>

          <Section title="Sub-agents">
            <p class={lede}>
              A sub-agent is registered as a tool the parent can call, and what the parent reads to choose is the
              sub-agent's card.
            </p>
            {#if !others.length}
              <p class={hint}>There is no other agent to delegate to yet.</p>
            {:else}
              <div class="space-y-1.5">
                {#each others as other (other.id)}
                  {@render check(
                    agent.subagents.includes(other.id),
                    other.name || S.untitled,
                    other.description || "No description — a caller has nothing to go on.",
                    () => (agent.subagents = toggle(agent.subagents, other.id)),
                  )}
                {/each}
              </div>
            {/if}
          </Section>
        {:else if section === "sandbox"}
          <Section title="Base">
            <p class={lede}>
              What this agent's shell and file tools see — cortex calls it the console's rootfs. A recipe rather than a
              directory: cortex builds what it names, and the console runs in that.
            </p>
            <label class="block max-w-md space-y-1.5">
              <span class={label}>Image</span>
              <input
                class={[input, mono]}
                bind:value={agent.sandbox.base}
                placeholder="python:3.13-slim"
                spellcheck="false"
              />
            </label>
          </Section>

          <Section title="Steps">
            <div class="flex flex-wrap gap-1.5">
              {#each STEP_KINDS as kind (kind.id)}
                <button class={btn.outline} onclick={() => addStep(agent, kind.id)}>
                  <Plus class="size-3.5" />
                  {kind.label}
                </button>
              {/each}
            </div>
            {#if !agent.sandbox.steps.length}
              <p class={hint}>No steps — the base is the whole of it.</p>
            {:else}
              <div class="space-y-1.5">
                {#each agent.sandbox.steps as step, i (step.id)}
                  {@const f = fields(step.kind)}
                  <div class="flex items-center gap-1.5">
                    <span class="w-5 shrink-0 text-right text-xs text-muted-foreground tabular-nums">{i + 1}</span>
                    <select class={[cell, mono, "w-28 shrink-0"]} bind:value={step.kind}>
                      {#each STEP_KINDS as kind (kind.id)}
                        <option value={kind.id}>{kind.label}</option>
                      {/each}
                    </select>
                    <input
                      class={[cell, mono, "flex-[2]"]}
                      bind:value={step.first}
                      placeholder={f.first}
                      spellcheck="false"
                    />
                    {#if f.second}
                      <input
                        class={[cell, mono, "flex-1"]}
                        bind:value={step.second}
                        placeholder={f.second}
                        spellcheck="false"
                      />
                    {/if}
                    <div class="flex shrink-0">
                      <button
                        class={btn.ghostIcon}
                        aria-label="Move up"
                        disabled={i === 0}
                        onclick={() => moveStep(agent, i, -1)}
                      >
                        <ArrowUp class="size-3.5" />
                      </button>
                      <button
                        class={btn.ghostIcon}
                        aria-label="Move down"
                        disabled={i === agent.sandbox.steps.length - 1}
                        onclick={() => moveStep(agent, i, 1)}
                      >
                        <ArrowDown class="size-3.5" />
                      </button>
                      <button
                        class={btn.ghostIcon}
                        aria-label="Remove step"
                        onclick={() => (agent.sandbox.steps = agent.sandbox.steps.filter((s) => s.id !== step.id))}
                      >
                        <X class="size-3.5" />
                      </button>
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
          </Section>

          <Section title="As the build reports it">
            {@const lines = [`FROM ${agent.sandbox.base || "…"}`, ...agent.sandbox.steps.map(stepLine)]}
            <pre class={[pre, "break-words whitespace-pre-wrap"]}>{lines.join("\n")}</pre>
          </Section>
        {:else}
          <Section title="What would be run">
            <div class="flex items-start gap-3">
              <p class={[lede, "flex-1"]}>
                The <code class={code}>AgentSpec</code> ailoy would be constructed from, and beside it the sandbox recipe
                and the context to mount. Separate objects: ailoy keeps runtime off the spec.
              </p>
              <button class={btn.outline} onclick={() => void copySpec()}>
                {#if copied}<Check class="size-3.5" /> {S.copied}{:else}<Copy class="size-3.5" /> {S.copy}{/if}
              </button>
            </div>
            <pre class={[pre, "max-h-[560px] overflow-auto"]}>{json}</pre>
          </Section>
        {/if}
      </div>
    </div>

    <footer class="flex items-center gap-1.5 border-t px-4 py-1.5 text-xs text-muted-foreground">
      <Bot class="size-3.5" />
      <span>{agents.list.length} {agents.list.length === 1 ? "agent" : "agents"}</span>
      <span>·</span>
      <span class="truncate font-mono">{agent.model}</span>
      <div class="flex-1"></div>
      {#if !inTauri}
        <span>{S.heldInMemory}</span>
      {:else if agents.saveState === "saving"}
        <span>{S.saving}</span>
      {:else if agents.saveState === "failed"}
        <span class="truncate text-destructive" title={agents.saveError ?? ""}>
          {S.notSaved(agents.saveError ?? "")}
        </span>
      {:else}
        <span>{S.savedAsFiles}</span>
      {/if}
    </footer>
  {/if}
</section>
