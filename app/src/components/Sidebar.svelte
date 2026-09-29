<script lang="ts">
  import {
    Bot,
    Box,
    Cog,
    Files,
    Folder,
    FolderPlus,
    FolderTree,
    MessageSquarePlus,
    MessageSquareText,
    MessagesSquare,
    Package,
    Pencil,
    Plus,
    SlidersHorizontal,
    Trash,
    Wrench,
  } from "@lucide/svelte";

  import NavRow from "@/components/NavRow.svelte";
  import NewContextDialog from "@/components/NewContextDialog.svelte";
  import { agents } from "@/lib/agents.svelte";
  import { confirm } from "@/lib/confirm";
  import { contexts } from "@/lib/contexts.svelte";
  import { groupSessions } from "@/lib/sessionGroups";
  import { store } from "@/lib/store.svelte";
  import { input } from "@/lib/ui";
  import { S } from "@/strings";
  import type { AgentSection, ContextCommand, MainView } from "@/views";

  let {
    selected,
    onSelect,
    onNewChat,
    drafting,
    view,
    onSelectView,
    context,
    onSelectContext,
    command,
    onSelectCommand,
    agent,
    onSelectAgent,
    section,
    onSelectSection,
  }: {
    selected: string | null;
    /** `null` after the selected session is deleted: `App` then picks the next one. */
    onSelect: (id: string | null) => void;
    /** Opens an unsaved chat. Nothing is stored until the user sends into it. */
    onNewChat: () => void;
    /** Whether the thread on screen is that unsaved chat, which is New chat's row to mark. */
    drafting: boolean;
    view: MainView;
    onSelectView: (view: MainView) => void;
    context: string | null;
    onSelectContext: (id: string) => void;
    command: ContextCommand;
    onSelectCommand: (command: ContextCommand) => void;
    /** The agent on screen, already resolved: `App` falls back to the default. */
    agent: string | null;
    onSelectAgent: (id: string) => void;
    section: AgentSection;
    onSelectSection: (section: AgentSection) => void;
  } = $props();

  const commands: { id: ContextCommand; label: string; icon: typeof Files }[] = [
    { id: "files", label: S.files, icon: Files },
  ];
  const sections: { id: AgentSection; label: string; icon: typeof Files }[] = [
    { id: "general", label: S.agentGeneral, icon: SlidersHorizontal },
    { id: "prompt", label: S.agentPrompt, icon: MessageSquareText },
    { id: "tools", label: S.agentTools, icon: Wrench },
    { id: "sandbox", label: S.agentSandbox, icon: Box },
    { id: "advanced", label: S.agentAdvanced, icon: Cog },
  ];

  let editing = $state<string | null>(null);
  let draft = $state("");
  // One instant for the whole list, advanced on its own clock, so a session moves out of
  // Today at midnight even if nothing else happens.
  let now = $state(Date.now());
  $effect(() => {
    const timer = setInterval(() => (now = Date.now()), 60_000);
    return () => clearInterval(timer);
  });

  const groups = $derived(groupSessions(store.sessions, now));

  $effect(() => {
    if (view === "context") void contexts.refresh();
  });

  /** A chat's agent as the sidebar names it; `null` or a deleted id runs with the default. */
  function agentName(id: string | null): string {
    const a = agents.list.find((x) => x.id === id) ?? agents.list.find((x) => x.id === agents.defaultId);
    return a ? a.name || S.untitled : S.agent;
  }

  /** A new agent inherits the open one's model, which is likelier than the first in the list. */
  function createAgent() {
    const model = agents.list.find((a) => a.id === agent)?.model ?? store.defaultModel;
    onSelectAgent(agents.create(S.newAgentName, model).id);
    onSelectSection("general");
  }

  let naming = $state(false);
  async function createContext(name: string) {
    const made = await contexts.create(name);
    if (made) onSelectContext(made.id);
  }

  function commitRename(id: string, current: string) {
    const title = draft.trim();
    editing = null;
    if (title && title !== current) store.rename(id, title);
  }
  async function askDelete(id: string) {
    if (!(await confirm(S.confirmDelete, S.delete))) return;
    store.remove(id);
    if (id === selected) onSelect(null);
  }
  function focus(node: HTMLInputElement) {
    node.focus();
    node.select();
  }

  const badge = "shrink-0 rounded-full bg-primary/10 px-1.5 text-[0.65rem] font-semibold uppercase text-primary";
  const rowAction =
    "rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100";
</script>

<aside class="flex h-full min-h-0 flex-col border-r bg-muted/60">
  <nav class="space-y-0.5 px-2 pt-2 pb-3">
    <NavRow
      icon={MessagesSquare}
      label={S.chats}
      active={view === "session" && !drafting}
      onclick={() => onSelectView("session")}
    />
    <NavRow
      icon={FolderTree}
      label={S.contexts}
      active={view === "context"}
      onclick={() => onSelectView("context")}
    />
    <NavRow icon={Bot} label={S.agents} active={view === "agent"} onclick={() => onSelectView("agent")} />
    <NavRow
      icon={Package}
      label={S.artifacts}
      active={view === "artifacts"}
      onclick={() => onSelectView("artifacts")}
    />
  </nav>

  <!-- The lower half belongs to whichever panel is open: contexts for the context, agents
       for the agent, conversations otherwise. -->
  {#if view === "agent"}
    <div class="px-2 pb-2">
      <NavRow icon={Plus} label={S.newAgent} active={false} onclick={createAgent} />
    </div>
    <div class="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2">
      {#each agents.list as a (a.id)}
        <div
          class={[
            "group flex items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
            agent === a.id && "bg-accent",
          ]}
        >
          <button class="flex min-w-0 flex-1 items-center gap-2 text-left" onclick={() => onSelectAgent(a.id)} title={a.model}>
            <Bot class="size-4 shrink-0 text-muted-foreground" />
            <span class="min-w-0 flex-1">
              <span class="block truncate">{a.name || S.untitled}</span>
              <span class="block truncate font-mono text-xs text-muted-foreground">{a.model}</span>
            </span>
            {#if a.id === agents.defaultId}
              <span class={badge}>{S.defaultBadge}</span>
            {/if}
          </button>
          <!-- Not offered on the last one: a chat always needs an agent to start with. -->
          {#if agents.list.length > 1}
            <button class={rowAction} aria-label={S.deleteAgent} onclick={() => void agents.confirmRemove(a)}>
              <Trash class="size-3.5" />
            </button>
          {/if}
        </div>
      {:else}
        <p class="p-3 text-xs text-muted-foreground">{agents.loaded ? S.empty : S.loadingAgents}</p>
      {/each}
    </div>
    <!-- The pages of the open agent, pinned to the bottom like a context's commands. -->
    {@const current = agents.list.find((a) => a.id === agent)}
    {#if current}
      <nav class="space-y-0.5 border-t px-2 pt-2 pb-3">
        <h3 class="truncate px-2 pb-1 text-xs font-medium text-muted-foreground">{current.name || S.untitled}</h3>
        {#each sections as c (c.id)}
          <NavRow icon={c.icon} label={c.label} active={section === c.id} onclick={() => onSelectSection(c.id)} />
        {/each}
      </nav>
    {/if}
  {:else if view === "context"}
    <div class="px-2 pb-2">
      <NavRow icon={FolderPlus} label={S.newContext} active={naming} onclick={() => (naming = true)} />
    </div>
    <div class="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2">
      {#each contexts.list as c (c.id)}
        <button
          class={[
            "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
            context === c.id && "bg-accent",
          ]}
          onclick={() => onSelectContext(c.id)}
        >
          <Folder class="size-4 shrink-0 text-muted-foreground" />
          <span class="min-w-0 flex-1 truncate text-left">{c.name}</span>
          {#if c.default}
            <span class={badge}>{S.defaultBadge}</span>
          {/if}
        </button>
      {:else}
        <p class="p-3 text-xs text-muted-foreground">{S.empty}</p>
      {/each}
    </div>
    <!-- What the selected context can be opened on, pinned to the bottom. -->
    {@const current = contexts.list.find((c) => c.id === context)}
    {#if current}
      <nav class="space-y-0.5 border-t px-2 pt-2 pb-3">
        <h3 class="truncate px-2 pb-1 text-xs font-medium text-muted-foreground">{current.name}</h3>
        {#each commands as c (c.id)}
          <NavRow icon={c.icon} label={c.label} active={command === c.id} onclick={() => onSelectCommand(c.id)} />
        {/each}
      </nav>
    {/if}
  {:else}
    <!-- Heads the chat list, where a new conversation will land, rather than the nav. -->
    <div class="px-2 pb-2">
      <NavRow icon={MessageSquarePlus} label={S.newChat} active={view === "session" && drafting} onclick={onNewChat} />
    </div>
    <div class="min-h-0 flex-1 overflow-y-auto px-2">
      {#each groups as group (group.label)}
        <section class="pb-3">
          <h3 class="px-2 pb-1 text-xs font-medium text-muted-foreground">{group.label}</h3>
          {#each group.sessions as s (s.id)}
            <div
              class={[
                "group flex items-center gap-1 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
                view === "session" && selected === s.id && "bg-accent",
              ]}
            >
              {#if editing === s.id}
                <input
                  use:focus
                  aria-label={S.rename}
                  class={[input, "h-6 flex-1"]}
                  bind:value={draft}
                  onkeydown={(e) => {
                    if (e.key === "Enter") commitRename(s.id, s.title);
                    else if (e.key === "Escape") editing = null;
                  }}
                  onblur={() => (editing = null)}
                />
              {:else}
                <button class="min-w-0 flex-1 text-left" onclick={() => onSelect(s.id)} title={`${s.title}\n${agentName(s.agent)}`}>
                  <div class="truncate">
                    {#if store.running[s.id]}
                      <span class="sr-only">Running</span>
                      <span class="mr-1 inline-block size-2 animate-pulse rounded-full bg-emerald-500"></span>
                    {/if}
                    {s.title}
                  </div>
                </button>
                <button
                  class={rowAction}
                  aria-label={S.rename}
                  onclick={() => {
                    draft = s.title;
                    editing = s.id;
                  }}
                >
                  <Pencil class="size-3.5" />
                </button>
                <button class={rowAction} aria-label={S.delete} onclick={() => void askDelete(s.id)}>
                  <Trash class="size-3.5" />
                </button>
              {/if}
            </div>
          {/each}
        </section>
      {:else}
        <p class="p-3 text-xs text-muted-foreground">{S.empty}</p>
      {/each}
    </div>
  {/if}
</aside>

<NewContextDialog bind:open={naming} onCreate={(name) => void createContext(name)} />
