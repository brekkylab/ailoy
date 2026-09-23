<script lang="ts">
  import { Bot, FolderTree, HardDrive, MessageSquarePlus, MessagesSquare, Package, Pencil, Trash } from "@lucide/svelte";

  import NavRow from "@/components/NavRow.svelte";
  import { confirm } from "@/lib/confirm";
  import { groupSessions } from "@/lib/sessionGroups";
  import { store } from "@/lib/store.svelte";
  import { input } from "@/lib/ui";
  import { S } from "@/strings";
  import type { MainView } from "@/views";

  let {
    selected,
    onSelect,
    onNewChat,
    drafting,
    view,
    onSelectView,
    source,
    onSelectSource,
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
    source: string | null;
    onSelectSource: (path: string | null) => void;
  } = $props();

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

  // Placeholder until sources come from the backend.
  const sources = [{ path: null, label: "My Computer" }];

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
      label={S.context}
      active={view === "context"}
      onclick={() => onSelectView("context")}
    />
    <NavRow icon={Bot} label={S.agent} active={view === "agent"} onclick={() => onSelectView("agent")} />
    <NavRow
      icon={Package}
      label={S.artifacts}
      active={view === "artifacts"}
      onclick={() => onSelectView("artifacts")}
    />
  </nav>

  <!-- The lower half belongs to whichever panel is open: sources for the context,
       conversations otherwise. -->
  {#if view === "context"}
    <div class="flex items-center justify-between px-4 pb-1">
      <h3 class="text-xs font-medium text-muted-foreground">{S.mounts}</h3>
      <button class="text-xs text-muted-foreground hover:text-foreground" disabled>{S.addMount}</button>
    </div>
    <div class="space-y-0.5 px-2">
      {#each sources as s (s.label)}
        <button
          class={[
            "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent",
            source === s.path && "bg-accent",
          ]}
          onclick={() => onSelectSource(s.path)}
        >
          <HardDrive class="size-4 text-muted-foreground" />
          <span class="truncate">{s.label}</span>
        </button>
      {/each}
    </div>
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
                <button class="min-w-0 flex-1 text-left" onclick={() => onSelect(s.id)} title={`${s.title}\n${s.model}`}>
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
