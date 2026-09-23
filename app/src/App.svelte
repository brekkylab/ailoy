<script lang="ts">
  import AgentPanel from "@/components/AgentPanel.svelte";
  import ArtifactsPanel from "@/components/ArtifactsPanel.svelte";
  import Banner from "@/components/Banner.svelte";
  import SettingsPanel from "@/components/SettingsPanel.svelte";
  import Sidebar from "@/components/Sidebar.svelte";
  import Thread from "@/components/Thread.svelte";
  import TitleBar from "@/components/TitleBar.svelte";
  import ContextPanel from "@/components/ContextPanel.svelte";
  import HelperPane from "@/components/helper/HelperPane.svelte";
  import { agents } from "@/lib/agents.svelte";
  import { contexts } from "@/lib/contexts.svelte";
  import { store } from "@/lib/store.svelte";
  import { syncDarkClass } from "@/lib/theme";
  import { S } from "@/strings";
  import type { AgentSection, ContextCommand, MainView } from "@/views";

  const KEY = "ailoy.session";
  const SIDEBAR_KEY = "ailoy.sidebarCollapsed";
  const HELPER_KEY = "ailoy.helperOpen";

  function read(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }
  function write(key: string, value: string | null) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      /* storage may be unavailable */
    }
  }

  $effect(() => syncDarkClass());
  // Loaded at startup so the default context is already selected when the tab is first opened.
  $effect(() => void contexts.refresh());

  let selected = $state<string | null>(read(KEY));
  // Not persisted: a relaunch should land on the conversation.
  let view = $state<MainView>("session");
  // A new chat not sent into yet. A counter rather than a flag, so a second click on
  // New chat still changes it and the composer can take focus again.
  let draft = $state<number | null>(null);
  let context = $state<string | null>(null);
  let command = $state<ContextCommand>("files");
  // Falls back to the default context, as the agent does to the default agent.
  const currentContext = $derived(
    contexts.list.find((c) => c.id === context) ?? contexts.list.find((c) => c.default) ?? null,
  );
  let agentId = $state<string | null>(null);
  let section = $state<AgentSection>("general");
  // Falls back to the default while the choice is unmade or was just deleted.
  const currentAgent = $derived(
    agents.list.find((a) => a.id === agentId) ??
      agents.list.find((a) => a.id === agents.defaultId) ??
      agents.list[0] ??
      null,
  );
  let collapsed = $state(read(SIDEBAR_KEY) === "1");
  // Open unless it was closed. One setting for every tab that has the helper.
  let helperOpen = $state(read(HELPER_KEY) !== "0");

  // `selected` is the user's choice; `effective` is what the window shows. It falls back to
  // the most recent session while the choice is unknown or gone.
  const stored = $derived(
    selected && store.sessions.some((s) => s.id === selected) ? selected : (store.sessions[0]?.id ?? null),
  );
  const effective = $derived(draft !== null ? null : stored);
  $effect(() => write(KEY, stored));

  const noKey = $derived(Object.keys(store.keys).length === 0);

  const title = $derived.by(() => {
    if (view === "session") return store.sessions.find((s) => s.id === effective)?.title ?? "";
    if (view === "context") return currentContext?.name ?? S.context;
    if (view === "agent") return currentAgent?.name || S.agent;
    if (view === "artifacts") return S.artifacts;
    return S.settings;
  });

  function selectSession(id: string | null) {
    selected = id;
    draft = null;
    view = "session";
  }
  function newChat() {
    draft = (draft ?? 0) + 1;
    view = "session";
  }
  function toggleSidebar() {
    collapsed = !collapsed;
    write(SIDEBAR_KEY, collapsed ? "1" : "0");
  }
  function toggleHelper() {
    helperOpen = !helperOpen;
    write(HELPER_KEY, helperOpen ? "1" : "0");
  }
  const hasHelper = $derived((view === "context" && !!currentContext) || (view === "agent" && agents.loaded));
</script>

<div
  style="--sidebar-w: 260px"
  class={[
    "grid h-full grid-rows-[auto_minmax(0,1fr)]",
    collapsed ? "grid-cols-[minmax(0,1fr)]" : "grid-cols-[var(--sidebar-w)_minmax(0,1fr)]",
  ]}
>
  <TitleBar
    {title}
    {collapsed}
    onToggleSidebar={toggleSidebar}
    settingsActive={view === "settings"}
    onOpenSettings={() => (view = "settings")}
    aside={hasHelper
      ? { open: helperOpen, label: helperOpen ? S.hideHelper : S.showHelper, onToggle: toggleHelper }
      : null}
  />
  <!-- Unmounted rather than hidden when collapsed: a zero-width sidebar still takes tab stops. -->
  {#if !collapsed}
    <Sidebar
      selected={effective}
      onSelect={selectSession}
      onNewChat={newChat}
      drafting={draft !== null}
      {view}
      onSelectView={(v) => (view = v)}
      context={currentContext?.id ?? null}
      onSelectContext={(id) => (context = id)}
      {command}
      onSelectCommand={(c) => (command = c)}
      agent={currentAgent?.id ?? null}
      onSelectAgent={(id) => (agentId = id)}
      {section}
      onSelectSection={(s) => (section = s)}
    />
  {/if}
  <main class="flex h-full min-w-0 flex-col">
    {#if noKey}
      <Banner text={S.noKey} tone="error" />
    {/if}
    {#if view === "session"}
      <Thread sessionId={effective} {draft} onCreated={selectSession} />
    {:else if view === "context"}
      <ContextPanel
        context={currentContext}
        {command}
        {helperOpen}
        onToggleHelper={toggleHelper}
      />
    {:else if view === "agent"}
      <div class="flex min-h-0 min-w-0 flex-1">
        <AgentPanel agentId={currentAgent?.id ?? null} {section} onSelectAgent={(id) => (agentId = id)} />
        {#if agents.loaded}
          <HelperPane
            helper="agentmaker"
            thread={currentAgent?.id ?? ""}
            intro={S.agentHelperIntro}
            starters={S.agentStarters}
            placeholder={S.agentComposerPlaceholder}
            open={helperOpen}
            onToggle={toggleHelper}
          />
        {/if}
      </div>
    {:else if view === "artifacts"}
      <ArtifactsPanel />
    {:else}
      <SettingsPanel />
    {/if}
  </main>
</div>
