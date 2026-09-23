<!--
  The strip along the top of the window, in two segments.

  The window runs with `titleBarStyle: "Overlay"`, so macOS draws the traffic lights over
  the webview. That costs a left inset the lights can sit in, and a drag region:
  `data-tauri-drag-region="deep"` makes the whole bar draggable except its buttons.

  The left segment keeps the sidebar's width so the divider below it runs straight. The
  right one names what the main panel is showing and holds the way into settings.
-->
<script lang="ts">
  import { PanelLeftClose, PanelLeftOpen, Settings } from "@lucide/svelte";

  import { btn } from "@/lib/ui";
  import { S } from "@/strings";

  let {
    title,
    collapsed,
    onToggleSidebar,
    settingsActive,
    onOpenSettings,
  }: {
    title: string;
    collapsed: boolean;
    onToggleSidebar: () => void;
    settingsActive: boolean;
    onOpenSettings: () => void;
  } = $props();
</script>

<!-- `pl-[78px]` pairs with `trafficLightPosition` in `tauri.conf.json`: it is the room the
     three lights need, and nothing interactive may sit under them. -->
<header data-tauri-drag-region="deep" class="col-span-full flex h-11 shrink-0 items-stretch">
  <div class={["flex shrink-0 items-center pl-[78px]", !collapsed && "w-(--sidebar-w)"]}>
    <button
      class={btn.ghostIcon}
      onclick={onToggleSidebar}
      aria-label={collapsed ? S.expandSidebar : S.collapseSidebar}
      aria-expanded={!collapsed}
    >
      {#if collapsed}<PanelLeftOpen class="size-4" />{:else}<PanelLeftClose class="size-4" />{/if}
    </button>
  </div>
  <div class={["flex min-w-0 flex-1 items-center gap-2 pr-2 pl-3", !collapsed && "border-l"]}>
    {#if title}
      <h1 class="min-w-0 truncate text-sm font-medium">{title}</h1>
    {/if}
    <button
      class={[btn.ghostIcon, "ml-auto", settingsActive && "bg-accent text-accent-foreground"]}
      onclick={onOpenSettings}
      aria-label={S.settings}
      aria-pressed={settingsActive}
    >
      <Settings class="size-4" />
    </button>
  </div>
</header>
