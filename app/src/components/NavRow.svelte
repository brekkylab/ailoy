<!-- One row of the sidebar nav, styled to match a session row so the list reads as one column. -->
<script lang="ts">
  import type { Component } from "svelte";

  let {
    icon: Icon,
    label,
    active,
    onclick,
  }: { icon: Component<{ class?: string }>; label: string; active: boolean; onclick: () => void } = $props();
</script>

<button
  class={["flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent", active && "bg-accent"]}
  aria-current={active ? "page" : undefined}
  {onclick}
>
  <Icon class="size-4" />
  <span class="truncate">{label}</span>
</button>
