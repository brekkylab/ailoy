<!--
  Which agent a chat runs with. `value` is an agent id; `null`, or an id no longer in the
  list, shows as the default agent, which is what such a chat would run with.
-->
<script lang="ts">
  import { Check, ChevronDown } from "@lucide/svelte";

  import { agents } from "@/lib/agents.svelte";
  import { S } from "@/strings";

  let {
    value,
    onChange,
    disabled = false,
  }: { value: string | null; onChange: (id: string) => void; disabled?: boolean } = $props();

  $effect(() => {
    void agents.load();
  });

  let open = $state(false);
  let root = $state<HTMLDivElement>();
  const current = $derived(
    agents.list.find((a) => a.id === value) ?? agents.list.find((a) => a.id === agents.defaultId) ?? null,
  );

  function pick(id: string) {
    onChange(id);
    open = false;
  }
</script>

<svelte:window
  onpointerdown={(e) => {
    if (open && root && !root.contains(e.target as Node)) open = false;
  }}
  onkeydown={(e) => {
    if (open && e.key === "Escape") open = false;
  }}
/>

<div bind:this={root} class="relative">
  <button
    class="flex min-w-0 items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
    onclick={() => (open = !open)}
    disabled={disabled || !agents.loaded}
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-label={S.chooseAgent}
  >
    <span class="truncate">{current ? current.name || S.untitled : S.agent}</span>
    <ChevronDown class="size-3.5 shrink-0" />
  </button>
  {#if open}
    <div
      role="listbox"
      aria-label={S.chooseAgent}
      class="absolute bottom-full left-0 z-10 mb-1 max-h-80 w-64 overflow-y-auto rounded-lg border bg-popover p-1 text-popover-foreground shadow-md"
    >
      {#each agents.list as a (a.id)}
        <button
          role="option"
          aria-selected={a.id === current?.id}
          class="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-accent"
          onclick={() => pick(a.id)}
        >
          <span class="min-w-0 flex-1">
            <span class="flex items-center gap-1.5 text-sm">
              <span class="truncate">{a.name || S.untitled}</span>
              {#if a.id === agents.defaultId}
                <span class="shrink-0 text-[10px] text-muted-foreground">{S.defaultBadge}</span>
              {/if}
            </span>
            <span class="block truncate font-mono text-[11px] text-muted-foreground">{a.model}</span>
          </span>
          {#if a.id === current?.id}<Check class="size-3.5 shrink-0 text-muted-foreground" />{/if}
        </button>
      {/each}
    </div>
  {/if}
</div>
