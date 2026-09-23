<script lang="ts">
  import { Check, ChevronDown } from "@lucide/svelte";

  import { MODELS, PROVIDERS } from "@/lib/mock";
  import { S } from "@/strings";

  let {
    value,
    onChange,
    disabled = false,
  }: { value: string; onChange: (id: string) => void; disabled?: boolean } = $props();

  let open = $state(false);
  let root = $state<HTMLDivElement>();
  const current = $derived(MODELS.find((m) => m.id === value));
  const providers = PROVIDERS.filter((p) => MODELS.some((m) => m.provider === p.key));

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
    {disabled}
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-label={S.chooseModel}
  >
    <span class="truncate">
      {current ? `${current.name}${current.provider === "bedrock" ? " · Bedrock" : ""}` : (value ?? S.model)}
    </span>
    <ChevronDown class="size-3.5 shrink-0" />
  </button>
  {#if open}
    <div
      role="listbox"
      aria-label={S.chooseModel}
      class="absolute bottom-full left-0 z-10 mb-1 w-64 rounded-lg border bg-popover p-1 text-popover-foreground shadow-md"
    >
      {#each providers as provider (provider.key)}
        <div class="px-2 pt-1.5 pb-1 text-[11px] font-medium text-muted-foreground">{provider.label}</div>
        {#each MODELS.filter((m) => m.provider === provider.key) as m (m.id)}
          <button
            role="option"
            aria-selected={m.id === value}
            class="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
            onclick={() => pick(m.id)}
          >
            <span class="flex-1 truncate">{m.name}</span>
            {#if m.id === value}<Check class="size-3.5 text-muted-foreground" />{/if}
          </button>
        {/each}
      {/each}
    </div>
  {/if}
</div>
