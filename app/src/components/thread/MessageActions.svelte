<!-- Copy and time under a message, shown on hover. -->
<script lang="ts">
  import { Check, Copy } from "@lucide/svelte";

  import { formatElapsed } from "@/lib/time";
  import { S } from "@/strings";

  let { text, at, end = false }: { text: string; at: number; end?: boolean } = $props();

  let copied = $state(false);
  let now = $state(Date.now());
  $effect(() => {
    const timer = setInterval(() => (now = Date.now()), 30_000);
    return () => clearInterval(timer);
  });

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      /* clipboard may be unavailable */
    }
  }
</script>

<div
  class={[
    "flex items-center gap-1 text-muted-foreground opacity-0 transition-opacity group-hover/msg:opacity-100 focus-within:opacity-100",
    end && "justify-end",
  ]}
>
  <button
    class="rounded-md p-1 hover:bg-accent hover:text-foreground"
    aria-label={copied ? S.copied : S.copy}
    title={copied ? S.copied : S.copy}
    onclick={copy}
  >
    {#if copied}<Check class="size-3.5" />{:else}<Copy class="size-3.5" />{/if}
  </button>
  <time class="px-1 text-[11px] tabular-nums" datetime={new Date(at).toISOString()} title={new Date(at).toLocaleString()}>
    {formatElapsed(at, now)}
  </time>
</div>
