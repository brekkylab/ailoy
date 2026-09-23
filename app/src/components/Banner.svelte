<script lang="ts">
  import { TriangleAlert } from "@lucide/svelte";

  let { text, tone = "warn" }: { text: string; tone?: "warn" | "error" } = $props();
</script>

<div
  class={[
    "flex items-center gap-2 border-b px-3 py-1.5 text-xs",
    tone === "error" ? "bg-destructive/10 text-destructive" : "bg-amber-500/10 text-amber-700 dark:text-amber-300",
  ]}
>
  <TriangleAlert class="size-3.5 shrink-0" />
  {text}
</div>
