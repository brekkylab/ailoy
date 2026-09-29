<script lang="ts">
  import type { Snippet } from "svelte";

  let { title, children }: { title: string; children: Snippet } = $props();
</script>

<section class="space-y-3">
  <h2 class="text-sm font-semibold">{title}</h2>
  {@render children()}
</section>
