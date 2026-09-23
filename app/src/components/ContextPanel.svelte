<!-- The selected context, through whichever of its commands is open. -->
<script lang="ts">
  import FilesView from "@/components/context/FilesView.svelte";
  import EmptyState from "@/components/EmptyState.svelte";
  import type { Context } from "@/lib/contexts.svelte";
  import { S } from "@/strings";
  import type { ContextCommand } from "@/views";

  let { context, command }: { context: Context | null; command: ContextCommand } = $props();
</script>

{#if !context}
  <section class="flex min-h-0 min-w-0 flex-1 flex-col border-t">
    <EmptyState text={S.pickContext} />
  </section>
{:else if command === "files"}
  <!-- Keyed so another context starts at its own root rather than at this one's folder. -->
  {#key context.id}
    <FilesView {context} />
  {/key}
{/if}
