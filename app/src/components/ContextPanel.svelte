<!-- The selected context, through whichever of its commands is open, and the helper beside it. -->
<script lang="ts">
  import FilesView from "@/components/context/FilesView.svelte";
  import EmptyState from "@/components/EmptyState.svelte";
  import HelperPane from "@/components/helper/HelperPane.svelte";
  import type { Context } from "@/lib/contexts.svelte";
  import { S } from "@/strings";
  import type { ContextCommand } from "@/views";

  let {
    context,
    command,
    helperOpen,
    onToggleHelper,
  }: { context: Context | null; command: ContextCommand; helperOpen: boolean; onToggleHelper: () => void } =
    $props();
</script>

{#if !context}
  <section class="flex min-h-0 min-w-0 flex-1 flex-col border-t">
    <EmptyState text={S.pickContext} />
  </section>
{:else}
  <div class="flex min-h-0 min-w-0 flex-1">
    {#if command === "files"}
      <!-- Keyed so another context opens where it was left rather than at this one's folder. -->
      {#key context.id}
        <FilesView {context} />
      {/key}
    {/if}
    <HelperPane
      helper="context"
      thread={context.id}
      intro={S.contextHelperIntro}
      starters={S.contextStarters}
      placeholder={S.contextComposerPlaceholder}
      open={helperOpen}
      onToggle={onToggleHelper}
    />
  </div>
{/if}
