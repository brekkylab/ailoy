<!-- Placeholder: what the agent has produced, once there is an agent. -->
<script lang="ts">
  import EmptyState from "@/components/EmptyState.svelte";
  import { S } from "@/strings";
</script>

<section class="flex min-h-0 min-w-0 flex-1 flex-col border-t">
  <EmptyState text={S.noArtifacts} />
</section>
