<script lang="ts">
  import MessageActions from "@/components/thread/MessageActions.svelte";
  import type { ChatMessage } from "@/types";

  let { message }: { message: ChatMessage } = $props();
</script>

<!-- Plain text for now; markdown rendering comes with real replies. -->
<div class="group/msg max-w-[92%] space-y-1">
  <div class="leading-relaxed whitespace-pre-wrap">{message.text}</div>
  <MessageActions text={message.text} at={message.created_at} />
</div>
