<script lang="ts">
  import MessageActions from "@/components/thread/MessageActions.svelte";
  import type { ChatMessage } from "@/types";

  let { message }: { message: ChatMessage } = $props();
</script>

<div class="group/msg ml-auto flex max-w-[80%] flex-col items-end gap-1">
  <!-- Quiet in both themes: the accent is kept for the send button. -->
  <div class="rounded-2xl bg-foreground/[0.07] px-4 py-2.5 whitespace-pre-wrap">{message.text}</div>
  <MessageActions text={message.text} at={message.created_at} end />
</div>
