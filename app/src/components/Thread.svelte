<script lang="ts">
  import { tick } from "svelte";

  import EmptyState from "@/components/EmptyState.svelte";
  import AssistantBubble from "@/components/thread/AssistantBubble.svelte";
  import Composer from "@/components/thread/Composer.svelte";
  import UserBubble from "@/components/thread/UserBubble.svelte";
  import { store } from "@/lib/store.svelte";
  import { S } from "@/strings";

  let {
    sessionId,
    draft,
    onCreated,
  }: {
    sessionId: string | null;
    /** A token that changes on every New chat click, `null` when not drafting. */
    draft: number | null;
    onCreated: (id: string) => void;
  } = $props();

  const messages = $derived(sessionId ? (store.messages[sessionId] ?? []) : []);
  const running = $derived(sessionId ? !!store.running[sessionId] : false);

  // Follows the end of the thread while the user is there; scrolling up lets go.
  let scroller = $state<HTMLDivElement>();
  let atBottom = true;
  function onScroll() {
    if (!scroller) return;
    atBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 24;
  }
  $effect(() => {
    void sessionId;
    atBottom = true;
  });
  $effect(() => {
    void messages.length;
    void running;
    if (!atBottom) return;
    void tick().then(() => scroller && (scroller.scrollTop = scroller.scrollHeight));
  });
</script>

{#if !sessionId && draft === null}
  <EmptyState text={S.noSession} />
{:else if !sessionId}
  <!-- A draft has nothing to scroll, so the greeting and the composer sit together in the
       middle; sending creates the session and the thread takes its usual shape. -->
  <div class="flex min-h-0 flex-1 flex-col justify-center pb-[12vh]">
    <h2 class="mb-6 text-center text-[26px] font-semibold tracking-tight">{S.greeting}</h2>
    <Composer sessionId={null} {draft} {onCreated} />
  </div>
{:else}
  <div
    bind:this={scroller}
    onscroll={onScroll}
    class="mask-fade-y min-h-0 flex-1 overflow-y-auto px-6 pt-[calc(var(--fade-top)+0.5rem)] pb-[calc(var(--fade-bottom)+0.5rem)]"
  >
    <div class="mx-auto flex max-w-3xl flex-col gap-2">
      {#each messages as m, i (m.id)}
        <div class={[m.role === "user" && i > 0 && "mt-4"]}>
          {#if m.role === "user"}
            <UserBubble message={m} />
          {:else}
            <AssistantBubble message={m} />
          {/if}
        </div>
      {/each}
      {#if running}
        <span class="animate-pulse text-sm text-muted-foreground">…</span>
      {/if}
    </div>
  </div>
  <Composer {sessionId} {draft} {onCreated} />
{/if}
