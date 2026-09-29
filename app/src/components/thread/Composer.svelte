<script lang="ts">
  import { ArrowUp, Square } from "@lucide/svelte";
  import { tick } from "svelte";

  import AgentPicker from "@/components/thread/AgentPicker.svelte";
  import { store } from "@/lib/store.svelte";
  import { btn } from "@/lib/ui";
  import { S } from "@/strings";

  let {
    sessionId,
    draft,
    onCreated,
  }: { sessionId: string | null; draft: number | null; onCreated: (id: string) => void } = $props();

  let text = $state("");
  let box = $state<HTMLTextAreaElement>();
  // A draft has no session to hold its agent yet, so it keeps one here until the first send.
  // `null` is the default agent, whichever that is by then.
  let draftAgent = $state<string | null>(null);

  const session = $derived(sessionId ? store.sessions.find((s) => s.id === sessionId) : undefined);
  const agent = $derived(session ? session.agent : draftAgent);
  const running = $derived(sessionId ? !!store.running[sessionId] : false);
  const canSend = $derived(text.trim().length > 0 && !running);

  // Back into the box whenever the thread or the draft token changes.
  $effect(() => {
    void sessionId;
    void draft;
    void tick().then(() => box?.focus());
  });

  // Grows with its content up to `max-h-60`.
  $effect(() => {
    void text;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${box.scrollHeight}px`;
  });

  function send() {
    if (!canSend) return;
    const id = store.send(sessionId, text.trim(), agent);
    text = "";
    if (!sessionId) onCreated(id);
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  }
</script>

<div class="px-6 pt-2 pb-4">
  <div class="mx-auto max-w-3xl">
    <!-- The card is the field: the textarea wears no chrome of its own. -->
    <div class="rounded-2xl border bg-card shadow-sm transition-colors focus-within:border-ring/60">
      <textarea
        bind:this={box}
        bind:value={text}
        onkeydown={onKey}
        placeholder={S.composerPlaceholder}
        rows="1"
        aria-label={S.messageInput}
        class="block max-h-60 min-h-12 w-full resize-none border-0 bg-transparent px-4 pt-3.5 pb-1 text-[15px] outline-none placeholder:text-muted-foreground"
      ></textarea>
      <div class="flex items-center gap-1 px-2 pb-2">
        <AgentPicker
          value={agent}
          onChange={(a) => {
            if (sessionId) store.setAgent(sessionId, a);
            else draftAgent = a;
          }}
          disabled={running}
        />
        <div class="ml-auto flex items-center gap-2">
          {#if running}
            <button class={btn.primaryIcon} onclick={() => sessionId && store.stop(sessionId)} aria-label={S.stop}>
              <Square class="size-3.5 fill-current" />
            </button>
          {:else}
            <button class={btn.primaryIcon} onclick={send} disabled={!canSend} aria-label={S.send}>
              <ArrowUp class="size-4" />
            </button>
          {/if}
        </div>
      </div>
    </div>
    {#if sessionId === null && !text}
      <div class="mt-4 flex flex-wrap justify-center gap-2">
        {#each S.starters as starter (starter)}
          <button
            class="rounded-full border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            onclick={() => {
              text = starter;
              box?.focus();
            }}
          >
            {starter}
          </button>
        {/each}
      </div>
    {/if}
  </div>
</div>
