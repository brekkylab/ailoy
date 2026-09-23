<!--
  The Agent helper: a pane on the right of a tab, where an agent is asked to change what
  the tab shows. Closed, it leaves a rail that opens it again — unmounted rather than hidden,
  like the sidebar, so it takes no tab stops.
-->
<script lang="ts">
  import { ArrowUp, Eraser, PanelRightClose, PanelRightOpen, Square } from "@lucide/svelte";
  import { tick } from "svelte";

  import AssistantBubble from "@/components/thread/AssistantBubble.svelte";
  import ModelPicker from "@/components/thread/ModelPicker.svelte";
  import UserBubble from "@/components/thread/UserBubble.svelte";
  import { type HelperId, helpers } from "@/lib/helpers.svelte";
  import { btn } from "@/lib/ui";
  import { S } from "@/strings";

  let {
    helper: helperId,
    thread,
    intro,
    starters,
    placeholder,
    open,
    onToggle,
  }: {
    helper: HelperId;
    /** What the thread is about: a context's id, an agent's. */
    thread: string;
    intro: string;
    starters: readonly string[];
    placeholder: string;
    open: boolean;
    onToggle: () => void;
  } = $props();

  const helper = $derived(helpers[helperId]);
  $effect(() => void helper.load());

  const messages = $derived(helper.messages[thread] ?? []);
  const running = $derived(!!helper.running[thread]);

  let text = $state("");
  let box = $state<HTMLTextAreaElement>();
  const canSend = $derived(text.trim().length > 0 && !running);

  // Follows the end of the thread while the user is there; scrolling up lets go.
  let scroller = $state<HTMLDivElement>();
  let atBottom = true;
  function onScroll() {
    if (!scroller) return;
    atBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 24;
  }
  $effect(() => {
    void thread;
    if (!open) return;
    atBottom = true;
    void tick().then(() => box?.focus());
  });
  $effect(() => {
    void messages.length;
    void running;
    if (!atBottom) return;
    void tick().then(() => scroller && (scroller.scrollTop = scroller.scrollHeight));
  });

  // Grows with its content up to `max-h-48`.
  $effect(() => {
    void text;
    if (!box) return;
    box.style.height = "auto";
    box.style.height = `${box.scrollHeight}px`;
  });

  function send() {
    if (!canSend) return;
    helper.send(thread, text.trim());
    text = "";
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  }
</script>

{#if !open}
  <button
    class="flex w-10 shrink-0 flex-col items-center gap-3 border-t border-l pt-3 text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground"
    onclick={onToggle}
    aria-label={S.showHelper}
    title={S.showHelper}
  >
    <PanelRightOpen class="size-4 shrink-0" />
    <span class="text-xs font-medium [writing-mode:vertical-rl]">{S.helper}</span>
  </button>
{:else}
  <aside class="flex min-h-0 w-[380px] shrink-0 flex-col border-t border-l" aria-label={S.helper}>
    <header class="flex h-[45px] shrink-0 items-center gap-1 border-b pr-1.5 pl-4">
      <h2 class="min-w-0 flex-1 truncate text-sm">
        <span class="font-medium">{S.helper}</span>
        {#if helper.agent?.name}
          <span class="text-muted-foreground"> · {helper.agent.name}</span>
        {/if}
      </h2>
      <button
        class={btn.ghostIcon}
        onclick={() => helper.clear(thread)}
        disabled={!messages.length}
        aria-label={S.clearChat}
        title={S.clearChat}
      >
        <Eraser class="size-4" />
      </button>
      <button class={btn.ghostIcon} onclick={onToggle} aria-label={S.hideHelper} title={S.hideHelper}>
        <PanelRightClose class="size-4" />
      </button>
    </header>

    {#if messages.length === 0}
      <div class="grid min-h-0 flex-1 place-items-center p-6">
        <div class="space-y-4 text-center">
          <p class="text-sm text-muted-foreground">{intro}</p>
          <div class="flex flex-col items-center gap-2">
            {#each starters as starter (starter)}
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
        </div>
      </div>
    {:else}
      <div bind:this={scroller} onscroll={onScroll} class="min-h-0 flex-1 overflow-y-auto px-4 py-4 text-sm">
        <div class="flex flex-col gap-2">
          {#each messages as m, i (m.id)}
            <div class={[m.role === "user" && i > 0 && "mt-3"]}>
              {#if m.role === "user"}
                <UserBubble message={m} />
              {:else}
                <AssistantBubble message={m} />
              {/if}
            </div>
          {/each}
          {#if running}
            <span class="animate-pulse text-muted-foreground">…</span>
          {/if}
        </div>
      </div>
    {/if}

    <div class="px-3 pt-1 pb-3">
      <div class="rounded-xl border bg-card shadow-sm transition-colors focus-within:border-ring/60">
        <textarea
          bind:this={box}
          bind:value={text}
          onkeydown={onKey}
          {placeholder}
          rows="1"
          aria-label={S.messageInput}
          class="block max-h-48 min-h-10 w-full resize-none border-0 bg-transparent px-3 pt-2.5 pb-1 text-sm outline-none placeholder:text-muted-foreground"
        ></textarea>
        <div class="flex items-center gap-1 px-1.5 pb-1.5">
          <ModelPicker value={helper.model} onChange={(m) => (helper.picked = m)} disabled={running} />
          <div class="ml-auto">
            {#if running}
              <button class={btn.primaryIcon} onclick={() => helper.stop(thread)} aria-label={S.stop}>
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
    </div>
  </aside>
{/if}
