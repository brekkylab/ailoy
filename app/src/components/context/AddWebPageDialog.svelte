<!-- Asks for the URL of a page to save into the open folder as HTML. -->
<script lang="ts">
  import { Globe, LoaderCircle, SquarePlay, TriangleAlert } from "@lucide/svelte";

  import { peekUrl, type UrlPeek } from "@/lib/contexts.svelte";
  import { btn, input } from "@/lib/ui";
  import { S } from "@/strings";

  let { open = $bindable(false), onAdd }: { open: boolean; onAdd: (url: string) => void } = $props();

  let dialog: HTMLDialogElement;
  let text = $state("");

  /** What was typed as a web address, `https://` supplied when there is no scheme; `null` when it is not one. */
  const url = $derived.by(() => {
    const trimmed = text.trim();
    if (!trimmed) return null;
    try {
      const parsed = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`);
      return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
    } catch {
      return null;
    }
  });

  /** What the address was found to be; `"error"` when it could not be reached. Follows `url` after a pause in typing. */
  let peek = $state<UrlPeek | "error" | null>(null);
  let peeking = $state(false);
  $effect(() => {
    const at = url;
    peek = null;
    peeking = false;
    if (!at) return;
    let stale = false;
    const timer = setTimeout(async () => {
      peeking = true;
      try {
        const found = await peekUrl(at);
        if (!stale) peek = found;
      } catch {
        if (!stale) peek = "error";
      } finally {
        if (!stale) peeking = false;
      }
    }, 400);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  });

  // A native `<dialog>` for the focus trap, Escape and the top layer; `open` drives it.
  $effect(() => {
    if (open && !dialog.open) {
      text = "";
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  });

  function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!url) return;
    onAdd(url);
    open = false;
  }
  function focus(node: HTMLInputElement) {
    node.focus();
  }
</script>

<dialog
  bind:this={dialog}
  aria-labelledby="add-web-page-title"
  class="m-auto w-[min(440px,calc(100vw-32px))] rounded-xl border bg-background p-0 text-foreground shadow-lg backdrop:bg-background/55 backdrop:backdrop-blur-[2px]"
  onclose={() => (open = false)}
  onclick={(e) => e.target === dialog && (open = false)}
>
  {#if open}
    <form class="space-y-4 p-5" onsubmit={submit}>
      <h2 id="add-web-page-title" class="text-base font-medium">{S.addWebPage}</h2>
      <label class="block space-y-1.5">
        <span class="text-xs text-muted-foreground">{S.url}</span>
        <input
          use:focus
          class={input}
          bind:value={text}
          type="url"
          placeholder="https://"
          autocomplete="off"
          spellcheck="false"
        />
        <span class="block text-xs text-muted-foreground">{S.webPageHint}</span>
      </label>
      {#if peeking || peek}
        <div class="flex items-start gap-2.5 rounded-lg border bg-muted/40 px-3 py-2.5" aria-live="polite">
          {#if peeking}
            <LoaderCircle class="mt-0.5 size-4 shrink-0 animate-spin text-muted-foreground" />
            <span class="text-sm text-muted-foreground">{S.lookingUp}</span>
          {:else if peek === "error"}
            <TriangleAlert class="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <span class="text-sm text-muted-foreground">{S.unreachable}</span>
          {:else if peek}
            {#if peek.kind === "youtube"}
              <SquarePlay class="mt-0.5 size-4 shrink-0 text-red-600" />
            {:else}
              <Globe class="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            {/if}
            <div class="min-w-0 flex-1">
              <div class="truncate text-sm font-medium {peek.title ? '' : 'text-muted-foreground'}" title={peek.title}>
                {peek.title ?? S.noTitle}
              </div>
              <div class="truncate text-xs text-muted-foreground">
                {peek.kind === "youtube" ? S.youtubeVideo : S.webPage}
                {#if peek.author}· {peek.author}{/if}
                {#if url}· {new URL(url).host}{/if}
              </div>
            </div>
          {/if}
        </div>
      {/if}
      <div class="flex justify-end gap-2">
        <button type="button" class={btn.outline} onclick={() => (open = false)}>{S.cancel}</button>
        <button type="submit" class={btn.primary} disabled={!url}>{S.add}</button>
      </div>
    </form>
  {/if}
</dialog>
