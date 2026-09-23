<!-- Asks for the URL of a page to save into the open folder as HTML. -->
<script lang="ts">
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
      <div class="flex justify-end gap-2">
        <button type="button" class={btn.outline} onclick={() => (open = false)}>{S.cancel}</button>
        <button type="submit" class={btn.primary} disabled={!url}>{S.add}</button>
      </div>
    </form>
  {/if}
</dialog>
