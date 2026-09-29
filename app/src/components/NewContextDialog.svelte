<!-- Asks for a new context's name, offering a made-up one so Enter alone is enough. -->
<script lang="ts">
  import { adjectives, animals, uniqueNamesGenerator } from "unique-names-generator";

  import { btn, input } from "@/lib/ui";
  import { S } from "@/strings";

  let { open = $bindable(false), onCreate }: { open: boolean; onCreate: (name: string) => void } = $props();

  let dialog: HTMLDialogElement;
  let name = $state("");
  const trimmed = $derived(name.trim());

  const suggest = () => uniqueNamesGenerator({ dictionaries: [adjectives, animals], separator: "-", length: 2 });

  // A native `<dialog>` for the focus trap, Escape and the top layer; `open` drives it.
  $effect(() => {
    if (open && !dialog.open) {
      name = suggest();
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  });

  function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!trimmed) return;
    onCreate(trimmed);
    open = false;
  }
  function focus(node: HTMLInputElement) {
    node.select();
  }
</script>

<dialog
  bind:this={dialog}
  aria-labelledby="new-context-title"
  class="m-auto w-[min(400px,calc(100vw-32px))] rounded-xl border bg-background p-0 text-foreground shadow-lg backdrop:bg-background/55 backdrop:backdrop-blur-[2px]"
  onclose={() => (open = false)}
  onclick={(e) => e.target === dialog && (open = false)}
>
  {#if open}
    <form class="space-y-4 p-5" onsubmit={submit}>
      <h2 id="new-context-title" class="text-base font-medium">{S.newContext}</h2>
      <label class="block space-y-1.5">
        <span class="text-xs text-muted-foreground">{S.contextName}</span>
        <input use:focus class={input} bind:value={name} autocomplete="off" spellcheck="false" />
      </label>
      <div class="flex justify-end gap-2">
        <button type="button" class={btn.outline} onclick={() => (open = false)}>{S.cancel}</button>
        <button type="submit" class={btn.primary} disabled={!trimmed}>{S.create}</button>
      </div>
    </form>
  {/if}
</dialog>
