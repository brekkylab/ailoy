<!--
  One file, opened over the pane it was opened from: its name, where it is, and the way
  out. Reading and rendering it is `FileContents`.

  `position: absolute`, so it covers the Files view and not the window — the sidebar
  stays reachable.
-->
<script lang="ts">
  import { File, X } from "@lucide/svelte";

  import { formatSize } from "@/lib/files";
  import { btn } from "@/lib/ui";
  import type { Entry } from "@/lib/viewers/entry";
  import FileContents from "@/lib/viewers/FileContents.svelte";
  import { viewerFor } from "@/lib/viewers/registry";
  import type { Source } from "@/lib/viewers/source";
  import { S } from "@/strings";

  let { entry, source, onClose }: { entry: Entry; source: Source; onClose: () => void } = $props();

  const viewer = $derived(viewerFor(entry));
  let panel = $state<HTMLElement | undefined>(undefined);

  // Takes focus so a screen reader lands inside it and not on what is behind.
  $effect(() => panel?.focus());
</script>

<!-- On the window: a click into the document moves focus, and Escape should still close. -->
<svelte:window onkeydown={(e) => e.key === "Escape" && onClose()} />

<div
  class="absolute inset-0 z-10 grid place-items-center bg-background/60 p-6 backdrop-blur-[3px]"
  role="presentation"
  onclick={(e) => e.target === e.currentTarget && onClose()}
>
  <div
    bind:this={panel}
    class="flex h-full max-h-[880px] w-full max-w-[1100px] flex-col overflow-hidden rounded-xl border bg-card shadow-lg outline-none"
    role="dialog"
    aria-modal="true"
    aria-label={entry.name}
    tabindex="-1"
  >
    <header class="flex shrink-0 items-center gap-2.5 border-b px-3.5 py-2.5">
      <File class="size-4 shrink-0 text-muted-foreground" />
      <div class="flex min-w-0 flex-1 flex-col">
        <strong class="truncate text-sm font-medium">{entry.name}</strong>
        <span class="truncate text-xs text-muted-foreground">
          {source.where(entry)} · {formatSize(entry.size)}{#if viewer}&nbsp;· {viewer.label}{/if}
        </span>
      </div>
      <button class={btn.ghostIcon} onclick={onClose} aria-label={S.closeViewer}>
        <X class="size-4" />
      </button>
    </header>
    <!-- The viewers come from agent-s and speak its token names; `viewer-tokens` maps them. -->
    <div class="viewer-tokens min-h-0 flex-1 overflow-auto">
      <FileContents {entry} {source} />
    </div>
  </div>
</div>
