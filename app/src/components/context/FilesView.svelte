<!--
  A context's tree exactly as it is on disk: one directory at a time, re-read on every
  change, nothing hidden and nothing followed. What an agent sees mounted is this.
-->
<script module lang="ts">
  import type { Segments } from "@/lib/contexts.svelte";
  import type { Entry } from "@/lib/viewers/entry";

  /** Where each context was left: the view unmounts with its tab, and coming back finds it as it was. */
  const kept = new Map<string, { cwd: Segments; viewing: Entry | null }>();
</script>

<script lang="ts">
  import { ChevronRight, Download, File, Folder, FolderPlus, Globe, Link, Plus, Trash, Upload } from "@lucide/svelte";
  import { open, save } from "@tauri-apps/plugin-dialog";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { untrack } from "svelte";

  import AddWebPageDialog from "@/components/context/AddWebPageDialog.svelte";
  import { confirm } from "@/lib/confirm";
  import { helpers } from "@/lib/helpers.svelte";
  import {
    addFiles,
    type Context,
    exportContext,
    type FileEntry,
    listFiles,
    makeDir,
    removeFile,
  } from "@/lib/contexts.svelte";
  import { formatSize } from "@/lib/files";
  import { btn } from "@/lib/ui";
  import FileViewer from "@/lib/viewers/FileViewer.svelte";
  import { contextSource } from "@/lib/viewers/source";
  import { S } from "@/strings";

  let { context }: { context: Context } = $props();

  const left = untrack(() => kept.get(context.id));
  let cwd = $state<Segments>(left?.cwd ?? []);
  /** The name of the entry the user clicked in this folder, if any. */
  let selected = $state<string | null>(null);
  let entries = $state<FileEntry[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let dragging = $state(false);
  let askingUrl = $state(false);
  /** The file open in the viewer, if any. */
  let viewing = $state<Entry | null>(left?.viewing ?? null);
  $effect(() => {
    kept.set(context.id, { cwd: $state.snapshot(cwd), viewing: $state.snapshot(viewing) });
  });
  const source = $derived(contextSource(context));

  const asEntry = (e: FileEntry): Entry => ({
    name: e.name,
    segments: [...cwd, e.name],
    type: e.kind === "folder" ? "folder" : "file",
    size: e.kind === "file" ? e.size : null,
    modified: e.modified,
  });

  const messageOf = (e: unknown) => (e instanceof Error ? e.message : String(e));

  async function load() {
    loading = true;
    try {
      entries = await listFiles(context.id, cwd);
      error = null;
    } catch (e) {
      entries = [];
      error = messageOf(e);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void cwd;
    selected = null;
    void load();
  });

  // The helper beside this view may have changed the tree: show it as it now is. The view
  // is keyed by context, so the count it starts from is read once.
  let seen = untrack(() => helpers.context.revision[context.id] ?? 0);
  $effect(() => {
    const revision = helpers.context.revision[context.id] ?? 0;
    if (revision === seen) return;
    seen = revision;
    void load();
  });

  /** Runs a change to the tree, then shows the tree as it now is — whether or not it worked. */
  async function change(run: () => Promise<void>) {
    try {
      await run();
      error = null;
    } catch (e) {
      error = messageOf(e);
    }
    await load();
  }

  /** What a double-click or Enter does: a folder is entered, a file opened in the viewer. */
  function enter(entry: FileEntry) {
    if (entry.kind === "folder") cwd = [...cwd, entry.name];
    else if (entry.kind === "file") viewing = asEntry(entry);
  }

  async function newFolder() {
    const taken = new Set(entries.map((e) => e.name));
    let name: string = S.newFolder;
    for (let n = 2; taken.has(name); n++) name = `${S.newFolder} ${n}`;
    await change(() => makeDir(context.id, [...cwd, name]));
  }

  async function pick(directory: boolean) {
    const picked = await open({ multiple: true, directory });
    if (!picked) return;
    const sources = Array.isArray(picked) ? picked : [picked];
    if (sources.length) await change(() => addFiles(context.id, cwd, sources));
  }

  /** Saves the page at `url` into this folder as HTML. Not wired to the backend yet. */
  function addWebPage(url: string) {
    console.info("add web page", url, "into", [context.id, ...cwd].join("/"));
  }

  /** The whole tree, wherever the user picks, as one archive. */
  async function exportTree() {
    const to = await save({
      defaultPath: `${context.name}.tar.gz`,
      filters: [{ name: "Archive", extensions: ["tar.gz", "tgz"] }],
    });
    if (!to) return;
    try {
      await exportContext(context.id, to);
      error = null;
    } catch (e) {
      error = S.exportFailed(messageOf(e));
    }
  }

  async function remove(entry: FileEntry) {
    if (!(await confirm(S.confirmDeleteFile(entry.name), S.delete))) return;
    await change(() => removeFile(context.id, [...cwd, entry.name]));
  }

  // Files dropped from Finder arrive as paths through Tauri, not as browser `File`s.
  $effect(() => {
    if (!("__TAURI_INTERNALS__" in window)) return;
    const unlisten = getCurrentWebview().onDragDropEvent((event) => {
      const { type } = event.payload;
      if (type === "enter" || type === "over") dragging = true;
      else if (type === "leave") dragging = false;
      else if (type === "drop") {
        dragging = false;
        const { paths } = event.payload;
        if (paths.length) void change(() => addFiles(context.id, cwd, paths));
      }
    });
    return () => void unlisten.then((stop) => stop());
  });

  const formatDate = (ms: number | null) => (ms === null ? "—" : new Date(ms).toLocaleString());

  const icons = { folder: Folder, file: File, link: Link };
  const crumb = "min-w-0 truncate rounded px-1.5 py-0.5 hover:bg-accent";
  const rowAction =
    "rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100";
</script>

<section class="relative flex min-h-0 min-w-0 flex-1 flex-col border-t">
  <header class="flex flex-wrap items-center gap-2 border-b px-4 py-2">
    <nav class="flex min-w-0 flex-1 items-center gap-0.5 text-sm text-muted-foreground" aria-label={S.path}>
      <button class={[crumb, cwd.length === 0 && "font-medium text-foreground"]} onclick={() => (cwd = [])}>
        {context.name}
      </button>
      {#each cwd as segment, i (i)}
        <ChevronRight class="size-3.5 shrink-0" />
        <button
          class={[crumb, i === cwd.length - 1 && "font-medium text-foreground"]}
          onclick={() => (cwd = cwd.slice(0, i + 1))}
        >
          {segment}
        </button>
      {/each}
    </nav>
    <button class={btn.outline} onclick={() => void newFolder()}><FolderPlus class="size-3.5" />{S.newFolder}</button>
    <button class={btn.outline} onclick={() => void pick(true)}><Plus class="size-3.5" />{S.addFolder}</button>
    <button class={btn.outline} onclick={() => void pick(false)}><Upload class="size-3.5" />{S.addFiles}</button>
    <button class={btn.outline} onclick={() => (askingUrl = true)}><Globe class="size-3.5" />{S.addWebPage}</button>
    <button class={btn.outline} onclick={() => void exportTree()}><Download class="size-3.5" />{S.export}</button>
  </header>

  {#if error}
    <p class="border-b bg-destructive/10 px-4 py-2 text-xs text-destructive">{error}</p>
  {/if}

  <div class="min-h-0 flex-1 overflow-y-auto">
    {#if loading && !entries.length}
      <p class="p-6 text-center text-sm text-muted-foreground">{S.loading}</p>
    {:else if entries.length === 0}
      <div class="grid h-full place-items-center p-6 text-center text-sm text-muted-foreground">
        <div class="space-y-1">
          <p>{S.emptyFolder}</p>
          <p class="text-xs">{S.dropHint}</p>
        </div>
      </div>
    {:else}
      <table class="w-full text-sm">
        <thead class="sticky top-0 bg-background text-left text-xs text-muted-foreground">
          <tr class="border-b">
            <th class="px-4 py-2 font-medium">{S.name}</th>
            <th class="w-24 px-2 py-2 text-right font-medium">{S.size}</th>
            <th class="w-44 px-2 py-2 font-medium">{S.modified}</th>
            <th class="w-10"></th>
          </tr>
        </thead>
        <tbody>
          {#each entries as entry (entry.name)}
            {@const Icon = icons[entry.kind]}
            <tr class={["group border-b", selected === entry.name ? "bg-accent" : "hover:bg-accent/60"]}>
              <td class="max-w-0 px-4 py-1.5">
                <button
                  class="flex w-full min-w-0 cursor-default items-center gap-2 text-left"
                  aria-pressed={selected === entry.name}
                  onclick={() => (selected = entry.name)}
                  ondblclick={() => enter(entry)}
                  onkeydown={(e) => {
                    if (e.key !== "Enter") return;
                    e.preventDefault();
                    enter(entry);
                  }}
                  title={entry.name}
                >
                  <Icon class="size-4 shrink-0 text-muted-foreground" />
                  <span class="truncate">{entry.name}</span>
                </button>
              </td>
              <td class="px-2 py-1.5 text-right whitespace-nowrap text-muted-foreground tabular-nums">
                {entry.kind === "file" ? formatSize(entry.size) : "—"}
              </td>
              <td class="px-2 py-1.5 whitespace-nowrap text-muted-foreground">{formatDate(entry.modified)}</td>
              <td class="pr-3 text-right whitespace-nowrap">
                <button class={rowAction} aria-label={`${S.delete} ${entry.name}`} onclick={() => void remove(entry)}>
                  <Trash class="size-3.5" />
                </button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  <footer class="flex items-center gap-2 border-t px-4 py-1.5 text-xs text-muted-foreground">
    <span>{S.items(entries.length)}</span>
    <span class="ml-auto min-w-0 truncate font-mono" title={context.dir}>{[context.dir, ...cwd].join("/")}</span>
  </footer>

  {#if viewing}
    <FileViewer entry={viewing} {source} onClose={() => (viewing = null)} />
  {/if}

  <AddWebPageDialog bind:open={askingUrl} onAdd={addWebPage} />

  {#if dragging}
    <div class="pointer-events-none absolute inset-0 grid place-items-center bg-background/70 backdrop-blur-[2px]">
      <div class="rounded-xl border-2 border-dashed border-primary px-10 py-8 text-center text-sm">
        <p class="font-medium">{S.dropHere}</p>
        <p class="text-xs text-muted-foreground">{cwd.at(-1) ?? context.name}</p>
      </div>
    </div>
  {/if}
</section>
