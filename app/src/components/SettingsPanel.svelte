<script lang="ts">
  import { Cpu, Palette } from "@lucide/svelte";

  import AppearancePage from "@/components/settings/AppearancePage.svelte";
  import ModelsPage from "@/components/settings/ModelsPage.svelte";
  import { S } from "@/strings";

  const PAGES = [
    { id: "models", label: S.modelsPage, Icon: Cpu, Page: ModelsPage },
    { id: "appearance", label: S.appearancePage, Icon: Palette, Page: AppearancePage },
  ];

  // Lands on the models page each time: it is the one a new install has to visit.
  let active = $state(PAGES[0].id);
  const Page = $derived(PAGES.find((p) => p.id === active)!.Page);
</script>

<section class="flex min-h-0 min-w-0 flex-1 flex-col">
  <div class="border-b px-6 pb-2">
    <div role="tablist" aria-label={S.settings} class="flex gap-1">
      {#each PAGES as { id, label, Icon } (id)}
        <button
          role="tab"
          aria-selected={active === id}
          class={[
            "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-sm transition-colors hover:bg-accent hover:text-foreground",
            active === id ? "bg-accent font-medium text-foreground" : "text-muted-foreground",
          ]}
          onclick={() => (active = id)}
        >
          <Icon class="size-4 shrink-0" />
          {label}
        </button>
      {/each}
    </div>
  </div>
  <div class="min-h-0 flex-1 overflow-y-auto px-6 pt-5 pb-10">
    <div class="mx-auto max-w-3xl" role="tabpanel">
      <Page />
    </div>
  </div>
</section>
