<script lang="ts">
  import { Monitor, Moon, Sun } from "@lucide/svelte";

  import Section from "@/components/settings/Section.svelte";
  import { readTheme, setTheme, type Theme } from "@/lib/theme";
  import { S } from "@/strings";

  const OPTIONS = [
    { id: "system" as Theme, label: S.themeSystem, Icon: Monitor },
    { id: "light" as Theme, label: S.themeLight, Icon: Sun },
    { id: "dark" as Theme, label: S.themeDark, Icon: Moon },
  ];

  let theme = $state<Theme>(readTheme());
  function choose(next: Theme) {
    theme = next;
    setTheme(next);
  }
</script>

<div class="space-y-8">
  <Section title={S.appearanceSection}>
    <div class="space-y-2">
      <div role="radiogroup" aria-label={S.theme} class="flex gap-1 rounded-lg border bg-muted/20 p-1">
        {#each OPTIONS as { id, label, Icon } (id)}
          <button
            role="radio"
            aria-checked={theme === id}
            onclick={() => choose(id)}
            class={[
              "flex flex-1 items-center justify-center gap-2 rounded-md border border-transparent px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
              theme === id && "border-primary/30 bg-primary/10 font-medium text-foreground",
            ]}
          >
            <Icon class="size-4 shrink-0" />
            {label}
          </button>
        {/each}
      </div>
      <p class="text-xs text-muted-foreground">{S.themeHint}</p>
    </div>
  </Section>
</div>
