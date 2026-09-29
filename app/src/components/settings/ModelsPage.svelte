<!-- Keys are held in memory only for now; the backend will own them. -->
<script lang="ts">
  import Section from "@/components/settings/Section.svelte";
  import { BEDROCK_REGIONS, MODELS, PROVIDERS } from "@/lib/mock";
  import { store } from "@/lib/store.svelte";
  import { btn, input } from "@/lib/ui";
  import { S } from "@/strings";

  let drafts = $state<Record<string, string>>({});

  function save(provider: string) {
    const key = drafts[provider]?.trim();
    if (!key) return;
    store.keys[provider] = key;
    drafts[provider] = "";
  }
  const labelOf = (key: string) => PROVIDERS.find((p) => p.key === key)?.label ?? key;
</script>

<div class="space-y-8">
  <Section title={S.apiKey}>
    <div class="divide-y rounded-lg border">
      {#each PROVIDERS as { key, label } (key)}
        <div class="space-y-2 p-4">
          <div class="flex items-center justify-between">
            <span class="text-sm font-medium">{label}</span>
            <span class="text-xs text-muted-foreground">
              {#if store.keys[key]}
                {S.keySaved} · …{store.keys[key].slice(-4)}
              {:else}
                {S.noKeyYet}
              {/if}
            </span>
          </div>
          <div class="flex gap-2">
            <input
              type="password"
              class={input}
              placeholder={key === "bedrock" ? S.bedrockKey : S.apiKey}
              aria-label={`${label} ${S.apiKey}`}
              bind:value={drafts[key]}
              onkeydown={(e) => e.key === "Enter" && save(key)}
            />
            <button class={btn.primary} onclick={() => save(key)} disabled={!drafts[key]?.trim()}>
              {S.saveKey}
            </button>
          </div>
          {#if key === "bedrock"}
            <label class="flex items-center justify-between gap-4 pt-1 text-sm">
              <span>{S.region}</span>
              <select class={[input, "w-48"]} bind:value={store.bedrockRegion}>
                {#each BEDROCK_REGIONS as r (r)}
                  <option value={r}>{r}</option>
                {/each}
              </select>
            </label>
          {/if}
        </div>
      {/each}
    </div>
  </Section>
  <Section title="Defaults">
    <label class="flex items-center justify-between gap-4 text-sm">
      <span>Default model</span>
      <select class={[input, "w-64"]} bind:value={store.defaultModel}>
        {#each MODELS as m (m.id)}
          <option value={m.id}>{m.name} ({labelOf(m.provider)})</option>
        {/each}
      </select>
    </label>
  </Section>
</div>
