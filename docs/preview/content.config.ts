import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

// The pages are in content/, not Starlight's default src/content/docs/: the folder holds the
// docs and nothing else, and ailoy.co reads the same folder. `i18n` overrides Starlight's own UI
// strings (preview/content/i18n/).
export const collections = {
  docs: defineCollection({ loader: glob({ base: './content', pattern: '**/*.{md,mdx}' }), schema: docsSchema() }),
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() })
};
