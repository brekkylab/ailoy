import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Starlight's docs live under src/content/docs/docs/ so they are served at /docs/.
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() })
};
