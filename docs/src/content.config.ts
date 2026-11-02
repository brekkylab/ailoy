import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Starlight's docs live under src/content/docs/, served at /docs/ through the site's base.
export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() })
};
