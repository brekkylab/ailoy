# Ailoy docs

The pages of the Ailoy documentation, kept next to the code they describe. ailoy.co reads this
folder and serves it at <https://ailoy.co/docs/>.

```
content/      the pages, as MDX; the path is the address: guides/tools.mdx is /docs/guides/tools/
sidebar.json  the sidebar: its groups and the order of the pages
images/       pictures the pages and the repo's README.md show
preview/      the preview's own code: content config, ailoy.co's header and styles, the link check
public/       the favicon and the logos (README.md links the logos by URL)
```

## Preview

```sh
npm ci
npm run dev      # http://localhost:4321/docs/
npm run build    # builds the preview, then checks every internal link and #anchor
```

The preview uses ailoy.co's header, theme and styles, so a page looks here as it does there.

## What a page may use

A page is rendered by the preview and by ailoy.co, so it uses only what both have.

- Markdown, and the frontmatter fields Starlight defines (`title`, `description`,
  `tableOfContents`, ...).
- Starlight's own components, imported from `@astrojs/starlight/components` (`Tabs`, `TabItem`,
  `Aside`, ...).
- Pictures from `images/`, by a path relative to the page.
- `HowDiagram`, from `preview/components/`, on the concepts page. A component of the preview is
  one ailoy.co has to have too, so add another only with ailoy.co's side.
- Links to other pages as absolute paths with `/docs/`: `[Tools](/docs/guides/tools/)`. They
  resolve the same in both, and the link check verifies them.

## Adding a page

Add the file under `content/` and its entry to `sidebar.json`. A `slug` is the page's path under
`content/` without the extension; a `link` is an address under `/docs/` (`"/"` is the overview).
