# Reference generator

Writes the API reference in `docs/content/reference/` from a release. The pages are generated, so a fix goes here, not in a page.

```sh
cd docs/reference-gen
npm run regen -- v0.3.2   # the release's tag, fetched (git fetch --tags)
cd .. && npm run build    # then look at the pages
```

It needs Node.js, git and [uv](https://docs.astral.sh/uv/). For the tag it:

- checks the tag out in a temporary worktree, removed at the end;
- **Python**: reads the signatures from the tag's stubs (`bindings/python/python/ailoy/`) with griffe, and the docstrings the stubs leave out from `ailoy-py` of the same version on PyPI (`gen_py.py`);
- **Node.js**: installs `@brekkylab/ailoy` of the same version from npm, since its `binding.d.ts` is written by napi at build, and reads it with typedoc's API (`node_api.mjs`);
- type-checks every example in `examples/` against those packages: mypy `--strict` for the `.py` files, tsc in strict `checkJs` for the `.js` files. A wrong call stops the run;
- writes both languages from one model with `render.mjs`, so they have the same pages in the same order;
- copies the shared Get started pages from `shared/`, each `{/* example: name */}` becoming tabs of that example in every language it has;
- replaces the reference groups at the end of `docs/sidebar.json`.

What goes where:

| File | Holds |
| --- | --- |
| `pages.mjs` | The pages, their sidebar groups, each page's intro, and the descriptions for names the source has no doc comment for (`FILL`) |
| `examples/<page>.py`, `.js` | The page's example. `intro.rs` is shown on the shared pages and isn't compiled |
| `shared/*.mdx` | Introduction, Choosing an SDK, Choosing a model provider |
| `render.mjs` | The page layout |

Every page carries `noindex` until the final review.
