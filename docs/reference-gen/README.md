# Reference generator

Writes the API reference in `docs/content/reference/` from a release. The pages are generated, so a fix goes here, not in a page.

```sh
cd docs/reference-gen
npm run regen -- v0.3.2   # the release's tag, fetched (git fetch --tags)
cd .. && npm run build    # then look at the pages
```

It needs Node.js, git and [uv](https://docs.astral.sh/uv/). For the tag it:

- checks the tag out in a temporary worktree, removed at the end;
- **Python**: reads the signatures from the tag's stubs (`bindings/python/python/ailoy/`) with griffe, takes the docstrings the stubs leave out from `ailoy-py` of the same version on PyPI, and writes one page per class in `ailoy.__all__`, a Functions page, a Types page and an overview (`gen_py.py`);
- **Node.js**: installs `@brekkylab/ailoy` of the same version from npm, since its `binding.d.ts` is written by napi at build, and runs typedoc over it, one page per class, function and type (`typedoc.json`, `starlight-plugin.mjs`);
- writes the reference's overview, which links Rust on docs.rs, and replaces the reference groups at the end of `docs/sidebar.json`.

Every page carries `noindex` and a line naming the release.
