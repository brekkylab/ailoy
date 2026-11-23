"""Writes the Python reference: one page per class in ``ailoy.__all__``, one for its functions and
one for ``ailoy.types``, plus an overview.

Usage (regen.mjs runs it): gen_py.py <the tag's bindings/python/python> <out dir> <tag>

Signatures come from the tag's stubs, which griffe reads without importing the native module.
Docstrings the stubs leave out come from the installed ailoy-py of the same version, whose native
module has them from the Rust doc comments.
"""

import re
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import ailoy as runtime
import griffe
import griffe2md

src, out_dir, tag = sys.argv[1:4]
out = Path(out_dir)
version = tag.removeprefix("v")

if installed_version("ailoy-py") != version:
    sys.exit(f"gen_py.py: ailoy-py {installed_version('ailoy-py')} is installed; {tag} needs ailoy-py=={version}")

pkg = griffe.load("ailoy", search_paths=[src], resolve_aliases=True, allow_inspection=False)
exported = sorted(pkg.exports or [])

# A class page's description, where the class's docstring makes a poor one.
DESCRIPTIONS = {
    "AgentBuilder": "sets up an agent and builds it.",
    "Agent": "runs turns and holds the history.",
    "AgentRun": "one turn, iterated with async for.",
    "LangModel": "a model run without an agent.",
    "AiloyError": "the exception ailoy raises.",
}

config = {
    **griffe2md.default_config,
    "search_paths": [src],
    "show_root_full_path": False,
    "show_object_full_path": False,
    "show_root_members_full_path": False,
    "show_signature_annotations": True,
    "members_order": "source",
    "heading_level": 2,
    "docstring_style": None,
    "summary": False,
    "show_submodules": False,
    # bases would link to anchors the site doesn't have (TypedDict, Protocol)
    "show_bases": False,
}


def kebab(name):
    return re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name).lower()


def target(obj):
    return obj.final_target if obj.is_alias else obj


def fill_docstrings(obj, live):
    if live is None:
        return
    obj = target(obj)
    doc = (getattr(live, "__doc__", None) or "").strip()
    if doc and not obj.docstring and not obj.is_module:
        obj.docstring = griffe.Docstring(doc, parent=obj)
    if obj.is_class:
        for name, member in obj.members.items():
            if not name.startswith("_"):
                fill_docstrings(member, getattr(live, name, None))


def functions_in(obj):
    obj = target(obj)
    if obj.is_function:
        yield obj
    elif obj.is_class:
        yield from (target(m) for m in obj.members.values() if target(m).is_function)


def render(name, heading):
    obj = pkg[name]
    cfg = {**config, "show_root_heading": heading}
    if target(obj).is_module:
        # what the module exports, not what it imports
        cfg["members"] = sorted(target(obj).exports or [])
    body = griffe2md.render_object_docs(obj, cfg)
    # griffe2md leaves async and staticmethod out of signatures
    for f in functions_in(obj):
        prefix = ("@staticmethod\n" if "staticmethod" in f.labels else "") + ("async " if "async" in f.labels else "")
        if prefix:
            body = re.sub(rf"(```python\n){re.escape(f.name)}\(", rf"\g<1>{prefix}{f.name}(", body, count=1)
    return body


def page(title, desc, body):
    return f"""---
title: {title}
description: "{desc}"
head:
  - tag: meta
    attrs:
      name: robots
      content: noindex
---

*Generated from the [{tag}](https://github.com/brekkylab/ailoy/tree/{tag}) release of `ailoy-py`.*

{body}"""


def clean(text):
    # rustdoc links: to a crate module -> that module on docs.rs; any other [`X`] -> `X`
    text = re.sub(
        r"\[`crate::([\w:]+)`\](?!\()",
        lambda m: f"[`crate::{m[1]}`](https://docs.rs/ailoy/{version}/ailoy/{m[1].replace('::', '/')}/index.html)",
        text,
    )
    text = re.sub(r"\[(`[^`\]]+`)\](?!\()", r"\1", text)
    # the docs use no em dashes; the docstrings use them as commas
    text = re.sub(r"\s*\u2014\s*", ", ", text)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip("\n") + "\n"


classes = [n for n in exported if target(pkg[n]).is_class]
functions = [n for n in exported if target(pkg[n]).is_function]
modules = [n for n in exported if target(pkg[n]).is_module]
if modules != ["types"] or len(classes) + len(functions) + 1 != len(exported):
    sys.exit(f"gen_py.py: ailoy exports something with no page here: {exported}")

for name in classes + functions:
    fill_docstrings(pkg[name], getattr(runtime, name))

# class order: the order a reader meets them, then any new ones by name
order = ["AgentBuilder", "Agent", "AgentRun", "LangModel"]
classes = sorted(classes, key=lambda n: (order.index(n) if n in order else len(order), n == "AiloyError", n))

pages = []  # (slug, title, what it covers, body)
for name in classes:
    first = (target(pkg[name]).docstring.value.split(". ")[0].rstrip(".") + ".") if target(pkg[name]).docstring else ""
    pages.append((kebab(name), name, DESCRIPTIONS.get(name, first or f"the {name} class."), render(name, False)))
pages.insert(
    len([c for c in classes if c != "AiloyError"]),
    ("functions", "Functions", "the module functions: model and tool providers, and MCP.",
     "\n".join(render(n, True) for n in functions)),
)
pages.insert(
    len(pages) - (1 if "AiloyError" in classes else 0),
    ("types", "Types", "the dict shapes in ailoy.types of messages, tools and specs.", render("types", True)),
)

out.mkdir(parents=True, exist_ok=True)
for slug, title, covers, body in pages:
    desc = f"Python reference for {title}: {covers}" if title not in ("Functions", "Types") else f"Python reference for {covers}"
    (out / f"{slug}.md").write_text(clean(page(title, desc, body)), encoding="utf-8", newline="\n")

rows = "\n".join(f"| [{title}](/docs/reference/python/{slug}/) | {covers[0].upper() + covers[1:]} |" for slug, title, covers, _ in pages)
overview = f"""The package is `ailoy-py` on PyPI. Every name below is imported from `ailoy`.

| Page | What it covers |
| --- | --- |
{rows}
"""
(out / "index.md").write_text(
    clean(page("Python API reference", f"The Python API of ailoy-py {version}, generated from {tag}.", overview)),
    encoding="utf-8",
    newline="\n",
)

# the sidebar's Python entries, for regen.mjs
print(" ".join(slug for slug, _, _, _ in pages))
