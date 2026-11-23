"""Reads the Python API of a release into the JSON model render.mjs writes the pages from.

Usage (regen.mjs runs it): gen_py.py <the tag's bindings/python/python> <out.json> <tag>

Signatures come from the tag's stubs, which griffe reads without importing the native module.
Docstrings the stubs leave out come from the installed ailoy-py of the same version, whose native
module has them from the Rust doc comments.

The model maps each name to {"kind": "class" | "function" | "type", "module", "doc", ...}: a
class has "members", a function "params", "returns" and "signature", a type "definition" and
"fields". Node's node_api.mjs writes the same shape.
"""

import json
import re
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import ailoy as runtime
import black
import griffe

src, out_file, tag = sys.argv[1:4]
version = tag.removeprefix("v")

if installed_version("ailoy-py") != version:
    sys.exit(f"gen_py.py: ailoy-py {installed_version('ailoy-py')} is installed; {tag} needs ailoy-py=={version}")

pkg = griffe.load("ailoy", search_paths=[src], resolve_aliases=True, allow_inspection=False)
# what set_lang_model_provider takes, declared in the stubs only
STUB_ONLY = ["CustomLangModel", "StreamingCustomLangModel", "LangModelFactory"]
BLACK = black.Mode(line_length=80)


def target(obj):
    return obj.final_target if obj.is_alias else obj


def text(value):
    # the stubs' private `_Path` is what pyo3 takes as a PathBuf: a str or an os.PathLike
    return None if value is None else re.sub(r"\b_Path\b", "str | os.PathLike[str]", str(value))


def live_doc(owner, name=None):
    """The docstring the native module has, from the Rust doc comments."""
    if owner is None:
        return ""
    obj = owner.__dict__.get(name) if name else owner
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    return (getattr(obj, "__doc__", None) or "").strip()


def doc_of(obj, owner=None, name=None):
    obj = target(obj)
    return obj.docstring.value.strip() if obj.docstring else live_doc(owner, name)


def function(fn, doc):
    fn = target(fn)
    params = [
        {"name": p.name, "type": text(p.annotation), "default": text(p.default), "keyword_only": p.kind.value == "keyword-only"}
        for p in fn.parameters
        if p.name not in ("self", "cls")
    ]
    out = {
        "doc": doc,
        "params": params,
        "returns": text(fn.returns),
        "is_async": "async" in fn.labels,
        "is_static": "staticmethod" in fn.labels,
    }
    out["signature"] = signature(fn.name, out)
    return out


def signature(name, fn):
    """The signature as Python, black-formatted: `async def build() -> Agent`."""
    parts, star = [], False
    for p in fn["params"]:
        if p["keyword_only"] and not star:
            parts.append("*")
            star = True
        part = p["name"] + (f": {p['type']}" if p["type"] else "")
        if p["default"] is not None:
            part += f" = {p['default']}"
        parts.append(part)
    returns = f" -> {fn['returns']}" if fn["returns"] else ""
    code = f"{'async ' if fn['is_async'] else ''}def {name}({', '.join(parts)}){returns}: ..."
    code = black.format_str(code, mode=BLACK).strip()
    code = re.sub(r":\s*\.\.\.$", "", code)
    return ("@staticmethod\n" if fn["is_static"] else "") + code


def class_of(name, obj, live):
    obj = target(obj)
    members = []
    for member_name, member in obj.members.items():
        # the dunders are how Python iterates and awaits; the page says so in words
        if member_name.startswith("_") and member_name != "__init__":
            continue
        member = target(member)
        doc = doc_of(member, live, member_name)
        if "property" in member.labels:
            t = text(member.returns if member.is_function else member.annotation)
            members.append({"name": member_name, "kind": "property", "doc": doc, "type": t, "signature": f"{member_name}: {t}"})
        elif member.is_function:
            kind = "constructor" if member_name == "__init__" else "method"
            fn = function(member, doc)
            if kind == "constructor":
                fn["signature"] = fn["signature"].replace("def __init__(", f"{name}(", 1).replace(" -> None", "")
            members.append({"name": member_name, "kind": kind, **fn})
    return {"kind": "class", "doc": doc_of(obj, live), "members": members}


def typed_dict(name, obj):
    """A TypedDict and its fields, a NotRequired[T] as an optional T."""
    fields = []
    for field_name, attr in target(obj).members.items():
        attr = target(attr)
        if not attr.is_attribute:
            continue
        t = text(attr.annotation) or "Any"
        optional = t.startswith("NotRequired[")
        fields.append({"name": field_name, "type": t[len("NotRequired[") : -1] if optional else t, "optional": optional})
    lines = [f"class {name}(TypedDict):"]
    lines += [f"    {f['name']}: {f'NotRequired[{f["type"]}]' if f['optional'] else f['type']}" for f in fields]
    return {"kind": "type", "doc": doc_of(obj), "definition": "\n".join(lines), "fields": fields}


def alias(name, obj):
    obj = target(obj)
    code = black.format_str(f"{name} = {text(obj.value)}", mode=BLACK).strip()
    doc = obj.docstring.value.strip() if obj.docstring else ""
    # types.py documents an alias in the comment above it (`# Anything JSON can hold.`)
    if not doc and obj.lineno and obj.filepath:
        lines = Path(obj.filepath).read_text(encoding="utf-8").splitlines()
        above, i = [], obj.lineno - 2
        while i >= 0 and lines[i].lstrip().startswith("#"):
            above.insert(0, lines[i].lstrip().lstrip("#").strip())
            i -= 1
        doc = " ".join(above)
    return {"kind": "type", "doc": doc, "definition": code, "fields": []}


api = {}

for name in pkg.exports or []:
    obj = target(pkg[name])
    live = getattr(runtime, name, None)
    if obj.is_module:
        continue
    if obj.is_class:
        api[name] = {**class_of(name, obj, live), "module": "ailoy"}
    elif obj.is_function:
        api[name] = {**function(obj, doc_of(obj, live)), "kind": "function", "module": "ailoy"}
    else:
        sys.exit(f"gen_py.py: ailoy exports {name}, neither a class, a function nor a module")

types = target(pkg["types"])
for name in types.exports or []:
    obj = target(types[name])
    api[name] = {**(typed_dict(name, obj) if obj.is_class else alias(name, obj)), "module": "ailoy.types"}

stubs = target(pkg["_ailoy"])
for name in STUB_ONLY:
    obj = target(stubs[name])
    entry = class_of(name, obj, None) if obj.is_class else alias(name, obj)
    api[name] = {**entry, "kind": "class" if obj.is_class else "type", "module": "ailoy._ailoy", "stub_only": True}

Path(out_file).write_text(json.dumps(api, indent=1), encoding="utf-8")
