// Reads the Node.js API of the installed @brekkylab/ailoy into the JSON model render.mjs writes the
// pages from, the same shape gen_py.py writes for Python. typedoc reads the declarations; nothing
// is rendered by it.
//
// Usage (regen.mjs runs it): node node_api.mjs <out.json>
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Application, Comment, ReflectionKind } from 'typedoc';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// napi's aliases of each class and its loader detail, not API
const HIDDEN = /^(Js[A-Z]\w*|__napiBindingTarget)$/;

const app = await Application.bootstrap({
  entryPoints: [path.join(HERE, 'node_modules/@brekkylab/ailoy/index.d.ts').replaceAll('\\', '/')],
  tsconfig: path.join(HERE, 'tsconfig.json'),
  skipErrorChecking: true,
  excludeExternals: true,
  externalPattern: ['**/@brekkylab/virtx/**', '**/@types/**'],
  readme: 'none',
  sort: ['source-order'],
  logLevel: 'Warn',
});
const project = await app.convert();
if (!project) throw new Error('node_api: typedoc could not read @brekkylab/ailoy');

const doc = (c) => (c ? Comment.combineDisplayParts(c.summary).trim() : '');
const type = (t) => (t ? t.toString() : 'unknown');

// `options?: { history?: Message[]; ... }` comes back as `options` and one row per key
function params(sig) {
  const out = [];
  for (const p of sig.parameters ?? []) {
    out.push({ name: p.name, type: type(p.type), optional: p.flags.isOptional, default: p.defaultValue ?? null, doc: doc(p.comment) });
    const keys = p.type?.type === 'reflection' ? p.type.declaration.children : null;
    for (const k of keys ?? []) {
      out.push({ name: `${p.name}.${k.name}`, type: type(k.type), optional: k.flags.isOptional, default: null, doc: doc(k.comment) });
    }
  }
  return out;
}

const wrap = (head, args, tail) => {
  const line = `${head}(${args.join(', ')})${tail}`;
  return line.length <= 80 ? line : `${head}(\n${args.map((a) => `  ${a},`).join('\n')}\n)${tail}`;
};

function signature(name, sig, prefix = '') {
  const args = (sig.parameters ?? []).map((p) => `${p.name}${p.flags.isOptional ? '?' : ''}: ${type(p.type)}`);
  return wrap(`${prefix}${name}`, args, sig.type ? `: ${type(sig.type)}` : '');
}

function fn(r, prefix = '') {
  const sig = r.signatures[0];
  return {
    doc: doc(sig.comment) || doc(r.comment),
    params: params(sig),
    returns: type(sig.type),
    signature: signature(r.name, sig, prefix),
  };
}

function member(r) {
  if (r.kind === ReflectionKind.Constructor) {
    const sig = r.signatures[0];
    return { name: 'constructor', kind: 'constructor', doc: doc(sig.comment), params: params(sig), returns: null, signature: signature('new ' + r.parent.name, sig).replace(/: \w+$/, '') };
  }
  if (r.kind === ReflectionKind.Method) {
    const static_ = r.flags.isStatic;
    return { name: r.name, kind: 'method', is_static: static_, ...fn(r, static_ ? 'static ' : '') };
  }
  if (r.kind === ReflectionKind.Accessor) {
    const get = r.getSignature;
    return { name: r.name, kind: 'property', doc: doc(get.comment), type: type(get.type), signature: `get ${r.name}(): ${type(get.type)}` };
  }
  if (r.kind === ReflectionKind.Property) {
    return { name: r.name, kind: 'property', doc: doc(r.comment), type: type(r.type), signature: `${r.name}${r.flags.isOptional ? '?' : ''}: ${type(r.type)}` };
  }
  return null;
}

// napi gives every class a bare constructor; one with no parameters and no doc is not API
const real = (m) => m && !(m.kind === 'constructor' && !m.params.length && !m.doc);

function iface(r) {
  const fields = [];
  const members = [];
  for (const c of r.children ?? []) {
    if (c.kind === ReflectionKind.Method) {
      const m = { name: c.name, kind: 'method', optional: c.flags.isOptional, ...fn(c) };
      if (m.optional) m.signature = m.signature.replace(`${c.name}(`, `${c.name}?(`);
      members.push(m);
    }
    else fields.push({ name: c.name, type: type(c.type), optional: c.flags.isOptional, doc: doc(c.comment) });
  }
  // `[field: string]: Value` is an index signature, not a child
  const index = r.indexSignatures?.[0];
  const indexLine = index ? `  [${index.parameters[0].name}: ${type(index.parameters[0].type)}]: ${type(index.type)};` : null;
  const lines = [
    ...fields.map((f) => `  ${f.name}${f.optional ? '?' : ''}: ${f.type};`),
    ...(indexLine ? [indexLine] : []),
    ...members.map((m) => `  ${m.signature.replace(/\n/g, '\n  ')};`),
  ];
  return { kind: members.length ? 'class' : 'type', doc: doc(r.comment), definition: `interface ${r.name} {\n${lines.join('\n')}\n}`, fields, members };
}

const api = {};
for (const r of project.children ?? []) {
  if (HIDDEN.test(r.name)) continue;
  const module = '@brekkylab/ailoy';
  if (r.kind === ReflectionKind.Class) {
    api[r.name] = { kind: 'class', module, doc: doc(r.comment), members: (r.children ?? []).map(member).filter(real) };
  } else if (r.kind === ReflectionKind.Function) {
    api[r.name] = { kind: 'function', module, ...fn(r, 'function ') };
  } else if (r.kind === ReflectionKind.Interface) {
    api[r.name] = { module, type_only: true, ...iface(r) };
  } else if (r.kind === ReflectionKind.TypeAlias) {
    api[r.name] = { kind: 'type', module, type_only: true, doc: doc(r.comment), definition: `type ${r.name} = ${type(r.type)};`, fields: [] };
  } else {
    throw new Error(`node_api: @brekkylab/ailoy exports ${r.name}, a ${ReflectionKind[r.kind]} with no place in the reference`);
  }
}

fs.writeFileSync(process.argv[2], JSON.stringify(api, null, 1));
