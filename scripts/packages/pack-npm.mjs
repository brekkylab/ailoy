// Pack the Node packages as `napi pre-publish` would publish them: every platform
// package, and the root with each of them as an optional dependency at its version.
//
//   node ../../scripts/packages/pack-npm.mjs <out-dir> [--allow-missing]   (from bindings/node, after `napi artifacts`)
//
// The root's `binding.js`/`binding.d.ts`, which `napi build` generates, have to be in
// bindings/node already: the workflow puts one build's there.
//
// `--allow-missing` packs only the platform packages that have their binary, and still
// names every one in the root -- for a local run on one machine. A release packs all.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const out = path.resolve(process.argv[2])
const allowMissing = process.argv.includes('--allow-missing')
fs.mkdirSync(out, { recursive: true })
const root = JSON.parse(fs.readFileSync('package.json', 'utf8'))
const optional = {}
let packed = 0
for (const dir of fs.readdirSync('npm').sort()) {
  const pkg = JSON.parse(fs.readFileSync(path.join('npm', dir, 'package.json'), 'utf8'))
  if (pkg.version !== root.version) throw new Error(`npm/${dir} is ${pkg.version}, the root ${root.version}`)
  optional[pkg.name] = pkg.version
  const missing = pkg.files.filter((file) => !fs.existsSync(path.join('npm', dir, file)))
  if (missing.length && allowMissing) {
    console.log(`skipping ${pkg.name}: no ${missing.join(', ')}`)
    continue
  }
  if (missing.length) throw new Error(`npm/${dir} has no ${missing.join(', ')}`)
  execFileSync('npm', ['pack', '--pack-destination', out], { cwd: path.join('npm', dir), stdio: 'inherit', shell: process.platform === 'win32' })
  packed++
}
const missing = root.files.filter((file) => !fs.existsSync(file))
if (missing.length) throw new Error(`the root package has no ${missing.join(', ')}: run \`napi build\` first`)
const staged = fs.mkdtempSync(path.join(out, 'root-'))
for (const file of root.files) fs.copyFileSync(file, path.join(staged, file))
// The README and the license are the repository's, two levels up; npm packs both whatever
// `files` says.
fs.copyFileSync(path.join('..', '..', 'README.md'), path.join(staged, 'README.md'))
fs.copyFileSync(path.join('..', '..', 'LICENSE.md'), path.join(staged, 'LICENSE.md'))
fs.writeFileSync(path.join(staged, 'package.json'), JSON.stringify({ ...root, optionalDependencies: optional }, null, 2) + '\n')
execFileSync('npm', ['pack', '--pack-destination', out], { cwd: staged, stdio: 'inherit', shell: process.platform === 'win32' })
fs.rmSync(staged, { recursive: true })
console.log(`packed ${packed + 1} packages into ${out}`)
