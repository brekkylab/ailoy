// Which viewer opens which file.
//
// One table keyed by extension, so adding a type is adding a line here and a branch in
// one component rather than a condition spread through the file panel. A name whose
// extension is absent falls back to plain text, which is what every file did before.
//
// Everything here reads *text*, because that is all the engine hands back: `fs_read`
// decodes what it read and returns `null` for bytes it could not, so a file the app
// cannot show as characters it cannot show at all. Images, PDFs and Office documents
// therefore have no entry — they need the engine to serve bytes first, and a viewer
// registered for them today would open on an empty pane.

/** What the content pane does with the text. */
export type ViewerKind = "markdown" | "table" | "code" | "text";

export interface Viewer {
  /** What the pane calls the format, beside the path. */
  label: string;
  kind: ViewerKind;
  /** For `code`: the grammar to ask shiki for. Must be one `lib/highlighter` registers. */
  lang?: string;
  /** For `table`: what separates the fields. */
  delimiter?: string;
}

/** The lowercased extension of a file name, or `""` when it has none. */
export function extOf(name: string): string {
  const cut = name.lastIndexOf(".");
  // A leading dot is the whole name of a dotfile, not an extension: `.gitignore` has none.
  if (cut <= 0) return "";
  return name.slice(cut + 1).toLowerCase();
}

const MARKDOWN: Viewer = { label: "Markdown", kind: "markdown" };
const TEXT: Viewer = { label: "Text", kind: "text" };
/** The same viewer under the name the file gave itself. */
const LOG: Viewer = { label: "Log", kind: "text" };
const CSV: Viewer = { label: "CSV", kind: "table", delimiter: "," };
const TSV: Viewer = { label: "TSV", kind: "table", delimiter: "\t" };
const code = (label: string, lang: string): Viewer => ({ label, kind: "code", lang });

/**
 * HTML is shown as source, not rendered.
 *
 * These files come off a disk or a bucket and some of them were written by the agent. The
 * webview has the app's own origin and its bridge to the engine on it, so a document that
 * ran here would be running inside the app rather than beside it. Reading the markup is
 * what a file browser is for; a live preview is a sandbox, and a sandbox is a decision to
 * make on purpose rather than by registering an extension.
 */
const HTML = code("HTML", "html");

const BY_EXT: Record<string, Viewer | undefined> = {
  md: MARKDOWN,
  markdown: MARKDOWN,
  mdown: MARKDOWN,
  mkd: MARKDOWN,
  txt: TEXT,
  text: TEXT,
  log: LOG,
  csv: CSV,
  tsv: TSV,
  html: HTML,
  htm: HTML,
  json: code("JSON", "json"),
  jsonl: code("JSON Lines", "json"),
  ts: code("TypeScript", "typescript"),
  tsx: code("TypeScript", "tsx"),
  js: code("JavaScript", "javascript"),
  jsx: code("JavaScript", "tsx"),
  mjs: code("JavaScript", "javascript"),
  cjs: code("JavaScript", "javascript"),
  py: code("Python", "python"),
  rs: code("Rust", "rust"),
  go: code("Go", "go"),
  sql: code("SQL", "sql"),
  css: code("CSS", "css"),
  toml: code("TOML", "toml"),
  yaml: code("YAML", "yaml"),
  yml: code("YAML", "yaml"),
  sh: code("Shell", "bash"),
  bash: code("Shell", "bash"),
  zsh: code("Shell", "bash"),
  diff: code("Diff", "diff"),
  patch: code("Diff", "diff"),
  dockerfile: code("Dockerfile", "docker"),
};

/**
 * The viewer for a file name. Never null: an unknown extension reads as text, which is
 * both what the pane did before and the only honest thing to do with characters whose
 * shape nothing here recognises.
 */
export function viewerFor(name: string): Viewer {
  const base = name.slice(name.lastIndexOf("/") + 1);
  // `Dockerfile` carries its type in the whole name rather than after a dot.
  if (base.toLowerCase() === "dockerfile") return BY_EXT.dockerfile!;
  return BY_EXT[extOf(base)] ?? TEXT;
}
