// The address of a workspace file, for the viewers that open a format.
//
// The engine serves these over the app's own URI scheme — see `src-tauri/src/wsfile.rs`
// for why it is a scheme and not a command. This is the one place that spells the URL, so
// the encoding and the handler's decoding are a pair rather than two guesses.

/** Must match `SCHEME` in `src-tauri/src/wsfile.rs`, and the CSP in `tauri.conf.json`. */
const SCHEME = "wsfile";

/**
 * Where to fetch a workspace path from.
 *
 * Each segment is encoded on its own, so the separators survive while everything else — a
 * space, a parenthesis, a Hangul syllable — is escaped. `encodeURIComponent` on the whole
 * path would escape the slashes too and the handler would be handed one long name.
 *
 * `localhost` is the authority because a custom scheme still needs one for the URL to
 * parse; the handler ignores it.
 */
export function wsfileUrl(path: string): string {
  const encoded = path.split("/").filter(Boolean).map(encodeURIComponent).join("/");
  return `${SCHEME}://localhost/${encoded}`;
}
