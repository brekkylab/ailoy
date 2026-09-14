// Assistant text, rendered.
//
// Two overrides exist for reasons outside markdown itself:
//
// `img` — the webview's CSP allows `img-src 'self' data: asset: http://asset.localhost`
// (`src-tauri/tauri.conf.json`), so a model that writes an `https://` image URL would
// otherwise produce a silently broken image. It becomes a link instead.
//
// `pre` — react-markdown wraps a fence in `<pre><code>` and shiki brings its own `<pre>`,
// so keeping both would nest one padded `prose` box inside another. Unwrapping leaves
// exactly one code block; inline code never reaches `pre`, so nothing else is affected.
//
// Anchors carry `target="_blank"`, which the webview turns into a no-op rather than a
// navigation. That is deliberate: a plain in-webview navigation would replace the running
// app with the remote page and leave the user no way back. Opening a link in the real
// browser needs the opener plugin, which v1 does not ship.

import ReactMarkdown from "react-markdown";
import ShikiHighlighter from "react-shiki";
import remarkGfm from "remark-gfm";

/** A fence with no info string still gets a code block, just without a grammar. */
const PLAIN = "text";

/**
 * Fence or backticks? react-markdown renders both through `code` and (since v9) marks
 * neither. An info string settles it outright; otherwise this is react-shiki's own
 * `isInlineCode` test — hast keeps a fence's trailing newline, backticks in a sentence
 * have none. Reimplemented rather than imported because react-shiki carries its own copy
 * of `@types/hast` and the two `Element` types do not unify.
 */
function isFence(className: string | undefined, code: string): boolean {
  return /^language-/.test(className ?? "") || code.includes("\n");
}

export function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ children }) => <>{children}</>,
          code({ className, children }) {
            const code = String(children);
            if (!isFence(className, code)) {
              return <code className="rounded bg-muted px-1 py-0.5">{children}</code>;
            }
            const lang = /language-([\w-]+)/.exec(className ?? "")?.[1] ?? PLAIN;
            return (
              <ShikiHighlighter language={lang} theme="github-dark" showLanguage={false} className="text-xs">
                {code.replace(/\n$/, "")}
              </ShikiHighlighter>
            );
          },
          img({ src, alt }) {
            const href = typeof src === "string" ? src : "";
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {alt || href}
              </a>
            );
          },
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
