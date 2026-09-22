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
// A link the app handles itself — a Notion page, say — is a button instead: there is no
// document to navigate to. Its href survives `urlTransform` only because the caller's
// `resolveLink` claims it; see there.
//
// Every other anchor carries `target="_blank"`, which the webview turns into a no-op rather than a
// navigation. That is deliberate: a plain in-webview navigation would replace the running
// app with the remote page and leave the user no way back. Opening a link in the real
// browser needs the opener plugin, which v1 does not ship.
//
// The highlighter is built in `lib/highlighter.ts` and not taken from `react-shiki`'s
// default entry, whose WebAssembly engine the packaged app's CSP refused before that
// policy gained `'wasm-unsafe-eval'`, and which costs 11 MB of grammars either way — see
// the header there. It resolves asynchronously, so a fence renders as a plain block for
// the frame or two before the grammars land.

import { memo, useEffect, useMemo, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import ShikiHighlighter from "react-shiki/core";
import remarkGfm from "remark-gfm";

import { splitMarkdown } from "@/lib/markdownChunks";
import { useHighlighter } from "@/lib/useHighlighter";

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

/**
 * One piece of a document, rendered once and then left alone.
 *
 * Memoized, and that is the whole reason it exists: a long document grows a piece at a time,
 * and without this every step would re-parse and re-render everything already on screen —
 * quadratic in the length of the file, which for the 817 KiB one is worse than the freeze it
 * replaced. Each piece's props are stable (`resolveLink` is the caller's, and callers hold it
 * across renders), so React skips the ones that have not changed.
 */
const Piece = memo(function Piece({
  text,
  resolveLink,
  highlighter,
}: {
  text: string;
  resolveLink?: (href: string) => (() => void) | null;
  highlighter: ReturnType<typeof useHighlighter>;
}) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      // react-markdown drops every scheme but the handful it considers safe, which is
      // right for a model's output and would silently empty the href of a link this app
      // handles itself. One a `resolveLink` claims is not going anywhere, so it is kept;
      // everything else still faces the default.
      urlTransform={(url) => (resolveLink?.(url) ? url : defaultUrlTransform(url))}
      components={{
        pre: ({ children }) => <>{children}</>,
        code({ className, children }) {
          const code = String(children);
          if (!isFence(className, code)) {
            // `before:/after:content-none` undoes @tailwindcss/typography, which renders
            // a literal backtick either side of `prose code` — a convention for printed
            // prose, and pure noise next to a background-tinted chip.
            return (
              <code className="rounded bg-muted px-1 py-0.5 before:content-none after:content-none">{children}</code>
            );
          }
          const lang = /language-([\w-]+)/.exec(className ?? "")?.[1] ?? PLAIN;
          const body = code.replace(/\n$/, "");
          if (!highlighter) {
            // Shaped like shiki's own output so the block does not jump when the
            // grammars arrive a frame later.
            return (
              <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                <code>{body}</code>
              </pre>
            );
          }
          return (
            <ShikiHighlighter
              language={lang}
              theme="github-dark"
              highlighter={highlighter}
              showLanguage={false}
              className="text-xs"
            >
              {body}
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
          const go = href ? resolveLink?.(href) : null;
          if (go) {
            // A button, not an anchor: there is no document to navigate to, and an
            // anchor with no destination is a link a reader cannot open in a new tab,
            // copy, or hover to see where it goes.
            return (
              <button
                type="button"
                className="cursor-pointer text-inherit underline underline-offset-2"
                onClick={go}
              >
                {children}
              </button>
            );
          }
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
  );
});

/**
 * Roughly how tall a piece is, per byte of markdown. Only a hint, and only until the browser
 * has laid the piece out once — see the `content-visibility` below.
 */
const PX_PER_BYTE = 0.45;

/**
 * How much of a long document is on screen, growing until all of it is.
 *
 * A markdown file in a workspace is not a chat message: one of the reports this is used to
 * read is 817 KiB, which renders in about two seconds of blocked main thread and 130,000
 * nodes — measured here, with this renderer. That is the window freezing, and it happens
 * again every time the file is opened, because what is cached is the text and not the render.
 *
 * So a long one arrives in pieces and each frame renders one more. The first piece is on
 * screen immediately, the rest fill in behind it, and nothing blocks for longer than a piece
 * takes. Short documents — every assistant message — are one piece and go through unchanged.
 */
function useProgressive(text: string): string[] {
  const pieces = useMemo(() => splitMarkdown(text), [text]);
  // How many are on screen is state; *which document* they belong to is not. Another file —
  // or another turn of a message still being written — starts again at one, and that is a
  // fact about this render rather than something to catch up to in an effect.
  const [counted, setCounted] = useState({ pieces, shown: 1 });
  const shown = counted.pieces === pieces ? counted.shown : 1;
  useEffect(() => {
    if (shown >= pieces.length) return;
    // A turn of the loop between pieces, which is what lets the window answer between them.
    const timer = setTimeout(() => setCounted({ pieces, shown: shown + 1 }), 0);
    return () => clearTimeout(timer);
  }, [pieces, shown]);
  return useMemo(() => pieces.slice(0, shown), [pieces, shown]);
}

export function Markdown({
  text,
  resolveLink,
}: {
  text: string;
  /**
   * Turns a link that points inside the app into what to do about it, or answers `null`
   * to leave it as an ordinary outbound link.
   *
   * Returning the handler rather than taking a click means the anchor knows at render time
   * which of the two it is, and can be drawn as the thing it actually does.
   */
  resolveLink?: (href: string) => (() => void) | null;
}) {
  const highlighter = useHighlighter();
  const pieces = useProgressive(text);
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words">
      {pieces.map((piece, i) => (
        <div
          key={i}
          // Mounted, and skipped while it is off screen: the browser does not lay out or
          // paint what is not in view, and remembers the size of what it has. Rendering is
          // still done once and kept — a piece that unmounted and came back would parse and
          // highlight itself again, which is a hitch in the middle of a scroll.
          style={{ contentVisibility: "auto", containIntrinsicSize: `auto ${PX_PER_BYTE * piece.length}px` }}
        >
          <Piece text={piece} resolveLink={resolveLink} highlighter={highlighter} />
        </div>
      ))}
    </div>
  );
}
