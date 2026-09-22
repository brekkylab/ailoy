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

import { memo, useEffect, useMemo, useRef, useState } from "react";
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
 * Roughly how tall a piece will be, per byte of markdown.
 *
 * Only ever a guess, and only until the piece has been on screen once: what it buys is a
 * scrollbar that is about the right length before anything below the fold has been rendered.
 * Measured on the document this was written for — prose with tables and fences, at this
 * app's prose width — and wrong for a file of nothing but one-word lines, where it will be
 * short and the scrollbar will settle as the reader goes.
 */
const PX_PER_BYTE = 0.45;

/**
 * One piece, rendered while it is near the viewport and a gap of its own height otherwise.
 *
 * A markdown file in a workspace is not a chat message: one of the reports this is used to
 * read is 817 KiB, which is 130,000 nodes — measured here, with this renderer. Rendering all
 * of it blocked the window for two seconds, and *keeping* all of it makes every scroll and
 * every teardown pay for a document the reader is looking at one screen of.
 *
 * So a piece mounts when it comes near and unmounts when it leaves, and what stays behind is
 * a box of the height it had. The height is remembered from when it was last on screen, which
 * is why a reader scrolling back up lands where they left rather than somewhere near it.
 *
 * `IntersectionObserver` rather than the scroll container's own geometry, because this
 * component does not own the scroller: the thread scrolls, the file pane scrolls, and a
 * Notion page scrolls inside something else again.
 */
function Window({
  text,
  resolveLink,
  highlighter,
}: {
  text: string;
  resolveLink?: (href: string) => (() => void) | null;
  highlighter: ReturnType<typeof useHighlighter>;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  // What it measured last time it was on screen, which is what the gap it leaves behind is
  // made of. State rather than a ref because the render reads it, and a height that changed
  // without a render would be a gap of the wrong size until something else caused one.
  const [height, setHeight] = useState<number | null>(null);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    // A screenful either side, so a piece is rendered before the reader reaches it and kept
    // for a moment after they pass it — scrolling back a little does not re-render anything.
    const io = new IntersectionObserver(([entry]) => setNear(entry.isIntersecting), {
      rootMargin: "1200px 0px",
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // Measured while it is up, so the gap it leaves behind is the size it was. A resize
  // observer rather than a read per render: the height settles as fences highlight and
  // images load, and this catches that without anything asking.
  useEffect(() => {
    const el = box.current;
    if (!el || !near) return;
    // Settles quickly — a fence highlighting, an image landing — and then stops, so this is
    // a handful of renders per piece and none of them change what is on screen while it is up.
    const ro = new ResizeObserver(() => {
      const measured = el.offsetHeight;
      if (measured > 0) setHeight((was) => (was === measured ? was : measured));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [near]);

  return (
    <div
      ref={box}
      style={near ? undefined : { height: height ?? Math.max(160, text.length * PX_PER_BYTE) }}
    >
      {near && <Piece text={text} resolveLink={resolveLink} highlighter={highlighter} />}
    </div>
  );
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
  const pieces = useMemo(() => splitMarkdown(text), [text]);
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words">
      {/* One piece is the overwhelming case — every assistant message, every note — and it
          goes straight through: a document that fits on a few screens has nothing to gain
          from a box that measures itself. */}
      {pieces.length === 1 ? (
        <Piece text={pieces[0]} resolveLink={resolveLink} highlighter={highlighter} />
      ) : (
        pieces.map((piece, i) => (
          <Window key={i} text={piece} resolveLink={resolveLink} highlighter={highlighter} />
        ))
      )}
    </div>
  );
}
