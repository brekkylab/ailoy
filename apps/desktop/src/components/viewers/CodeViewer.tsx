// A source file, highlighted.
//
// The same highlighter the thread's fences use — built by hand in `lib/highlighter` for
// the reason recorded there, and shared, so opening a file costs no second set of
// grammars. A file whose language is not registered still renders; shiki falls back to
// plain text, which is what the pane would have shown anyway.

import ShikiHighlighter from "react-shiki/core";

import { useHighlighter } from "@/lib/useHighlighter";

export function CodeViewer({ text, lang }: { text: string; lang: string }) {
  const highlighter = useHighlighter();
  if (!highlighter) {
    // Shaped like shiki's own output, so the pane does not jump when the grammars land a
    // frame later.
    return (
      <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
        <code>{text}</code>
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
      {text}
    </ShikiHighlighter>
  );
}
