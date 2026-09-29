<script lang="ts">
  import type { TextProps } from './registry';

  let { entry, text, base }: TextProps = $props();

  /// The file's own directory, so relative `<img>`s and stylesheets resolve against its tree;
  /// a `srcdoc` frame would otherwise resolve them against the app's address.
  let href = $derived(base);
  let doc = $derived(withHead(text, href));

  /// Prepended to the document's head.
  ///
  /// A second lock behind the sandbox, which is what actually stops scripts: if either is
  /// got wrong, the other still blocks them.
  function headOf(href: string): string {
    return (
      `<base href="${href}">` +
      '<meta http-equiv="Content-Security-Policy" ' +
      `content="script-src 'none'; object-src 'none'; frame-src 'none'">`
    );
  }

  /// `html` with that head inserted after `<head>`, or at the front if there is none (where
  /// a browser reads it either way).
  function withHead(html: string, href: string): string {
    const head = headOf(href);
    const open = html.match(/<head[^>]*>/i) ?? html.match(/<html[^>]*>/i);
    if (!open || open.index === undefined) return head + html;
    const at = open.index + open[0].length;
    return html.slice(0, at) + head + html.slice(at);
  }
</script>

<div class="frame">
  <!--
    `sandbox` with no tokens is the point of this viewer: no scripts, forms, or navigation
    of its tab, and an opaque origin, so the document cannot reach the app. `srcdoc`, not
    `src`, because the server sends these files as attachments (see `is_active_content`).
  -->
  <iframe
    title={entry.name}
    srcdoc={doc}
    sandbox=""
    referrerpolicy="no-referrer"
    loading="lazy"
  ></iframe>
</div>
<p class="note">Preview only — scripts and forms in this document are not run.</p>

<style>
  .frame {
    height: calc(100% - 26px);
    /* Documents are written for a white page, whatever theme the app is in. */
    background: #fff;
  }
  iframe {
    display: block;
    width: 100%;
    height: 100%;
    border: none;
  }
  .note {
    display: flex;
    align-items: center;
    height: 26px;
    margin: 0;
    padding: 0 14px;
    border-top: 1px solid var(--border);
    background: var(--bg-panel);
    font-size: var(--fs-sm);
    color: var(--text-faint);
  }
</style>
