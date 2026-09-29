<script lang="ts">
  import type { TextProps } from './registry';

  let { entry, text, base }: TextProps = $props();

  /// The file's own directory, so a relative `<img>` or stylesheet in the document
  /// resolves against the tree it was read from. A `srcdoc` frame otherwise resolves
  /// relative URLs against the app's own address, where none of them exist.
  let href = $derived(base);
  let doc = $derived(withHead(text, href));

  /// What is put at the top of the document's head.
  ///
  /// The sandbox below is what actually stops scripts; this is the second lock. A
  /// document that reaches the DOM through two independent "no scripts" is a document
  /// that still runs none of them when one of them is got wrong.
  function headOf(href: string): string {
    return (
      `<base href="${href}">` +
      '<meta http-equiv="Content-Security-Policy" ' +
      `content="script-src 'none'; object-src 'none'; frame-src 'none'">`
    );
  }

  /// `html` with that head inserted — after `<head>` when the document has one, and at
  /// the front when it does not, which is where a browser will read it either way.
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
    `sandbox` with no tokens is the whole point of this viewer: no scripts, no forms,
    no navigation of the tab it sits in, and an opaque origin, so the document cannot
    reach the app it is being previewed inside. `srcdoc` rather than `src` because the
    server sends these files as attachments — see `is_active_content` on the server.
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
