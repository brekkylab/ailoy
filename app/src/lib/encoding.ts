// `res.text()` decodes only UTF-8, and CP949 workspace files would read as silent mojibake,
// so the app decodes itself.

/// A decoded file, and what it turned out to be written in.
export interface Decoded {
  text: string;
  /// The encoding's display name, shown to the reader.
  encoding: 'UTF-8' | 'UTF-16' | 'CP949';
}

/// `buffer` as text, with the encoding inferred from the bytes.
///
/// A byte order mark decides outright. Otherwise the file is decoded as strict UTF-8, and
/// only on failure as CP949. ASCII is identical in both, so English is never "detected" as
/// Korean, and CP949 Hangul is invalid UTF-8, so it fails on the first syllable rather
/// than on a heuristic.
export function decodeText(buffer: ArrayBuffer): Decoded {
  const bytes = new Uint8Array(buffer);

  if (bytes.length >= 2) {
    // UTF-16 BOM, which Windows editors write and no UTF-8 decoder recovers from.
    // `TextDecoder` drops the mark itself.
    if (bytes[0] === 0xff && bytes[1] === 0xfe) return utf16(buffer, 'utf-16le');
    if (bytes[0] === 0xfe && bytes[1] === 0xff) return utf16(buffer, 'utf-16be');
  }

  const utf8 = strict(buffer, 'utf-8');
  if (utf8 !== null) return { text: utf8, encoding: 'UTF-8' };

  // Browsers map the `euc-kr` label to the Windows-949 index, i.e. CP949: EUC-KR plus the
  // extra syllables the files use.
  const cp949 = strict(buffer, 'euc-kr');
  if (cp949 !== null) return { text: cp949, encoding: 'CP949' };

  // Neither decodes, so likely a binary with a text extension: UTF-8 with replacement
  // characters, which stay visible as such.
  return { text: new TextDecoder('utf-8').decode(buffer), encoding: 'UTF-8' };
}

/// `buffer` decoded as `label`, or null when the bytes are not valid in it.
function strict(buffer: ArrayBuffer, label: string): string | null {
  try {
    return new TextDecoder(label, { fatal: true }).decode(buffer);
  } catch {
    // Invalid bytes, or a label the environment lacks; either way not this encoding.
    return null;
  }
}

function utf16(buffer: ArrayBuffer, label: string): Decoded {
  return { text: new TextDecoder(label).decode(buffer), encoding: 'UTF-16' };
}
