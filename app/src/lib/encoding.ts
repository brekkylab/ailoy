// What a file's bytes say, as characters.
//
// `fetch`'s own `res.text()` decodes as UTF-8 and nothing else, and the workspace has
// files that are not: `erp/` is a Korean ERP's nightly output, written in CP949 the way
// every Windows-hosted Korean system writes it. Read as UTF-8 those come out as
// mojibake — not an error anywhere, just a document nobody can read. So the app decodes
// rather than letting `fetch` do it, which is the whole of this module.

/// A decoded file, and what it turned out to be written in.
export interface Decoded {
  text: string;
  /// What to call the encoding on screen. The app says which one it settled on rather
  /// than quietly guessing: a document that is being decoded by inference is one whose
  /// reader should be told so.
  encoding: 'UTF-8' | 'UTF-16' | 'CP949';
}

/// `buffer` as text, with the encoding worked out from the bytes.
///
/// A byte order mark settles it outright. Failing that, the file is decoded as UTF-8
/// *strictly* — any sequence UTF-8 does not allow makes the whole attempt fail — and
/// only a file that fails it is tried as CP949. That order is what keeps the guess
/// honest in both directions: ASCII is the same bytes in both, so a plain English file
/// is never "detected" as Korean, and Korean text in CP949 is not valid UTF-8, so it
/// falls through on the first Hangul syllable rather than on a heuristic.
export function decodeText(buffer: ArrayBuffer): Decoded {
  const bytes = new Uint8Array(buffer);

  if (bytes.length >= 2) {
    // A UTF-16 mark, which Windows editors write and which no UTF-8 decoder recovers
    // from. `TextDecoder` drops the mark itself.
    if (bytes[0] === 0xff && bytes[1] === 0xfe) return utf16(buffer, 'utf-16le');
    if (bytes[0] === 0xfe && bytes[1] === 0xff) return utf16(buffer, 'utf-16be');
  }

  const utf8 = strict(buffer, 'utf-8');
  if (utf8 !== null) return { text: utf8, encoding: 'UTF-8' };

  // `euc-kr` is the label; what a browser maps it to is the Windows-949 index, which is
  // CP949 — EUC-KR plus the extra syllables, which is what the files actually use.
  const cp949 = strict(buffer, 'euc-kr');
  if (cp949 !== null) return { text: cp949, encoding: 'CP949' };

  // Neither, so the file is not text in either of them — a binary served with a text
  // extension, most likely. UTF-8 with the undecodable bytes replaced is what is left,
  // and the substitutions are visible for what they are.
  return { text: new TextDecoder('utf-8').decode(buffer), encoding: 'UTF-8' };
}

/// `buffer` decoded as `label`, or null when the bytes are not valid in it.
function strict(buffer: ArrayBuffer, label: string): string | null {
  try {
    return new TextDecoder(label, { fatal: true }).decode(buffer);
  } catch {
    // Either the bytes are invalid, or the environment does not know the label — a
    // decoder that is not there is one this file is not in.
    return null;
  }
}

function utf16(buffer: ArrayBuffer, label: string): Decoded {
  return { text: new TextDecoder(label).decode(buffer), encoding: 'UTF-16' };
}
