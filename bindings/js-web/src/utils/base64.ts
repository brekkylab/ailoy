export function uint8ArrayToBase64(buffer: Uint8Array<ArrayBufferLike>) {
  const CHUNK_SIZE = 8192; // Process 8KB chunks to avoid call stack overflow
  let result = "";

  for (let i = 0; i < buffer.length; i += CHUNK_SIZE) {
    const chunk = buffer.subarray(i, i + CHUNK_SIZE);
    result += String.fromCharCode(...chunk);
  }

  return btoa(result);
}
