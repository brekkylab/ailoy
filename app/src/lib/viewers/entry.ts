// A file as the viewers need to know it — what the registry picks a viewer by and what
// the viewers name, size and resolve against.

export interface Entry {
  name: string;
  /** Its path in the tree it was opened from, one name per segment. */
  segments: string[];
  type: "file" | "folder";
  size: number | null;
  /** Milliseconds since the epoch, when known. */
  modified: number | null;
}
