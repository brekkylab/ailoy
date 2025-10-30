type DataType = "text" | "json" | "arraybuffer";

/**
 * Joins a base path and a file name into a single normalized path.
 * Ensures that duplicate slashes are replaced with a single slash.
 * @param paths The path segments to join.
 * @returns The combined and normalized file path.
 */
export function joinPath(...paths: string[]): string {
  return paths.join("/").replace(/\/+/g, "/");
}

/**
 * Reads a file from OPFS.
 * @param path The file path (e.g., "/folder/file.txt")
 * @param type The type of data to retrieve: 'text' | 'json' | 'arraybuffer'
 * @returns The file contents (as text, object, or ArrayBuffer depending on the type)
 */
export async function readOPFSFile(
  path: string,
  type: DataType = "arraybuffer"
): Promise<string | object | ArrayBuffer> {
  const root = await navigator.storage.getDirectory();
  const parts = path.split("/").filter(Boolean);

  let dirHandle = root;
  for (let i = 0; i < parts.length - 1; i++)
    dirHandle = await dirHandle.getDirectoryHandle(parts[i], { create: false });

  const fileName = parts[parts.length - 1];
  const fileHandle = await dirHandle.getFileHandle(fileName);
  const file = await fileHandle.getFile();

  switch (type) {
    case "text":
      return await file.text();
    case "json":
      return JSON.parse(await file.text());
    case "arraybuffer":
      return await file.arrayBuffer();
    default:
      throw new Error("Unsupported output type.");
  }
}

/**
 * Saves a file to OPFS.
 * @param path The full file path to save to (e.g., "/folder/file.txt")
 * @param data The data to be saved (string, object, or ArrayBuffer)
 */
export async function writeOPFSFile(
  path: string,
  data: string | object | ArrayBuffer
): Promise<DataType> {
  let dirHandle = await navigator.storage.getDirectory();
  const parts = path.split("/").filter(Boolean);

  for (let i = 0; i < parts.length - 1; i++)
    dirHandle = await dirHandle.getDirectoryHandle(parts[i], { create: true });

  const fileName = parts[parts.length - 1];
  const fileHandle = await dirHandle.getFileHandle(fileName, { create: true });
  const writable = await fileHandle.createWritable();

  let ret: DataType = "arraybuffer";
  if (typeof data === "string") {
    await writable.write(data as string);
    ret = "text";
  } else if (data instanceof ArrayBuffer) {
    await writable.write(data as ArrayBuffer);
    ret = "arraybuffer";
  } else if (typeof data === "object" && data !== null) {
    await writable.write(JSON.stringify(data));
    ret = "json";
  } else {
    throw new Error(`Not supported type: ${typeof data}`);
  }

  await writable.close();
  return ret;
}

/**
 * Removes a file from OPFS at the given path.
 * @param path The full file path to remove (e.g., "folder/file.txt")
 */
export async function removeOPFSFile(path: string): Promise<void> {
  const root = await navigator.storage.getDirectory();

  const parts = path.split("/").filter(Boolean);
  if (parts.length === 0) throw new Error("Invalid path");

  const fileName = parts.pop()!;
  let currentDir = root;
  for (const dir of parts) {
    currentDir = await currentDir.getDirectoryHandle(dir, { create: false });
  }

  await currentDir.removeEntry(fileName);
}

/**
 * Checks if the given path corresponds to a file in OPFS.
 * @param path The file path to check.
 * @returns true if the path is a file, false otherwise.
 */
export async function isOPFSFile(path: string): Promise<boolean> {
  try {
    const root = await navigator.storage.getDirectory();
    const parts = path.split("/").filter(Boolean);
    let handle: FileSystemDirectoryHandle | FileSystemFileHandle = root;

    for (let i = 0; i < parts.length - 1; i++) {
      handle = await (handle as FileSystemDirectoryHandle).getDirectoryHandle(
        parts[i],
        { create: false }
      );
    }
    const name = parts[parts.length - 1];

    try {
      await (handle as FileSystemDirectoryHandle).getFileHandle(name, {
        create: false,
      });
      return true;
    } catch {
      return false;
    }
  } catch {
    return false;
  }
}

/**
 * Checks if the given path corresponds to a directory in OPFS.
 * @param path The directory path to check.
 * @returns true if the path is a directory, false otherwise.
 */
export async function isOPFSDir(path: string): Promise<boolean> {
  try {
    const root = await navigator.storage.getDirectory();
    const parts = path.split("/").filter(Boolean);
    let handle: FileSystemDirectoryHandle | FileSystemFileHandle = root;

    for (let i = 0; i < parts.length - 1; i++) {
      handle = await (handle as FileSystemDirectoryHandle).getDirectoryHandle(
        parts[i],
        { create: false }
      );
    }
    const name = parts[parts.length - 1];

    try {
      await (handle as FileSystemDirectoryHandle).getDirectoryHandle(name, {
        create: false,
      });
      return true;
    } catch {
      return false;
    }
  } catch {
    return false;
  }
}
