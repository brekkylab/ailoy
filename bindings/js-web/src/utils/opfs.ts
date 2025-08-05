type DataType = "text" | "json" | "arraybuffer";

export function joinPath(basePath: string, fileName: string) {
  return [basePath, fileName].join("/").replace(/\/+/g, "/");
}

/**
 * OPFS에서 파일을 읽어옵니다.
 * @param path 파일 경로 (예시: "/folder/file.txt")
 * @param type 가져올 데이터 타입: 'text' | 'json' | 'arraybuffer'
 * @returns 파일 내용 (타입에 따라 텍스트, 객체, 또는 ArrayBuffer)
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
 * OPFS에 파일을 저장합니다.
 * @param path 저장할 파일 전체 경로 (예: "/folder/file.txt")
 * @param data 저장할 데이터 (string, object, 또는 ArrayBuffer)
 * @param type 데이터 타입 구분: text, json, arraybuffer
 */
export async function writeOPFSFile(
  path: string,
  data: string | object | ArrayBuffer,
  type: DataType = "text"
): Promise<void> {
  let dirHandle = await navigator.storage.getDirectory();
  const parts = path.split("/").filter(Boolean);

  for (let i = 0; i < parts.length - 1; i++)
    dirHandle = await dirHandle.getDirectoryHandle(parts[i], { create: true });

  const fileName = parts[parts.length - 1];
  const fileHandle = await dirHandle.getFileHandle(fileName, { create: true });
  const writable = await fileHandle.createWritable();

  if (type === "text") {
    await writable.write(data as string);
  } else if (type === "json") {
    await writable.write(JSON.stringify(data));
  } else if (type === "arraybuffer") {
    await writable.write(data as ArrayBuffer);
  } else {
    throw new Error("지원하지 않는 타입입니다.");
  }

  await writable.close();
}

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
