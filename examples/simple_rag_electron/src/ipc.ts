import { BrowserWindow, ipcMain, dialog } from "electron";
import * as fs from "fs/promises";
import * as ai from "ailoy-node";

let runtime: ai.Runtime | undefined = undefined;
let agent: ai.Agent | undefined = undefined;
let vectorstore: ai.VectorStore | undefined = undefined;

export const initializeComponents = async (mainWindow: BrowserWindow) => {
  if (runtime === undefined) {
    runtime = new ai.Runtime();
    await runtime.start();
  }

  if (vectorstore === undefined) {
    vectorstore = await ai.defineVectorStore(runtime, "BAAI/bge-m3", "faiss");
  }

  if (agent === undefined) {
    mainWindow.webContents.send(
      "indicate-loading",
      "Loading AI model...",
      false
    );
    agent = await ai.defineAgent(
      runtime,
      ai.LocalModel({ id: "Qwen/Qwen3-8B" })
    );
    mainWindow.webContents.send("indicate-loading", "", true);
  }
};

export const registerIpcHandlers = async (mainWindow: BrowserWindow) => {
  ipcMain.handle("open-file", async () => {
    const result = await dialog.showOpenDialog({
      properties: ["openFile"],
      filters: [{ name: "Text Files", extensions: ["txt"] }],
    });

    if (!result.canceled && result.filePaths.length > 0) {
      const content = await fs.readFile(result.filePaths[0], "utf-8");
      return content;
    }
    return null;
  });

  ipcMain.handle("update-vector-store", async (event, document: string) => {
    mainWindow.webContents.send("vector-store-update-started");

    await vectorstore.clear();

    // split texts into chunks
    const { chunks }: { chunks: Array<string> } = await runtime.call(
      "split_text",
      {
        text: document,
        chunk_size: 500,
        chunk_overlap: 200,
      }
    );

    let chunkIdx = 0;
    for (const chunk of chunks) {
      await vectorstore.insert({
        document: chunk,
        metadata: null,
      });
      mainWindow.webContents.send(
        "vector-store-update-progress",
        chunkIdx + 1,
        chunks.length
      );
      chunkIdx += 1;
    }

    mainWindow.webContents.send("vector-store-update-finished");
  });

  ipcMain.handle("retrieve-similar-documents", async (event, query: string) => {
    const results = await vectorstore.retrieve(query, 5);
    return results;
  });

  ipcMain.handle("infer-language-model", async (event, message: string) => {
    for await (const resp of agent.query(message)) {
      mainWindow.webContents.send("assistant-answer", resp);
    }
  });
};

export const removeIpcHandlers = () => {
  ipcMain.removeHandler("open-file");
  ipcMain.removeHandler("update-vector-store");
  ipcMain.removeHandler("retrieve-similar-documents");
  ipcMain.removeHandler("infer-language-model");
};

export const destroyComponents = async () => {
  if (vectorstore !== undefined) {
    await vectorstore.delete();
    vectorstore = undefined;
  }
  if (agent !== undefined) {
    await agent.delete();
    agent = undefined;
  }
  if (runtime !== undefined) {
    await runtime.stop();
    runtime = undefined;
  }
};
