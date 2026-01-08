import { BrowserWindow, ipcMain, dialog } from "electron";
import * as fs from "fs/promises";
import * as ai from "ailoy-node";

let agent: ai.Agent | undefined = undefined;
let embeddingModel: ai.EmbeddingModel | undefined = undefined;
let vectorstore: ai.VectorStore | undefined = undefined;
let knowledge: ai.Knowledge | undefined = undefined;

function delay(t: number, val?: any) {
  return new Promise(function (resolve) {
    setTimeout(function () {
      resolve(val);
    }, t);
  });
}

export const initializeComponents = async (mainWindow: BrowserWindow) => {
  await delay(1000);

  if (knowledge === undefined) {
    embeddingModel = await ai.EmbeddingModel.newLocal("BAAI/bge-m3", {
      progressCallback: (prog) => {
        const percent = Math.round((prog.current / prog.total) * 100);
        mainWindow.webContents.send(
          "indicate-loading",
          `Initializing Embedding Model... ${percent}%`,
          false
        );
      },
      validateChecksum: true,
    });
    vectorstore = await ai.VectorStore.newFaiss(1024);
    knowledge = ai.Knowledge.newVectorStore(vectorstore, embeddingModel);
  }

  if (agent === undefined) {
    const langmodel = await ai.LangModel.newLocal("Qwen/Qwen3-8B", {
      progressCallback: (prog) => {
        const percent = Math.round((prog.current / prog.total) * 100);
        mainWindow.webContents.send(
          "indicate-loading",
          `Initializing Language Model... ${percent}%`,
          false
        );
      },
      validateChecksum: true,
    });
    agent = new ai.Agent(langmodel);
    agent.setKnowledge(knowledge);
  }

  mainWindow.webContents.send("indicate-loading", "", true);
};

function splitTextIntoChunks(
  text: string,
  chunkSize: number,
  chunkOverlap: number
) {
  const chunks = [];
  let startIndex = 0;

  while (startIndex < text.length) {
    let endIndex = startIndex + chunkSize;
    // Ensure endIndex does not exceed text length
    if (endIndex > text.length) {
      endIndex = text.length;
    }

    // Extract the chunk
    const chunk = text.substring(startIndex, endIndex);
    chunks.push(chunk);

    // Calculate the next startIndex for the next chunk with overlap
    startIndex += chunkSize - chunkOverlap;

    // Prevent negative or excessive overlap if the remaining text is too short
    if (startIndex < 0) {
      startIndex = 0;
    }
  }
  return chunks;
}

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
    const chunks = splitTextIntoChunks(document, 500, 200);

    let chunkIdx = 0;
    for (const chunk of chunks) {
      const embedding = await embeddingModel.infer(chunk);
      await vectorstore.addVector({
        embedding,
        document: chunk,
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

  ipcMain.handle("infer-language-model", async (event, messages: Message[]) => {
    for await (const resp of agent.runDelta(messages, {
      inference: {
        documentPolyfill: ai.getDocumentPolyfill("Qwen3"),
      },
      knowledge: {
        topK: 5,
      },
    })) {
      if (
        resp.delta.contents.length > 0 &&
        resp.delta.contents[0].type === "text"
      ) {
        mainWindow.webContents.send(
          "assistant-answer",
          resp.delta.contents[0].text
        );
      }
    }
    mainWindow.webContents.send("assistant-answer-finished");
  });
};

export const removeIpcHandlers = () => {
  ipcMain.removeHandler("open-file");
  ipcMain.removeHandler("update-vector-store");
  ipcMain.removeHandler("infer-language-model");
};

export const destroyComponents = async () => {
  if (knowledge !== undefined) {
    knowledge = undefined;
    vectorstore = undefined;
    embeddingModel = undefined;
  }

  if (agent !== undefined) {
    agent = undefined;
  }
};
