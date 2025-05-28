import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  /** Client -> Server IPCs */
  openFile: async () => await ipcRenderer.invoke("open-file"),
  updateVectorStore: async (document: string) =>
    await ipcRenderer.invoke("update-vector-store", document),
  retrieveSimilarDocuments: async (query: string) =>
    await ipcRenderer.invoke("retrieve-similar-documents", query),
  inferLanguageModel: async (message: string) =>
    await ipcRenderer.invoke("infer-language-model", message),

  /** Server -> Client IPCs */
  onIndicateLoading: (
    callback: (indicator: string, finished: boolean) => void
  ) => {
    ipcRenderer.on(
      "indicate-loading",
      (_event, indicator: string, finished: boolean) =>
        callback(indicator, finished)
    );
  },
  onVectorStoreUpdateStarted: (callback: () => void) =>
    ipcRenderer.on("vector-store-update-started", () => callback()),
  onVectorStoreUpdateProgress: (
    callback: (loaded: number, total: number) => void
  ) =>
    ipcRenderer.on(
      "vector-store-update-progress",
      (_event, loaded: number, total: number) => callback(loaded, total)
    ),
  onVectorStoreUpdateFinished: (callback: () => void) =>
    ipcRenderer.on("vector-store-update-finished", () => callback()),
  onAssistantAnswer: (callback: (resp: AgentResponse) => void) => {
    ipcRenderer.on("assistant-answer", (event, resp: AgentResponse) =>
      callback(resp)
    );
  },
});
