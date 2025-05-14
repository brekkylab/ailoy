import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronAPI", {
  /** Client -> Server IPCs */
  openFile: async () => await ipcRenderer.invoke("open-file"),
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
  onTextLoadProgress: (callback: (loaded: number, total: number) => void) =>
    ipcRenderer.on(
      "text-load-progress",
      (_event, loaded: number, total: number) => callback(loaded, total)
    ),
  onFileSelected: (callback: () => void) => {
    ipcRenderer.on("file-selected", () => callback());
  },
  onAssistantAnswer: (callback: (resp: AgentResponse) => void) => {
    ipcRenderer.on("assistant-answer", (event, resp: AgentResponse) =>
      callback(resp)
    );
  },
});
