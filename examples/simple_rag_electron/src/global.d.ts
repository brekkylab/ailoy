import type { VectorStoreRetrieveItem, Message as _Message } from "ailoy-node";

export interface IElectronAPI {
  openFile: () => Promise<string>;
  updateVectorStore: (document: string) => Promise<void>;
  inferLanguageModel: (messages: Array<Message>) => Promise<void>;

  onIndicateLoading: (
    callback: (indicator: string, finished: boolean) => void
  ) => void;
  onVectorStoreUpdateStarted: (callback: () => void) => void;
  onVectorStoreUpdateProgress: (
    callback: (loaded: number, total: number) => void
  ) => void;
  onVectorStoreUpdateFinished: (callback: () => void) => void;
  onAssistantAnswer: (callback: (delta: string) => void) => void;
  onAssistantAnswerFinished: (callback: () => void) => void;
}

declare global {
  interface Window {
    electronAPI: IElectronAPI;
  }
  type Message = _Message;
}
