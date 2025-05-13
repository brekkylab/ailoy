import type {
  VectorStoreRetrieveItem,
  AgentResponse as _AgentResponse,
} from "ailoy-node";

export interface IElectronAPI {
  openFile: () => Promise<string>;
  retrieveSimilarDocuments: (
    query: string
  ) => Promise<Array<VectorStoreRetrieveItem>>;
  inferLanguageModel: (message: string) => Promise<void>;
  onIndicateLoading: (
    callback: (indicator: string, finished: boolean) => void
  ) => void;
  onTextLoadProgress: (
    callback: (loaded: number, total: number) => void
  ) => void;
  onFileSelected: (callback: () => void) => void;
  onAssistantAnswer: (callback: (delta: MessageDelta) => void) => void;
}

declare global {
  interface Window {
    electronAPI: IElectronAPI;
  }
  type AgentResponse = _AgentResponse;
}
