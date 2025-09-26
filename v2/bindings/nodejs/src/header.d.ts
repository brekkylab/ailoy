// prettier-ignore
/* @ts-ignore */
export type LanguageModel = LocalLanguageModel | OpenAILanguageModel | GeminiLanguageModel | AnthropicLanguageModel | XAILanguageModel;
/* @ts-ignore */
export type EmbeddingModel = LocalEmbeddingModel;
/* @ts-ignore */
export type VectorStore = FaissVectorStore | ChromaVectorStore;
/* @ts-ignore */
export type Tool = BuiltinTool | MCPTool | JsFunctionTool;
