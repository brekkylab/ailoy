import React, { useState, useEffect } from "react";
import Markdown from "marked-react";

import "./index.css";

type LoadingStatus = "loading" | "loaded";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const App: React.FC = () => {
  const [loadingStatus, setLoadingStatus] = useState<LoadingStatus>("loaded");
  const [loadingIndicator, setLoadingIndicator] = useState<string>("");
  const [textContent, setTextContent] = useState("");
  const [currentContent, setCurrentContent] = useState("");
  const [query, setQuery] = useState("");

  const [messages, setMessages] = useState<Array<Message>>([]);
  const [isAnswering, setIsAnswering] = useState<boolean>(false);

  useEffect(() => {
    window.electronAPI.onIndicateLoading((indicator, finished) => {
      console.log(indicator, finished);
      setLoadingIndicator(indicator);
      setLoadingStatus(finished ? "loaded" : "loading");
    });
    window.electronAPI.onVectorStoreUpdateStarted(() => {
      setLoadingStatus("loading");
    });
    window.electronAPI.onVectorStoreUpdateProgress(
      (loaded: number, total: number) => {
        const progress = Math.round((100 * loaded) / total);
        setLoadingIndicator(`Loading texts to vector store: ${progress}%`);
      }
    );
    window.electronAPI.onVectorStoreUpdateFinished(() => {
      setLoadingStatus("loaded");
      setLoadingIndicator("");
    });
    window.electronAPI.onAssistantAnswer((resp: AgentResponse) => {
      if (resp.type === "output_text") {
        setMessages((prevMessages) => {
          const last = prevMessages[prevMessages.length - 1];
          last.content += resp.content;
          return [...prevMessages.slice(0, prevMessages.length - 1), last];
        });
        if (resp.endOfTurn) {
          setIsAnswering(false);
        }
      }
    });
  }, []);

  const handleFileLoad = async () => {
    const content = await window.electronAPI.openFile();
    setCurrentContent(content);
  };

  const handleUpdateVectorStore = async () => {
    if (textContent === currentContent || currentContent === "") return;
    setTextContent(currentContent);
    await window.electronAPI.updateVectorStore(currentContent);
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // Clear query prompt
    setQuery("");
    // Add user message and empty assistant message
    setMessages([
      ...messages,
      { role: "user", content: query },
      { role: "assistant", content: "" },
    ]);
    // Retrieve similar documents from vector store
    const results = await window.electronAPI.retrieveSimilarDocuments(query);
    // TODO: visualize retrieved results
    console.log(results);
    // Augment results to prompt and infer
    const augmentedPrompt = `
Based on the given context, answer to user's query.

Context: 
${results.map((result) => result.document).join("\n")}


Query: ${query}
`;

    setIsAnswering(true);
    window.electronAPI.inferLanguageModel(augmentedPrompt);
  };

  return (
    <main className="h-screen flex flex-col">
      <div className="flex flex-1 overflow-hidden">
        <div className="w-1/2 p-4 flex flex-col ">
          <div className="w-full flex justify-between gap-2">
            <button
              className="mb-4 bg-blue-500 text-white px-4 py-2 rounded cursor-pointer"
              onClick={handleFileLoad}
            >
              Load Text File
            </button>
            <button
              className="mb-4 bg-yellow-400 px-4 py-1 rounded cursor-pointer disabled:opacity-30 disabled:cursor-default"
              disabled={textContent === currentContent || currentContent === ""}
              onClick={handleUpdateVectorStore}
            >
              Update Vectorstore
            </button>
          </div>
          <textarea
            className="w-full h-full border rounded p-2"
            value={currentContent}
            onChange={(e) => setCurrentContent(e.target.value)}
          />
        </div>
        <div className="w-1/2 p-4 overflow-auto">
          {messages.map((message, index) => (
            <div key={index} className="mb-2 p-2 border rounded overflow-scroll">
              <Markdown>
                {`${message.role}: ${message.content}`}
              </Markdown>
            </div>
          ))}
        </div>
      </div>
      <form
        className="p-4 border-t flex gap-2 items-center"
        onSubmit={handleSubmit}
      >
        <input
          type="text"
          className="w-full p-2 border rounded"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query"
        />
        <button
          type="submit"
          className="bg-green-500 text-white px-4 py-2 rounded cursor-pointer disabled:opacity-70 disabled:cursor-default"
          disabled={isAnswering || loadingStatus !== "loaded" || query === ""}
        >
          Submit
        </button>
      </form>
      {loadingStatus === "loading" && (
        <div className="w-full h-full absolute bg-gray-700 opacity-70 flex flex-col justify-center items-center">
          <h1 className="text-gray-300 text-4xl">{loadingIndicator}</h1>
        </div>
      )}
    </main>
  );
};

export default App;
