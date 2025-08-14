import { Thread } from "@/components/assistant-ui/thread";
import { AiloyRuntimeProvider } from "./AiloyRuntimeProvider";

function App() {
  return (
    <AiloyRuntimeProvider>
      <Thread />
    </AiloyRuntimeProvider>
  );
}

export default App;
