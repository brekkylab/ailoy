<p align="center">
  <picture>
    <img alt="Ailoy" src="https://brekkylab.github.io/ailoy/img/ailoy-logo-letter.png" width="352" style="max-width: 50%;">
  </picture>
</p>

<h3 align="center">Comprehensive library for building intelligent AI agents</h3>
<p align="center">
  <img src="https://cdn.simpleicons.org/python" width="16"/> <a href="https://pypi.org/project/ailoy-py/"><img src="https://img.shields.io/pypi/v/ailoy-py?color=blue&label=ailoy-py" alt="PyPI"></a>
  <img src="https://cdn.simpleicons.org/nodedotjs" width="16"/> <a href="https://www.npmjs.com/package/ailoy-node"><img src="https://img.shields.io/npm/v/ailoy-node?label=ailoy-node&color=339933" alt="npm node"></a>
  <img src="https://cdn.simpleicons.org/webassembly" width="16"/> <a href="https://www.npmjs.com/package/ailoy-web"><img src="https://img.shields.io/npm/v/ailoy-web?label=ailoy-web&color=654ff0" alt="npm web"></a>
</p>

</p>
<p align="center">
  <a href="https://brekkylab.github.io/ailoy/"><img src="https://img.shields.io/badge/docs-latest-5a9cae" alt="Documentation"></a>
  <a href="https://discord.gg/27rx3EJy3P"><img src="https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://x.com/ailoy_co"><img src="https://img.shields.io/badge/X-000000?logo=x&logoColor=white" alt="X"></a>
</p>

## 🚀 See how it is easy by an example

### Get your agent just in a **single line of code**. 



```python
import ailoy as ai

# Create an agent with a local model in a single line of code.
agent = ai.Agent(ai.LangModel.new_local_sync("Qwen/Qwen3-8B"))

# Get the response from the agent simply by calling the `run` method.
response = agent.run("Explain quantum computing in one sentence")
print(response.contents[0].text)
```

### 🌐 Browser-Native AI (WebAssembly)

You can build your agent entirely in the browser using WebAssembly just in a few lines of code.

```typescript
import * as ai from "ailoy-web";

// Check WebGPU support
const { supported } = await ai.isWebGPUSupported();

// Run AI entirely in the browser - no server needed!
const agent = new ai.Agent(
  await ai.LangModel.newLocal("Qwen/Qwen3-0.6B")
);
```

### 🔥 Quick-customizable Web Agent UI Boilerplate

Just **Copy&Paste** to build your own web agent in minutes.

- https://github.com/brekkylab/ailoy-web-ui



## Key Features

### 🚀 Super Simple Framework for AI Agents

- No boilerplate, no complex setup

### 💻 Cross-Platform & Multi-Language APIs

- Provide <img src="https://cdn.simpleicons.org/python" width="16"/> **Python** and <img src="https://cdn.simpleicons.org/nodedotjs" width="16"/> **JavaScript** APIs

- Support <img src="https://www.microsoft.com/favicon.ico?v2" width="16"/> **Windows**, <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" width="16"/> **Linux**, and <img src="https://www.apple.com/favicon.ico" width="16"/> **macOS**

- Support Synchronous and Asynchronous APIs

### 🌐 Browser-Native AI (WebAssembly)

- Run AI entirely in the browser - no server needed!

### ☁️ Cloud API or 💻 Local Model, whatever you want!

- Use a single API to work with both local and remote (API) models. That flexibility keeps you in full control of your stack.


### 🔧 Built-in RAG & Tools

- Vector stores, MCP integration, function calling out of the box

### <img src="https://cdn.simpleicons.org/rust" width="16"/> Rust-Powered
- Fast, memory-safe, minimal dependencies

### 📚 Documentation & Community

- 📚 [Documentation](https://brekkylab.github.io/ailoy/) - English and Korean. (To be translated into other languages)
- <img src="https://cdn.simpleicons.org/discord" width="16"/> [Discord Community](https://discord.gg/27rx3EJy3P) - Join to ask questions, share your projects, and get help.


<br/>

## Usage

### Installation

```bash
pip install ailoy-py      # Python
npm install ailoy-node    # Node.js
npm install ailoy-web     # Browser (WebAssembly)
```



### Supported Models

**Local** (runs on your hardware):

- Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 32B

**Cloud APIs**:

- OpenAI (GPT-4o, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Grok

### Platform Support

**Languages**
- <img src="https://cdn.simpleicons.org/python" width="16"/> **Python**
- <img src="https://cdn.simpleicons.org/nodedotjs" width="16"/> **JavaScript**

**Platforms**
- <img src="https://www.microsoft.com/favicon.ico?v2" width="16"/> **Windows**
- <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" width="16"/> **Linux**
- <img src="https://www.apple.com/favicon.ico" width="16"/> **macOS**

**System Requirements for Local AI:**

- <img src="https://www.apple.com/favicon.ico" width="16"/> **macOS**: Apple Silicon with Metal
- <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" width="16"/> **Linux** and <img src="https://www.microsoft.com/favicon.ico?v2" width="16"/> **Windows**: Vulkan 1.3 compatible GPU
- <img src="https://cdn.simpleicons.org/webassembly" width="16"/> **Web Browser**: WebGPU with shader-f16 support

## Example Projects

| Project                                            | Description                          |
| -------------------------------------------------- | ------------------------------------ |
| [Gradio Chatbot](./examples/gradio_chatbot)        | Web UI chatbot with tool integration |
| [Web Assistant](./examples/web-assistant-ui)       | Browser-based AI assistant (WASM)    |
| [RAG Electron App](./examples/simple_rag_electron) | Desktop app with document Q&A        |
| [MCP Integration](./examples/mcp_examples)         | GitHub & Playwright tools via MCP    |

---

> [!WARNING]
> Ailoy is under active development. APIs may change with version updates.

> [!TIP]
> Questions? Join our [Discord](https://discord.gg/27rx3EJy3P)!
