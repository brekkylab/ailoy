# ailoy-js-node

Node.js binding for Ailoy runtime APIs

## Install
```bash
# npm
npm install ailoy-js-node
# yarn
yarn add ailoy-js-node
```

## Quickstart

```typescript
import * as ailoy from "ailoy-js-node";

// Create a new runtime
const runtime = new ailoy.Runtime();

// Start runtime
await runtime.start();

// Call function
for await (const resp of runtime.call("echo", "hello world!")) {
  console.log(resp);
  // Prints "hello world!"
}

// Define components
await rt.define("tvm_language_model", "lm0", {
  model: "Qwen/Qwen3-8B",
});
// Returns true

// Call method and iterate for responses
for await (const resp of runtime.callIterMethod("lm0", "infer", {
  messages: [{role: "user", content: "Hello World!"}],
})) {
  console.log(resp);
  // Prints { finish_reason: null, message: { role: "assistant", content: "<token>"} } until finished
}

```

