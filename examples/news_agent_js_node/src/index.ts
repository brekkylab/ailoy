import { startRuntime, defineAgent } from "ailoy-node";
import type { ToolAuthenticator } from "ailoy-node";

import readline from "readline";

function getUserInput(query: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) =>
    rl.question(query, (answer) => {
      rl.close();
      resolve(answer);
    })
  );
}

async function main() {
  const rt = await startRuntime();

  const agent = await defineAgent(rt, "qwen3-8b");

  let nytimesApiKey = process.env.NYTIMES_API_KEY;
  if (nytimesApiKey === undefined) {
    nytimesApiKey = await getUserInput("Enter New York Times API Key: ");
  }

  const authenticator: ToolAuthenticator = (request: any) => {
    let url = new URL(request.url);
    url.searchParams.append("api-key", nytimesApiKey);
    request.url = url.toString();
    return request;
  };
  agent.addToolsFromPreset("nytimes", {
    authenticator,
  });

  // "Find me some articles about 'Agentic AI'."
  console.log(
    'Simple news assistant (Please type "exit" to stop conversation)'
  );
  while (true) {
    const query = await getUserInput("User: ");
    if (query === "" || query === "exit") break;

    process.stdout.write(`\nAssistant: `);
    for await (const resp of agent.query(query)) {
      agent.print(resp);
    }
  }

  await agent.delete();

  await rt.stop();
}

main().catch((err) => {
  console.error("Unhandled error:", err);
  process.exit(1);
});
