import { startRuntime, defineAgent, bearerAutenticator } from "ailoy-node";

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
  let tmdbApiKey = process.env.TMDB_API_KEY;
  if (tmdbApiKey === undefined) {
    tmdbApiKey = await getUserInput("Enter TMDB API Key: ");
  }
  let openaiApiKey = process.env.OPENAI_API_KEY;
  if (openaiApiKey === undefined) {
    openaiApiKey = await getUserInput("Enter OpenAI API Key: ");
  }

  console.log("Initializing AI...");

  const ex = await defineAgent(rt, "qwen3-8b");
  ex.addToolsFromPreset("tmdb", {
    authenticator: bearerAutenticator(tmdbApiKey),
  });

  // "I'm looking for a movie similar to Avatar.";
  // "Please give me a link can watch Avatar."
  // "Please recommend me a branding-new movie with good reviews"
  console.log(
    'Simple movie assistant (Please type "exit" to stop conversation)'
  );
  while (true) {
    const query = await getUserInput("User: ");
    if (query === "" || query === "exit") break;

    process.stdout.write(`\nAssistant: `);
    for await (const resp of ex.run(query, { enableReasoning: true })) {
      if (resp.type === "output_text") {
        process.stdout.write(resp.content);
        if (resp.endOfTurn) {
          process.stdout.write("\n\n");
        }
      } else if (resp.type === "reasoning") {
        process.stdout.write(`\x1b[93m${resp.content}\x1b[0m`);
        if (resp.endOfTurn) {
          process.stdout.write("\n\n");
        }
      } else if (resp.type === "tool_call") {
        process.stdout.write(`
Tool Call
- ID: ${resp.content.id}
- name: ${resp.content.function.name}
- arguments: ${JSON.stringify(resp.content.function.arguments)}
`);
      } else if (resp.type === "tool_call_result") {
        process.stdout.write(`
Tool Call Result
- ID: ${resp.content.tool_call_id}
- name: ${resp.content.name}
- Result: ${resp.content.content}
`);
      }
    }
  }

  await rt.stop();
}

main().catch((err) => {
  console.error("Unhandled error:", err);
  process.exit(1);
});
