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

  const agent = await defineAgent(rt, "Qwen/Qwen3-8B");
  agent.addToolsFromPreset("tmdb", {
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
    for await (const resp of agent.query(query, { enableReasoning: true })) {
      agent.print(resp);
    }
  }

  await rt.stop();
}

main().catch((err) => {
  console.error("Unhandled error:", err);
  process.exit(1);
});
