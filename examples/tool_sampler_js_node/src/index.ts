import {
  startRuntime,
  defineAgent,
  ToolAuthenticator,
  bearerAutenticator,
} from "ailoy-node";
import { select, password, input } from "@inquirer/prompts";

async function main() {
  // Start the Ailoy runtime
  const rt = await startRuntime();

  // Select and initialize an agent (model)
  async function getAgent() {
    const model_name = await select({
      message: "Select the model",
      choices: [
        {
          name: "Qwen/Qwen3-8B",
          value: "Qwen/Qwen3-8B",
        },
        {
          name: "Qwen/Qwen3-4B",
          value: "Qwen/Qwen3-4B",
        },
        {
          name: "Qwen/Qwen3-1.7B",
          value: "Qwen/Qwen3-1.7B",
        },
        {
          name: "Qwen/Qwen3-0.6B",
          value: "Qwen/Qwen3-0.6B",
        },
        {
          name: "gpt-4o",
          value: "gpt-4o",
        },
      ],
    });
    if (model_name === "gpt-4o") {
      let apiKey = process.env.OPENAI_API_KEY;
      if (apiKey === undefined)
        apiKey = await password({ message: "Enter OpenAI API key:" });
      return await defineAgent(rt, model_name as any, { apiKey });
    } else {
      return await defineAgent(rt, model_name as any);
    }
  }

  const agent = await getAgent();

  // Select and configure tool to use
  async function getTool() {
    const preset_name = await select({
      message: "Select the tool",
      choices: [
        {
          name: "Frankfurter",
          value: "frankfurter",
          description: "Real-time currency API (https://frankfurter.dev/)",
        },
        {
          name: "The Movie Database (TMDB)",
          value: "TMDB",
          description:
            "Information about the movie (https://developer.themoviedb.org/reference/intro/getting-started)",
        },
        {
          name: "New York Times",
          value: "nytimes",
          description:
            "New York Times articles (https://developer.nytimes.com/apis)",
        },
      ],
    });
    if (preset_name === "TMDB") {
      let api_key = process.env.TMDB_API_KEY;
      if (api_key === undefined) {
        api_key = await password({ message: "Enter TMDB API key:" });
      }
      agent.addToolsFromPreset("tmdb", {
        authenticator: bearerAutenticator(api_key),
      });
    } else if (preset_name === "nytimes") {
      let api_key = process.env.NYTIMES_API_KEY;
      if (api_key === undefined) {
        api_key = await password({ message: "Enter NYTimes API key:" });
      }
      const authenticator: ToolAuthenticator = (request: any) => {
        let url = new URL(request.url);
        url.searchParams.append("api-key", api_key);
        request.url = url.toString();
        return request;
      };
      agent.addToolsFromPreset("nytimes", { authenticator });
    } else {
      agent.addToolsFromPreset(preset_name);
    }
  }

  await getTool();

  // Ask whether reasoning mode should be enabled
  const enableReasoning =
    (
      await input({
        message:
          "Do you want to enable reasoning? (Please type 'y' to enable):",
      })
    ).toLowerCase() === "y";

  // Start conversation loop
  while (true) {
    const prompt = await input({
      message: 'User (Please type "exit" to stop conversation):',
    });
    if (prompt === "exit") break;
    if (prompt === "") continue;

    // Stream assistant's response
    for await (const resp of agent.query(prompt, { enableReasoning })) {
      agent.print(resp);
    }
  }

  await rt.stop();
}

main().catch((err) => {
  console.error("Unhandled error:", err);
  process.exit(1);
});
