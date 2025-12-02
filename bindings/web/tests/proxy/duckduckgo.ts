import * as http from "http";
import * as https from "https";

const PROXY_TARGET = "https://html.duckduckgo.com/html";

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(200, getCorsHeaders());
    res.end();
    return;
  }

  if (req.method === "POST") {
    handleRequest(req, res);
    return;
  }

  res.writeHead(405, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Method not allowed" }));
});

async function handleRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  try {
    const body = await readBody(req);

    const headers = {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
      "Content-Type": "application/x-www-form-urlencoded",
      "Content-Length": Buffer.byteLength(body).toString(),
    };

    const proxyReq = https.request(
      PROXY_TARGET,
      {
        method: "POST",
        headers,
      },
      (proxyRes) => {
        const responseBody: Buffer[] = [];
        proxyRes.on("data", (chunk: Buffer) => {
          responseBody.push(chunk);
        });
        proxyRes.on("end", () => {
          const fullBody = Buffer.concat(responseBody);
          res.writeHead(proxyRes.statusCode || 200, {
            ...getCorsHeaders(),
            ...getFilteredHeaders(proxyRes.headers),
          });
          res.end(fullBody);
        });
      }
    );

    proxyReq.on("error", (error: Error) => {
      res.writeHead(500, {
        ...getCorsHeaders(),
        "Content-Type": "application/json",
      });
      res.end(JSON.stringify({ error: error.message }));
    });

    proxyReq.write(body);
    proxyReq.end();
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    res.writeHead(500, {
      ...getCorsHeaders(),
      "Content-Type": "application/json",
    });
    res.end(JSON.stringify({ error: errorMessage }));
  }
}

function readBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      chunks.push(chunk);
    });
    req.on("end", () => {
      resolve(Buffer.concat(chunks));
    });
    req.on("error", reject);
  });
}

function getCorsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function getFilteredHeaders(
  headers: http.IncomingHttpHeaders
): Record<string, string | string[]> {
  const excludedHeaders = new Set([
    "transfer-encoding",
    "connection",
    "content-encoding",
  ]);
  const filtered: Record<string, string | string[]> = {};

  for (const [key, value] of Object.entries(headers)) {
    if (!excludedHeaders.has(key.toLowerCase())) {
      filtered[key] = value as string | string[];
    }
  }

  return filtered;
}

export default server;
