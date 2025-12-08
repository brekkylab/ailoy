import * as http from "http";
import * as https from "https";
import * as url from "url";

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36";

const server = http.createServer(async (req, res) => {
  const parsedUrl = url.parse(req.url || "", true);
  const pathname = parsedUrl.pathname || "";

  if (req.method === "OPTIONS") {
    res.writeHead(200, getCorsHeaders());
    res.end();
    return;
  }

  if (pathname === "/web-search-duckduckgo" && req.method === "POST") {
    handleDuckDuckGoRequest(req, res);
    return;
  }

  if (pathname === "/web-fetch" && req.method === "GET") {
    handleWebFetchRequest(req, res, parsedUrl.query);
    return;
  }

  res.writeHead(405, {
    ...getCorsHeaders(),
    "Content-Type": "application/json",
  });
  res.end(JSON.stringify({ error: "Method not allowed" }));
});

async function handleDuckDuckGoRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  try {
    const body = await readBody(req);

    const headers = {
      "User-Agent": USER_AGENT,
      "Content-Type": "application/x-www-form-urlencoded",
      "Content-Length": Buffer.byteLength(body).toString(),
    };

    const proxyReq = https.request(
      "https://html.duckduckgo.com/html",
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

async function handleWebFetchRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  query: Record<string, string | string[] | undefined>
): Promise<void> {
  try {
    const targetUrl = query.url as string;

    if (!targetUrl) {
      res.writeHead(400, {
        ...getCorsHeaders(),
        "Content-Type": "application/json",
      });
      res.end(JSON.stringify({ error: "Missing 'url' query parameter" }));
      return;
    }

    const headers = {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    };

    await fetchWithRedirects(targetUrl, headers, res);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    res.writeHead(500, {
      ...getCorsHeaders(),
      "Content-Type": "application/json",
    });
    res.end(JSON.stringify({ error: errorMessage }));
  }
}

async function fetchWithRedirects(
  targetUrl: string,
  headers: Record<string, string>,
  res: http.ServerResponse,
  maxRedirects: number = 10
): Promise<void> {
  if (maxRedirects <= 0) {
    res.writeHead(500, {
      ...getCorsHeaders(),
      "Content-Type": "application/json",
    });
    res.end(JSON.stringify({ error: "Too many redirects" }));
    return;
  }

  const parsedUrl = url.parse(targetUrl);
  const protocol = parsedUrl.protocol === "https:" ? https : http;

  const proxyReq = protocol.request(targetUrl, { headers }, (proxyRes) => {
    const statusCode = proxyRes.statusCode || 200;

    // Handle redirects
    if (statusCode >= 300 && statusCode < 400 && proxyRes.headers.location) {
      const redirectUrl = proxyRes.headers.location as string;
      // Resolve relative URLs
      const absoluteRedirectUrl = new URL(redirectUrl, targetUrl).toString();
      fetchWithRedirects(absoluteRedirectUrl, headers, res, maxRedirects - 1);
      return;
    }

    const responseBody: Buffer[] = [];
    proxyRes.on("data", (chunk: Buffer) => {
      responseBody.push(chunk);
    });
    proxyRes.on("end", () => {
      const fullBody = Buffer.concat(responseBody);
      res.writeHead(statusCode, {
        ...getCorsHeaders(),
        ...getFilteredHeaders(proxyRes.headers),
      });
      res.end(fullBody);
    });
  });

  proxyReq.on("error", (error: Error) => {
    res.writeHead(500, {
      ...getCorsHeaders(),
      "Content-Type": "application/json",
    });
    res.end(JSON.stringify({ error: error.message }));
  });

  proxyReq.end();
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
