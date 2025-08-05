// This server implementation is from https://github.com/invariantlabs-ai/mcp-streamable-http

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
  Notification,
  CallToolRequestSchema,
  ListToolsRequestSchema,
  LoggingMessageNotification,
  ToolListChangedNotification,
  JSONRPCNotification,
  JSONRPCError,
  InitializeRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { randomUUID } from "crypto";
import express, { Request, Response } from "express";
import cors from "cors";

const SESSION_ID_HEADER_NAME = "mcp-session-id";
const JSON_RPC = "2.0";
const NWS_API_BASE = "https://api.weather.gov";
const USER_AGENT = "weather-app/1.0";

// Helper function for making NWS API requests
async function makeNWSRequest<T>(url: string): Promise<T | null> {
  const headers = {
    "User-Agent": USER_AGENT,
    Accept: "application/geo+json",
  };

  try {
    const response = await fetch(url, { headers });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    console.error("Error making NWS request:", error);
    return null;
  }
}

interface AlertFeature {
  properties: {
    event?: string;
    areaDesc?: string;
    severity?: string;
    status?: string;
    headline?: string;
  };
}

// Format alert data
function formatAlert(feature: AlertFeature): string {
  const props = feature.properties;
  return [
    `Event: ${props.event || "Unknown"}`,
    `Area: ${props.areaDesc || "Unknown"}`,
    `Severity: ${props.severity || "Unknown"}`,
    `Status: ${props.status || "Unknown"}`,
    `Headline: ${props.headline || "No headline"}`,
    "---",
  ].join("\n");
}

interface ForecastPeriod {
  name?: string;
  temperature?: number;
  temperatureUnit?: string;
  windSpeed?: string;
  windDirection?: string;
  shortForecast?: string;
}

interface AlertsResponse {
  features: AlertFeature[];
}

interface PointsResponse {
  properties: {
    forecast?: string;
  };
}

interface ForecastResponse {
  properties: {
    periods: ForecastPeriod[];
  };
}

class MCPServer {
  server: Server;

  // to support multiple simultaneous connections
  transports: { [sessionId: string]: StreamableHTTPServerTransport } = {};

  private toolInterval: NodeJS.Timeout | undefined;
  private getAlertsToolName = "get-alerts";
  private getForecastToolName = "get-forecast";

  constructor(server: Server) {
    this.server = server;
    this.setupTools();
  }

  async handleGetRequest(req: Request, res: Response) {
    // if server does not offer an SSE stream at this endpoint.
    // res.status(405).set('Allow', 'POST').send('Method Not Allowed')

    const sessionId = req.headers["mcp-session-id"] as string | undefined;
    if (!sessionId || !this.transports[sessionId]) {
      res
        .status(400)
        .json(
          this.createErrorResponse("Bad Request: invalid session ID or method.")
        );
      return;
    }

    console.log(`Establishing SSE stream for session ${sessionId}`);
    const transport = this.transports[sessionId];
    await transport.handleRequest(req, res);
    await this.streamMessages(transport);

    return;
  }

  async handlePostRequest(req: Request, res: Response) {
    const sessionId = req.headers[SESSION_ID_HEADER_NAME] as string | undefined;
    let transport: StreamableHTTPServerTransport;

    try {
      // reuse existing transport
      if (sessionId && this.transports[sessionId]) {
        transport = this.transports[sessionId];
        await transport.handleRequest(req, res, req.body);
        return;
      }

      // create new transport
      if (!sessionId && this.isInitializeRequest(req.body)) {
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => randomUUID(),
        });

        await this.server.connect(transport);
        await transport.handleRequest(req, res, req.body);

        // session ID will only be available (if in not Stateless-Mode)
        // after handling the first request
        const sessionId = transport.sessionId;
        if (sessionId) {
          this.transports[sessionId] = transport;
        }

        return;
      }

      res
        .status(400)
        .json(
          this.createErrorResponse("Bad Request: invalid session ID or method.")
        );
      return;
    } catch (error) {
      console.error("Error handling MCP request:", error);
      res.status(500).json(this.createErrorResponse("Internal server error."));
      return;
    }
  }

  async cleanup() {
    this.toolInterval?.close();
    await this.server.close();
  }

  private setupTools() {
    // Define available tools
    const setToolSchema = () =>
      this.server.setRequestHandler(ListToolsRequestSchema, async () => {
        const getAlertsTool = {
          name: this.getAlertsToolName,
          description: "Get weather alerts for a state",
          inputSchema: {
            type: "object",
            properties: {
              state: {
                type: "string",
                description: "Two-letter state code (e.g. CA, NY)",
              },
            },
            required: ["state"],
          },
        };

        const getForecastTool = {
          name: this.getForecastToolName,
          description: "Get weather forecast for a location",
          inputSchema: {
            type: "object",
            properties: {
              latitude: {
                type: "number",
                description: "Latitude of the location",
              },
              longitude: {
                type: "number",
                description: "Longitude of the location",
              },
            },
            required: ["latitude", "longitude"],
          },
        };

        return {
          tools: [getAlertsTool, getForecastTool],
        };
      });

    setToolSchema();

    // set tools dynamically, changing 5 second
    this.toolInterval = setInterval(async () => {
      setToolSchema();
      // to notify client that the tool changed
      Object.values(this.transports).forEach((transport) => {
        const notification: ToolListChangedNotification = {
          method: "notifications/tools/list_changed",
        };
        this.sendNotification(transport, notification);
      });
    }, 5000);

    // handle tool calls
    this.server.setRequestHandler(
      CallToolRequestSchema,
      async (request, extra) => {
        const args = request.params.arguments;
        const toolName = request.params.name;
        console.log("Received request for tool with argument:", toolName, args);

        if (!args) {
          throw new Error("arguments undefined");
        }

        if (!toolName) {
          throw new Error("tool name undefined");
        }

        if (toolName === this.getAlertsToolName) {
          const state = args.state as string;
          const stateCode = state.toUpperCase();
          const alertsUrl = `${NWS_API_BASE}/alerts?area=${stateCode}`;
          const alertsData = await makeNWSRequest<AlertsResponse>(alertsUrl);

          if (!alertsData) {
            return {
              content: [
                {
                  type: "text",
                  text: "Failed to retrieve alerts data",
                },
              ],
            };
          }

          const features = alertsData.features || [];
          if (features.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: `No active alerts for ${stateCode}`,
                },
              ],
            };
          }

          const formattedAlerts = features.map(formatAlert);
          const alertsText = `Active alerts for ${stateCode}:\n\n${formattedAlerts.join(
            "\n"
          )}`;

          return {
            content: [
              {
                type: "text",
                text: alertsText,
              },
            ],
          };
        }

        if (toolName === this.getForecastToolName) {
          const latitude = args.latitude as number;
          const longitude = args.longitude as number;
          // Get grid point data
          const pointsUrl = `${NWS_API_BASE}/points/${latitude.toFixed(
            4
          )},${longitude.toFixed(4)}`;
          const pointsData = await makeNWSRequest<PointsResponse>(pointsUrl);

          if (!pointsData) {
            return {
              content: [
                {
                  type: "text",
                  text: `Failed to retrieve grid point data for coordinates: ${latitude}, ${longitude}. This location may not be supported by the NWS API (only US locations are supported).`,
                },
              ],
            };
          }

          const forecastUrl = pointsData.properties?.forecast;
          if (!forecastUrl) {
            return {
              content: [
                {
                  type: "text",
                  text: "Failed to get forecast URL from grid point data",
                },
              ],
            };
          }

          // Get forecast data
          const forecastData = await makeNWSRequest<ForecastResponse>(
            forecastUrl
          );
          if (!forecastData) {
            return {
              content: [
                {
                  type: "text",
                  text: "Failed to retrieve forecast data",
                },
              ],
            };
          }

          const periods = forecastData.properties?.periods || [];
          if (periods.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: "No forecast periods available",
                },
              ],
            };
          }

          // Format forecast periods
          const formattedForecast = periods.map((period: ForecastPeriod) =>
            [
              `${period.name || "Unknown"}:`,
              `Temperature: ${period.temperature || "Unknown"}°${
                period.temperatureUnit || "F"
              }`,
              `Wind: ${period.windSpeed || "Unknown"} ${
                period.windDirection || ""
              }`,
              `${period.shortForecast || "No forecast available"}`,
              "---",
            ].join("\n")
          );

          const forecastText = `Forecast for ${latitude}, ${longitude}:\n\n${formattedForecast.join(
            "\n"
          )}`;

          return {
            content: [
              {
                type: "text",
                text: forecastText,
              },
            ],
          };
        }

        throw new Error("Tool not found");
      }
    );
  }

  // send message streaming message every second
  private async streamMessages(transport: StreamableHTTPServerTransport) {
    try {
      // based on LoggingMessageNotificationSchema to trigger setNotificationHandler on client
      const message: LoggingMessageNotification = {
        method: "notifications/message",
        params: { level: "info", data: "SSE Connection established" },
      };

      this.sendNotification(transport, message);

      let messageCount = 0;

      const interval = setInterval(async () => {
        messageCount++;

        const data = `Message ${messageCount} at ${new Date().toISOString()}`;

        const message: LoggingMessageNotification = {
          method: "notifications/message",
          params: { level: "info", data: data },
        };

        try {
          this.sendNotification(transport, message);

          if (messageCount === 2) {
            clearInterval(interval);

            const message: LoggingMessageNotification = {
              method: "notifications/message",
              params: { level: "info", data: "Streaming complete!" },
            };

            this.sendNotification(transport, message);
          }
        } catch (error) {
          console.error("Error sending message:", error);
          clearInterval(interval);
        }
      }, 1000);
    } catch (error) {
      console.error("Error sending message:", error);
    }
  }

  private async sendNotification(
    transport: StreamableHTTPServerTransport,
    notification: Notification
  ) {
    const rpcNotificaiton: JSONRPCNotification = {
      ...notification,
      jsonrpc: JSON_RPC,
    };
    await transport.send(rpcNotificaiton);
  }

  private createErrorResponse(message: string): JSONRPCError {
    return {
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message: message,
      },
      id: randomUUID(),
    };
  }

  private isInitializeRequest(body: any): boolean {
    const isInitial = (data: any) => {
      const result = InitializeRequestSchema.safeParse(data);
      return result.success;
    };
    if (Array.isArray(body)) {
      return body.some((request) => isInitial(request));
    }
    return isInitial(body);
  }
}

const server = new MCPServer(
  new Server(
    { name: "mcp-streamable-http-server", version: "1.0.0" },
    {
      capabilities: {
        tools: {},
        logging: {},
      },
    }
  )
);

const app = express();
app.use(express.json());
app.use(
  cors({
    credentials: true,
    exposedHeaders: "*",
  })
);

const router = express.Router();

// single endpoint for the client to send messages to
const MCP_ENDPOINT = "/mcp";

router.post(MCP_ENDPOINT, async (req: Request, res: Response) => {
  await server.handlePostRequest(req, res);
});

router.get(MCP_ENDPOINT, async (req: Request, res: Response) => {
  await server.handleGetRequest(req, res);
});

app.use("/", router);

export default app;
