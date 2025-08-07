import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

// NWS API base URL
const NWS_BASE_URL = "https://api.weather.gov";

// Helper function to make HTTP requests
async function makeHttpRequest(url: string): Promise<any> {
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "mcp-nws-server/1.0.0 (contact@example.com)",
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(
      `Request failed: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  }
}

function createMcpServer() {
  // Create the server
  const server = new McpServer({
    name: "nws-weather-server",
    version: "1.0.0",
  });
  server.registerTool(
    "get_weather_alerts",
    {
      title: "get_weather_alerts",
      description:
        "Get active weather alerts for a specific area using coordinates or zone ID",
      inputSchema: {
        area: z.string({
          description:
            "Area identifier - can be state abbreviation (e.g., 'CA'), zone ID (e.g., 'CAZ006'), or coordinates as 'lat,lon' (e.g., '37.7749,-122.4194')",
        }),
      },
    },
    async ({ area }) => {
      let alertsUrl: string;

      // Determine the type of area identifier and construct appropriate URL
      if (area.includes(",")) {
        // Coordinates format: lat,lon
        const [lat, lon] = area.split(",");
        alertsUrl = `${NWS_BASE_URL}/alerts/active?point=${lat},${lon}`;
      } else if (area.length === 2) {
        // State abbreviation
        alertsUrl = `${NWS_BASE_URL}/alerts/active?area=${area.toUpperCase()}`;
      } else {
        // Assume it's a zone ID
        alertsUrl = `${NWS_BASE_URL}/alerts/active?zone=${area.toUpperCase()}`;
      }

      const alertsData = await makeHttpRequest(alertsUrl);

      // Format the alerts for better readability
      const formattedAlerts =
        alertsData.features?.map((alert: any) => ({
          id: alert.id,
          type: alert.properties.event,
          headline: alert.properties.headline,
          description: alert.properties.description,
          severity: alert.properties.severity,
          urgency: alert.properties.urgency,
          areas: alert.properties.areaDesc,
          effective: alert.properties.effective,
          expires: alert.properties.expires,
          senderName: alert.properties.senderName,
        })) || [];

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                alertsCount: formattedAlerts.length,
                alerts: formattedAlerts,
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );
  server.registerTool(
    "get_weather_forecast",
    {
      description:
        "Get weather forecast for a specific location using latitude and longitude coordinates",
      inputSchema: {
        latitude: z.number({
          description: "Latitude coordinate (e.g., 37.7749)",
        }),
        longitude: z.number({
          description: "Longitude coordinate (e.g., -122.4194)",
        }),
        period: z.enum(["current", "hourly", "daily"], {}).optional(),
      },
    },
    async ({ latitude, longitude, period }) => {
      // First, get the grid point data
      const pointUrl = `${NWS_BASE_URL}/points/${latitude},${longitude}`;
      const pointData = await makeHttpRequest(pointUrl);

      let forecastUrl: string;

      switch (period) {
        case "hourly":
          forecastUrl = pointData.properties.forecastHourly;
          break;
        case "current":
          // For current conditions, we'll use the gridpoint data
          forecastUrl = pointData.properties.forecastGridData;
          break;
        case "daily":
        default:
          forecastUrl = pointData.properties.forecast;
          break;
      }

      const forecastData = await makeHttpRequest(forecastUrl);

      if (period === "current") {
        // For current conditions, extract relevant current data
        const currentConditions = {
          location: `${pointData.properties.relativeLocation.properties.city}, ${pointData.properties.relativeLocation.properties.state}`,
          gridId: pointData.properties.gridId,
          gridX: pointData.properties.gridX,
          gridY: pointData.properties.gridY,
          temperature: forecastData.properties.temperature?.values?.[0],
          humidity: forecastData.properties.relativeHumidity?.values?.[0],
          windSpeed: forecastData.properties.windSpeed?.values?.[0],
          windDirection: forecastData.properties.windDirection?.values?.[0],
        };

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(currentConditions, null, 2),
            },
          ],
        };
      } else {
        // Format forecast periods
        const formattedForecast = {
          location: `${pointData.properties.relativeLocation.properties.city}, ${pointData.properties.relativeLocation.properties.state}`,
          updated: forecastData.properties.updated,
          periods:
            forecastData.properties.periods
              ?.slice(0, period === "hourly" ? 24 : 7)
              .map((p: any) => ({
                name: p.name,
                startTime: p.startTime,
                endTime: p.endTime,
                temperature: p.temperature,
                temperatureUnit: p.temperatureUnit,
                windSpeed: p.windSpeed,
                windDirection: p.windDirection,
                shortForecast: p.shortForecast,
                detailedForecast: p.detailedForecast,
              })) || [],
        };

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(formattedForecast, null, 2),
            },
          ],
        };
      }
    }
  );

  return server;
}

export default createMcpServer;
