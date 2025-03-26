#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import fetch from "node-fetch";
import { z } from "zod";

// Response interfaces (using zod for stronger type safety)
const GoogleMapsResponseSchema = z.object({
  status: z.string(),
  error_message: z.string().optional(),
});

const GeocodeResponseSchema = GoogleMapsResponseSchema.extend({
  results: z.array(z.object({
    place_id: z.string(),
    formatted_address: z.string(),
    geometry: z.object({
      location: z.object({
        lat: z.number(),
        lng: z.number(),
      }),
    }),
    address_components: z.array(z.object({
      long_name: z.string(),
      short_name: z.string(),
      types: z.array(z.string()),
    })),
  })),
});

const ElevationResponseSchema = GoogleMapsResponseSchema.extend({
  results: z.array(z.object({
    elevation: z.number(),
    location: z.object({
      lat: z.number(),
      lng: z.number(),
    }),
    resolution: z.number(),
  })),
});

function getApiKey(): string {
  const apiKey = process.env.GOOGLE_MAPS_API_KEY;
  if (!apiKey) {
    console.error("GOOGLE_MAPS_API_KEY environment variable is not set");
    process.exit(1);
  }
  return apiKey;
}

const GOOGLE_MAPS_API_KEY = getApiKey();

// Tool definitions (using const assertions for type safety)
const GEOCODE_TOOL = {
  name: "maps_geocode",
  description: "Convert an address into geographic coordinates",
  inputSchema: {
    type: "object",
    properties: {
      address: {
        type: "string",
        description: "The address to geocode",
      },
    },
    required: ["address"],
  },
} as const;

const REVERSE_GEOCODE_TOOL = {
  name: "maps_reverse_geocode",
  description: "Convert coordinates into an address",
  inputSchema: {
    type: "object",
    properties: {
      latitude: {
        type: "number",
        description: "Latitude coordinate",
      },
      longitude: {
        type: "number",
        description: "Longitude coordinate",
      },
    },
    required: ["latitude", "longitude"],
  },
} as const;

const ELEVATION_TOOL = {
  name: "maps_elevation",
  description: "Get elevation data for locations on the earth",
  inputSchema: {
    type: "object",
    properties: {
      locations: {
        type: "array",
        items: {
          type: "object",
          properties: {
            latitude: { type: "number" },
            longitude: { type: "number" },
          },
          required: ["latitude", "longitude"],
        },
        description: "Array of locations to get elevation for",
      },
    },
    required: ["locations"],
  },
} as const;

const MAPS_TOOLS = [
  GEOCODE_TOOL,
  REVERSE_GEOCODE_TOOL,
  ELEVATION_TOOL,
] as const;

console.log("✅ Schema definitions complete");

// API handlers (using zod to validate responses)
async function handleGeocode(address: string) {
  const url = new URL("https://maps.googleapis.com/maps/api/geocode/json");
  url.searchParams.append("address", address);
  url.searchParams.append("key", GOOGLE_MAPS_API_KEY);

  const response = await fetch(url.toString());
  const data = GeocodeResponseSchema.parse(await response.json()); // Zod parse

  if (data.status !== "OK") {
    return {
      content: [{
        type: "text",
        text: `Geocoding failed: ${data.error_message || data.status}`,
      }],
      isError: true,
    };
  }

  return {
    content: [{
      type: "text",
      text: JSON.stringify({
        location: data.results[0].geometry.location,
        formatted_address: data.results[0].formatted_address,
        place_id: data.results[0].place_id,
      }, null, 2),
    }],
    isError: false,
  };
}

async function handleReverseGeocode(latitude: number, longitude: number) {
  const url = new URL("https://maps.googleapis.com/maps/api/geocode/json");
  url.searchParams.append("latlng", `<span class="math-inline">\{latitude\},</span>{longitude}`);
  url.searchParams.append("key", GOOGLE_MAPS_API_KEY);

  const response = await fetch(url.toString());
  const data = GeocodeResponseSchema.parse(await response.json());

  if (data.status !== "OK") {
    return {
      content: [{
        type: "text",
        text: `Reverse geocoding failed: ${data.error_message || data.status}`,
      }],
      isError: true,
    };
  }

  return {
    content: [{
      type: "text",
      text: JSON.stringify({
        formatted_address: data.results[0].formatted_address,
        place_id: data.results[0].place_id,
        address_components: data.results[0].address_components,
      }, null, 2),
    }],
    isError: false,
  };
}

async function handleElevation(locations: Array<{ latitude: number; longitude: number }>) {
  const url = new URL("https://maps.googleapis.com/maps/api/elevation/json");
  const locationString = locations
    .map((loc) => `${loc.latitude},${loc.longitude}`)
    .join("|");
  url.searchParams.append("locations", locationString);
  url.searchParams.append("key", GOOGLE_MAPS_API_KEY);

  const response = await fetch(url.toString());
  const data = ElevationResponseSchema.parse(await response.json());

  if (data.status !== "OK") {
    return {
      content: [{
        type: "text",
        text: `Elevation request failed: ${data.error_message || data.status}`,
      }],
      isError: true,
    };
  }

  return {
    content: [{
      type: "text",
      text: JSON.stringify({
        results: data.results.map((result) => ({
          elevation: result.elevation,
          location: result.location,
          resolution: result.resolution,
        })),
      }, null, 2),
    }],
    isError: false,
  };
}

// Server setup
// Server setup
const server = new Server(
  {
    name: "mcp-server/google-maps",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

console.log("✅ Server initialized");

console.log("🔧 Registering list_tools handler...");


// Modify the server initialization
// Create the server
const mcpServer = new Server(
  {
    name: "mcp-server/google-maps",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

console.log("✅ Server initialized");

console.log("🔧 Registering list_tools handler...");

// Set up request handlers
mcpServer.setRequestHandler(ListToolsRequestSchema, async (request) => {
  console.error('🔍 FULL REQUEST DETAILS:', JSON.stringify(request, null, 2));
  console.error('🔍 Request Method:', request.method);
  console.error('🔍 Request Keys:', Object.keys(request));
  console.error(`Received method: '${request.method}'`);

  try {
    const allowedMethods = ["list_tools", "tools/list", "tools/discover"];
    if (!allowedMethods.includes(request.method)) {
      console.error(`❌ Unsupported method: ${request.method}`);
      throw new Error(`Unsupported method: ${request.method}`);
    }

    const toolResponse = {
      tools: MAPS_TOOLS.map((tool) => ({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
      })),
    };
    
    console.error('✅ Tool Response:', JSON.stringify(toolResponse, null, 2));
    return { tools: toolResponse.tools };
  } catch (error) {
    console.error("❌ Error in tool discovery:", error);
    return {
      tools: [], 
      error: error instanceof Error ? error.message : String(error)
    };
  }
});

console.log("🔧 Registering call_tool handler...");

mcpServer.setRequestHandler(CallToolRequestSchema, async (request) => {
  console.log("📩 Received call_tool request:", JSON.stringify(request, null, 2));
  console.log("Request Arguments: ", JSON.stringify(request.params.arguments));
  try {
    switch (request.params.name) {
      case "maps_geocode": {
        const { address } = request.params.arguments as { address: string };
        const result = await handleGeocode(address);
        return {
          content: result.content,
          _meta: {}, // Ensure _meta is always provided
        };
      }
      case "maps_reverse_geocode": {
        const { latitude, longitude } = request.params.arguments as {
          latitude: number;
          longitude: number;
        };
        const result = await handleReverseGeocode(latitude, longitude);
        return {
          content: result.content,
          _meta: {},
        };
      }
      case "maps_elevation": {
        const { locations } = request.params.arguments as {
          locations: Array<{ latitude: number; longitude: number }>;
        };
        const result = await handleElevation(locations);
        return {
          content: result.content,
          _meta: {},
        };
      }
      default:
        return {
          content: [{
            type: "text",
            text: `Unknown tool: ${request.params.name}`
          }],
          isError: true,
          _meta: {},
        };
    }
  } catch (error) {
    return {
      content: [{
        type: "text",
        text: `Error: ${error instanceof z.ZodError ? JSON.stringify(error.errors) : error instanceof Error ? error.message : String(error)}`,
      }],
      isError: true,
      _meta: {},
    };
  }
});

console.log("✅ Successfully registered call_tool handler.");

async function runServer() {
  console.log("🚀 Starting Google Maps MCP Server...");
  const transport = new StdioServerTransport();
  await mcpServer.connect(transport);
  console.log("✅ Google Maps MCP Server running on stdio");
  process.stderr.write("✅ Server ready to process requests\n");
}

runServer().catch((error) => {
  console.error("Fatal error running server:", error);
  process.exit(1);
});