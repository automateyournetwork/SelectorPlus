#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

import * as drawings from './src/operations/drawings.js';
import * as exportOps from './src/operations/export.js';
import {
  ExcalidrawError,
  ExcalidrawValidationError,
  ExcalidrawResourceNotFoundError,
  ExcalidrawAuthenticationError,
  ExcalidrawPermissionError,
  ExcalidrawRateLimitError,
  ExcalidrawConflictError,
  isExcalidrawError,
} from './src/common/errors.js';

const server = new Server(
  {
    name: "excalidraw-mcp-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

function formatExcalidrawError(error: ExcalidrawError): string {
  let message = `Excalidraw API Error: ${error.message}`;
  
  if (error instanceof ExcalidrawValidationError) {
    message = `Validation Error: ${error.message}`;
    if (error.response) {
      message += `\nDetails: ${JSON.stringify(error.response)}`;
    }
  } else if (error instanceof ExcalidrawResourceNotFoundError) {
    message = `Not Found: ${error.message}`;
  } else if (error instanceof ExcalidrawAuthenticationError) {
    message = `Authentication Failed: ${error.message}`;
  } else if (error instanceof ExcalidrawPermissionError) {
    message = `Permission Denied: ${error.message}`;
  } else if (error instanceof ExcalidrawRateLimitError) {
    message = `Rate Limit Exceeded: ${error.message}\nResets at: ${error.resetAt.toISOString()}`;
  } else if (error instanceof ExcalidrawConflictError) {
    message = `Conflict: ${error.message}`;
  }

  return message;
}

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "create_drawing",
        description: "Create a new Excalidraw drawing",
        inputSchema: zodToJsonSchema(drawings.CreateDrawingSchema),
      },
      {
        name: "get_drawing",
        description: "Get an Excalidraw drawing by ID",
        inputSchema: zodToJsonSchema(drawings.GetDrawingSchema),
      },
      {
        name: "update_drawing",
        description: "Update an Excalidraw drawing by ID",
        inputSchema: zodToJsonSchema(drawings.UpdateDrawingSchema),
      },
      {
        name: "delete_drawing",
        description: "Delete an Excalidraw drawing by ID",
        inputSchema: zodToJsonSchema(drawings.DeleteDrawingSchema),
      },
      {
        name: "list_drawings",
        description: "List all Excalidraw drawings",
        inputSchema: zodToJsonSchema(drawings.ListDrawingsSchema),
      },
      {
        name: "export_to_json",
        description: "Export an Excalidraw drawing to JSON",
        inputSchema: zodToJsonSchema(exportOps.ExportToJsonSchema),
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    if (!request.params.arguments) {
      throw new Error("Arguments are required");
    }

    switch (request.params.name) {
      case "create_drawing": {
        const args = drawings.CreateDrawingSchema.parse(request.params.arguments);
        const result = await drawings.createDrawing(args.name, args.content);
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      }

      case "get_drawing": {
        const args = drawings.GetDrawingSchema.parse(request.params.arguments);
        const result = await drawings.getDrawing(args.id);
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      }

      case "update_drawing": {
        const args = drawings.UpdateDrawingSchema.parse(request.params.arguments);
        const result = await drawings.updateDrawing(args.id, args.content);
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      }

      case "delete_drawing": {
        const args = drawings.DeleteDrawingSchema.parse(request.params.arguments);
        await drawings.deleteDrawing(args.id);
        return {
          content: [{ type: "text", text: JSON.stringify({ success: true }, null, 2) }],
        };
      }

      case "list_drawings": {
        const args = drawings.ListDrawingsSchema.parse(request.params.arguments);
        const result = await drawings.listDrawings(args.page, args.perPage);
        return {
          content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        };
      }

      case "export_to_json": {
        const args = exportOps.ExportToJsonSchema.parse(request.params.arguments);
        const result = await exportOps.exportToJson(args.id);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      default:
        throw new Error(`Unknown tool: ${request.params.name}`);
    }
  } catch (error) {
    console.error("Error handling request:", error);
    
    if (isExcalidrawError(error)) {
      return {
        error: formatExcalidrawError(error),
      };
    }
    
    return {
      error: `Error: ${(error as Error).message}`,
    };
  }
});

async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

runServer().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
