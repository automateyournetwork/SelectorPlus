#!/usr/bin/env node
console.log("🚀 Server starting...");

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import * as files from "./operations/files.js";
import { VERSION } from "./common/version.js";

// ✅ Manually Define Both Request Schemas
const ListToolsRequestSchema = z.object({
  jsonrpc: z.literal("2.0"),
  method: z.literal("list_tools"),
  params: z.object({}).optional(),
  id: z.union([z.string(), z.number()]),
});

// Define our own CallToolRequestSchema since the imported one seems to have issues
const CallToolRequestSchema = z.object({
  jsonrpc: z.literal("2.0"),
  method: z.literal("call_tool"),
  params: z.object({
    name: z.string(),
    arguments: z.record(z.any()).optional(),
  }),
  id: z.union([z.string(), z.number()]),
});

console.log("✅ Schema definitions complete");

// ✅ Server Initialization without the debug option that caused errors
const server = new Server(
  {
    name: "github-mcp-server",
    version: VERSION,
  },
  {
    capabilities: {
      tools: {},
    }
    // Removed debug: true that caused TypeScript error
  }
);

console.log("✅ Server initialized");

// ✅ Register ListToolsRequestSchema
console.log("🔧 Registering list_tools handler...");
server.setRequestHandler(ListToolsRequestSchema, async (request) => {
  console.log("✅ list_tools handler is RUNNING!");
  console.log("📩 Received list_tools request:", JSON.stringify(request, null, 2));
  return {
    tools: [
      {
        name: "create_or_update_file",
        description: "Create or update a single file in a GitHub repository",
        inputSchema: zodToJsonSchema(files.CreateOrUpdateFileSchema),
      },
    ],
  };
});
console.log("✅ Successfully registered list_tools handler.");

// ✅ Register CallToolRequestSchema
console.log("🔧 Registering call_tool handler...");
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  console.log("📩 Received call_tool request:", JSON.stringify(request, null, 2));

  try {
    if (!request.params.arguments) {
      throw new Error("❌ Arguments are required");
    }

    switch (request.params.name) {
      case "create_or_update_file": {
        console.log("✏️ Processing create_or_update_file...");
        const args = files.CreateOrUpdateFileSchema.parse(request.params.arguments);

        try {
          const result = await files.createOrUpdateFile(
            args.owner,
            args.repo,
            args.path,
            args.content,
            args.message,
            args.branch,
            args.sha
          );

          console.log("✅ File created/updated successfully:", JSON.stringify(result, null, 2));

          return {
            content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
          };
        } catch (error) {
          console.error("❌ Error creating/updating file:", error);
          throw error;
        }
      }

      default:
        console.error(`❌ Unknown tool requested: ${request.params.name}`);
        throw new Error(`Unknown tool: ${request.params.name}`);
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error("❌ Invalid input error:", JSON.stringify(error.errors));
      throw new Error(`Invalid input: ${JSON.stringify(error.errors)}`);
    }
    throw error;
  }
});
console.log("✅ Successfully registered call_tool handler.");

// ✅ Run Server
async function runServer() {
  console.log("🚀 Starting GitHub MCP Server...");
  
  const transport = new StdioServerTransport();
  
  // Removed onMessage handler that caused TypeScript error
  
  await server.connect(transport);
  console.error("✅ GitHub MCP Server running on stdio");
  
  // Add test calls
  process.stderr.write("✅ Server ready to process requests\n");
}

runServer().catch((error) => {
  console.error("❌ Fatal error in main():", error);
  process.exit(1);
});