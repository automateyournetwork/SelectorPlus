#!/usr/bin/env node
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  // Create a transport connected to the server process
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
  });

  // Create a client
  const client = new Client(
    {
      name: 'excalidraw-mcp-test',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  try {
    // Connect to the server
    await client.connect(transport);
    console.log('Connected to server');

    // List available tools
    const tools = await client.listTools();
    console.log('Available tools:', tools);

    // Create a drawing
    const createResult = await client.callTool({
      name: 'create_drawing',
      arguments: {
        name: 'Test Drawing',
        content: JSON.stringify({
          type: 'excalidraw',
          version: 2,
          source: 'excalidraw-mcp-test',
          elements: [
            {
              id: 'rectangle1',
              type: 'rectangle',
              x: 100,
              y: 100,
              width: 200,
              height: 100,
              angle: 0,
              strokeColor: '#000000',
              backgroundColor: '#ffffff',
              fillStyle: 'solid',
              strokeWidth: 1,
              strokeStyle: 'solid',
              roughness: 1,
              opacity: 100,
              groupIds: [],
              strokeSharpness: 'sharp',
              seed: 123456,
              version: 1,
              versionNonce: 1,
              isDeleted: false,
              boundElementIds: null,
            },
          ],
          appState: {
            viewBackgroundColor: '#ffffff',
          },
        }),
      },
    });
    console.log('Created drawing:', createResult);

    // Get the drawing ID from the result
    const drawingId = JSON.parse(createResult.content[0].text).id;

    // Get the drawing
    const getResult = await client.callTool({
      name: 'get_drawing',
      arguments: {
        id: drawingId,
      },
    });
    console.log('Retrieved drawing:', getResult);

    // Export the drawing to SVG
    const svgResult = await client.callTool({
      name: 'export_to_svg',
      arguments: {
        id: drawingId,
      },
    });
    console.log('SVG export:', svgResult);

    // Export the drawing to PNG
    const pngResult = await client.callTool({
      name: 'export_to_png',
      arguments: {
        id: drawingId,
        quality: 0.9,
        scale: 2,
        exportWithDarkMode: true,
        exportBackground: true,
      },
    });
    console.log('PNG export:', pngResult);

    // List all drawings
    const listResult = await client.callTool({
      name: 'list_drawings',
      arguments: {
        page: 1,
        perPage: 10,
      },
    });
    console.log('List of drawings:', listResult);

    // Update the drawing
    const updateResult = await client.callTool({
      name: 'update_drawing',
      arguments: {
        id: drawingId,
        content: JSON.stringify({
          type: 'excalidraw',
          version: 2,
          source: 'excalidraw-mcp-test',
          elements: [
            {
              id: 'rectangle1',
              type: 'rectangle',
              x: 100,
              y: 100,
              width: 200,
              height: 100,
              angle: 0,
              strokeColor: '#ff0000', // Changed to red
              backgroundColor: '#ffffff',
              fillStyle: 'solid',
              strokeWidth: 2, // Increased width
              strokeStyle: 'solid',
              roughness: 1,
              opacity: 100,
              groupIds: [],
              strokeSharpness: 'sharp',
              seed: 123456,
              version: 2,
              versionNonce: 2,
              isDeleted: false,
              boundElementIds: null,
            },
          ],
          appState: {
            viewBackgroundColor: '#ffffff',
          },
        }),
      },
    });
    console.log('Updated drawing:', updateResult);

    // Delete the drawing
    const deleteResult = await client.callTool({
      name: 'delete_drawing',
      arguments: {
        id: drawingId,
      },
    });
    console.log('Deleted drawing:', deleteResult);

    console.log('All tests passed!');
  } catch (error) {
    console.error('Error:', error);
  } finally {
    // Close the client connection
    await client.close();
  }
}

main().catch((error) => {
  console.error('Test error:', error);
  process.exit(1);
}); 