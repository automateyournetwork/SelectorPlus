# Excalidraw MCP Server

This is a Model Context Protocol (MCP) server for Excalidraw, providing API functionality for operating on Excalidraw drawings.

## Features

- Create, read, update, and delete Excalidraw drawings
- Export drawings to SVG, PNG, and JSON formats
- Simple file-based storage system

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/excalidraw-mcp.git
cd excalidraw-mcp

# Install dependencies
npm install

# Build the project
npm run build
```

## Usage

### Starting the Server

```bash
npm start
```

### API Endpoints

The server provides the following tools:

#### Drawing Management

- `create_drawing`: Create a new Excalidraw drawing

#### Export Operations

- `export_to_json`: Export an Excalidraw drawing to JSON

## Development

### Project Structure

```
excalidraw-mcp/
├── src/
│   ├── common/
│   │   └── errors.ts
│   └── operations/
│       ├── drawings.ts
│       └── export.ts
├── index.ts
├── package.json
├── tsconfig.json
└── README.md
```

### Building

```bash
npm run build
```

### Running in Development Mode

```bash
npm run dev
```

## License

MIT 