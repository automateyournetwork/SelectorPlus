Selector AI FastMCP

This repository provides a full implementation of the Model Context Protocol (MCP) for Selector AI. It includes a streaming-capable server and a Docker-based interactive client that communicates via stdin/stdout.

✨ Features

✅ Server

FastMCP-compatible and built on Python

Real-time SSE streaming support

Interactive AI chat with Selector AI

Minimal boilerplate

Built-in health check for container orchestration

Request/response logging and retries

✅ Client

Python client spawns server via Docker

Supports both CLI and programmatic access

Reads/writes via stdin and stdout

Environment variable configuration using .env

🚀 Quick Start

Prerequisites

Python 3.8+

Docker

A Selector AI API Key

Selector API URL

⚙️ Installation

Clone the Repository

git clone https://github.com/automateyournetwork/selector-mcp-server

cd selector-ai-mcp

Install Python Dependencies

pip install -r requirements.txt

Set Environment Variables Create a .env file:

SELECTOR_URL=https://your-selector-api-url

SELECTOR_AI_API_KEY=your-api-key

🐳 Dockerfile

The server runs in a lightweight container using the following Dockerfile:

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "mcp_server.py"]

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s \
  CMD python -c "import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect('/tmp/mcp.sock'); s.send(b'{\"tool_name\": \"ready\"}\\n'); data = s.recv(1024); s.close(); import json; result = json.loads(data); exit(0 if result.get('status') == 'ready' else 1)" || exit 1

Build the Docker Image

docker build -t selector-mcp .

🧠 Using the Client

Start the Client

This will spawn the Docker container and open an interactive shell.

python mcp_client.py

Example CLI Session

You> What is AIOps?

Selector> AIOps refers to the application of AI to IT operations...

Programmatic Access

from selector_client import call_tool, spawn_server

proc = spawn_server()

call_tool(proc, "ready")

response = call_tool(proc, "ask_selector", {"content": "What is AIOps?"})

print(response)

🖥️ Using with Claude Desktop

If you're integrating with Claude Desktop, you can run this server and expose a socket or HTTP endpoint locally:

Run the server using Docker or natively:

python mcp_server.py

Connect to the socket or HTTP endpoint from Claude Desktop's external tool configuration.

Ensure your messages match the format:
```json
{
  "method": "tools/call",
  "tool_name": "ask_selector",
  "content": "What can you tell me about device S6?"
}
```

Claude Desktop will receive the AI's structured response via stdout.

🛠️ Build Your Own Container

To customize this setup:

Fork or clone this repo

Modify the selector_fastmcp_server.py to integrate your preferred model or routing logic

Rebuild the Docker image:

docker build -t my-custom-mcp .

Update the client to spawn my-custom-mcp instead:

"docker", "run", "-i", "--rm", "my-custom-mcp"

📁 Project Structure

selector-ai-mcp/

```console
├── selector_fastmcp_server.py     # Server: MCP + Selector AI integration
├── selector_client.py             # Client: Docker + stdin/stdout CLI
├── Dockerfile                     # Container config
├── requirements.txt               # Python deps
├── .env                           # Environment secrets
└── README.md                      # You are here
```

✅ Requirements

Dependencies in requirements.txt:

requests

python-dotenv

📜 License

Apache License 2.0