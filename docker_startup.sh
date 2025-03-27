#!/bin/bash

# Load environment variables from .env
if [ -f .env ]; then
  source .env
  echo "Loaded environment variables from .env"
else
  echo ".env file not found. Using default values."
fi

# Build containers
echo "Building github-mcp image..."
docker build -t github-mcp ./github
if [ $? -ne 0 ]; then echo "Error building github-mcp image."; exit 1; fi
echo "github-mcp image built successfully."

echo "Building google-maps-mcp image..."
docker build -t google-maps-mcp ./google_maps
if [ $? -ne 0 ]; then echo "Error building google-maps-mcp image."; exit 1; fi
echo "google-maps-mcp image built successfully."

echo "Building sequentialthinking-mcp image..."
docker build -t sequentialthinking-mcp ./sequentialthinking
if [ $? -ne 0 ]; then echo "Error building sequentialthinking-mcp image."; exit 1; fi
echo "sequentialthinking-mcp image built successfully."

echo "Building slack-mcp image..."
docker build -t slack-mcp ./slack
if [ $? -ne 0 ]; then echo "Error building slack-mcp image."; exit 1; fi
echo "slack-mcp image built successfully."

echo "Building selector-mcp image..."
docker build -t selector-mcp ./selector
if [ $? -ne 0 ]; then echo "Error building selector-mcp image."; exit 1; fi
echo "selector-mcp image built successfully."

echo "Building excalidraw-mcp image..."
docker build -t excalidraw-mcp ./excalidraw
if [ $? -ne 0 ]; then echo "Error building excalidraw-mcp image."; exit 1; fi
echo "excalidraw-mcp image built successfully."

echo "Building brave-search-mcp image..."
docker build -t brave-search-mcp ./brave-search
if [ $? -ne 0 ]; then echo "Error building brave-search-mcp image."; exit 1; fi
echo "brave-search-mcp image built successfully."

# Build langgraph container
echo "Building langgraph container..."
docker build -t langgraph-selectorplus -f ./selectorplus/Dockerfile ./selectorplus
if [ $? -ne 0 ]; then echo "Error building langgraph-selectorplus image."; exit 1; fi
echo "langgraph-selectorplus image built successfully."

# Use environment variables in docker run commands
echo "Starting github-mcp container..."
docker run -dit --name github-mcp -e GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_GITHUB_TOKEN}" github-mcp
echo "github-mcp container started."

echo "Starting google-maps-mcp container..."
docker run -dit --name google-maps-mcp -e GOOGLE_MAPS_API_KEY="${GOOGLE_MAPS_API_KEY:-YOUR_GOOGLE_MAPS_API_KEY}" google-maps-mcp
echo "google-maps-mcp container started."

echo "Starting sequentialthinking-mcp container..."
docker run -dit --name sequentialthinking-mcp sequentialthinking-mcp
echo "sequentialthinking-mcp container started."

echo "Starting slack-mcp container..."
docker run -dit --name slack-mcp -e SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-YOUR_SLACK_BOT_TOKEN}" -e SLACK_TEAM_ID="${SLACK_TEAM_ID:-YOUR_SLACK_TEAM_ID}" slack-mcp
echo "slack-mcp container started."

echo "Starting selector-mcp container..."
docker run -d --name selector-mcp -e SELECTOR_URL="${SELECTOR_URL:-YOUR_SELECTOR_URL}" -e SELECTOR_AI_API_KEY="${SELECTOR_AI_API_KEY:-YOUR_SELECTOR_AI_API_KEY}" selector-mcp python3 mcp_server.py --restart unless-stopped
echo "selector-mcp container started."

echo "Starting excalidraw-mcp container..."
docker run -dit --name excalidraw-mcp excalidraw-mcp
echo "excalidraw-mcp container started."

# Ensure the /projects directory exists on the host
mkdir -p /projects

# Filesystem MCP Server
echo "Starting filesystem-mcp container..."
docker run -dit --name filesystem-mcp -v "${FILESYSTEM_PATH:-/projects}:/projects" mcp/filesystem /projects
echo "filesystem-mcp container started."

echo "Starting brave-search-mcp container..."
docker run -dit --name brave-search-mcp -e BRAVE_API_KEY="${BRAVE_API_KEY}" brave-search-mcp
echo "brave-search-mcp container started."

# Start langgraph container
echo "Starting langgraph-selectorplus container..."
docker run -p 2024:2024 -dit --name langgraph-selectorplus langgraph-selectorplus
echo "langgraph-selectorplus container started."

echo "All containers started."