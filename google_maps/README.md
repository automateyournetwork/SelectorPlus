# Google Maps MCP Server

MCP Server for the Google Maps API.

## Tools

1. `maps_geocode`
   - Convert address to coordinates
   - Input: `address` (string)
   - Returns: location, formatted_address, place_id

2. `maps_reverse_geocode`
   - Convert coordinates to address
   - Inputs:
     - `latitude` (number)
     - `longitude` (number)
   - Returns: formatted_address, place_id, address_components

6. `maps_elevation`
   - Get elevation data for locations
   - Input: `locations` (array of {latitude, longitude})
   - Returns: elevation data for each point

## Setup

### API Key
Get a Google Maps API key by following the instructions [here](https://developers.google.com/maps/documentation/javascript/get-api-key#create-api-keys).

### Usage with Claude Desktop

Add the following to your `claude_desktop_config.json`:

#### Docker

```json
{
  "mcpServers": {
    "google-maps": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "GOOGLE_MAPS_API_KEY",
        "mcp/google-maps"
      ],
      "env": {
        "GOOGLE_MAPS_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

### NPX

```json
{
  "mcpServers": {
    "google-maps": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-google-maps"
      ],
      "env": {
        "GOOGLE_MAPS_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

## Build

Docker build:

```bash
docker build -t mcp/google-maps -f src/google-maps/Dockerfile .
```

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.
