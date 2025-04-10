import os
import sys
import json
import time
import logging
import threading
import asyncio
import requests
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SelectorMCPServer")

# Environment
SELECTOR_URL = os.getenv("SELECTOR_URL")
SELECTOR_AI_API_KEY = os.getenv("SELECTOR_AI_API_KEY")

if not SELECTOR_URL or not SELECTOR_AI_API_KEY:
    logger.critical("Missing SELECTOR_URL or SELECTOR_AI_API_KEY")
    sys.exit(1)

# API Endpoints
SELECTOR_CHAT = "/api/collab2-slack/copilot/v1/chat"
SELECTOR_QUERY = "/api/collab2-slack/command"
SELECTOR_PHRASES = "/api/nlt2/alias"

class SelectorClient:
    def __init__(self):
        self.base_url = SELECTOR_URL
        self.headers = {
            "Authorization": f"Bearer {SELECTOR_AI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def ask(self, content: str) -> Dict[str, Any]:
        logger.info(f"Calling Selector Chat API with: '{content}'")
        return self._post(self.base_url + SELECTOR_CHAT, {"content": content})

    async def query(self, command: str) -> Dict[str, Any]:
        logger.info(f"Calling Selector Query API with: '{command}'")
        return self._post(self.base_url + SELECTOR_QUERY, {"command": command})

    async def get_phrases(self, source: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Calling Selector Phrases API")
        try:
            response = requests.get(self.base_url + SELECTOR_PHRASES, headers=self.headers, timeout=15)
            response.raise_for_status()
            phrases = response.json()
            if source:
                phrases = [p for p in phrases if p.get("source") == source]
            return {"status": "completed", "output": phrases}
        except Exception as e:
            logger.error(f"Error fetching phrases: {e}")
            return {"status": "error", "error": str(e)}

    def _post(self, url: str, payload: dict) -> Dict[str, Any]:
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            response.raise_for_status()
            return {"status": "completed", "output": response.json()}
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            return {"status": "error", "error": str(e)}

selector = SelectorClient()

def send_response(data: Dict[str, Any]):
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()

def discover_tools():
    return {
        "jsonrpc": "2.0",
        "result": [
            {
                "name": "ask_selector",
                "description": "Ask Selector a question",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"]
                }
            },
            {
                "name": "query_selector",
                "description": "Get raw data back from Selector",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            },
            {
                "name": "get_selector_phrases",
                "description": "Get the list of Selector Natural Language phrases",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Optional filter to return phrases from a specific source"
                        }
                    }
                }
            }
        ]
    }

def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "ask_selector":
        return asyncio.run(selector.ask(args.get("content", "")))
    elif name == "query_selector":
        return asyncio.run(selector.query(args.get("command", "")))
    elif name == "get_selector_phrases":
        return asyncio.run(selector.get_phrases(args.get("source")))
    return {"error": "Tool not found"}

def process_request(request: Dict[str, Any]) -> Dict[str, Any]:
    method = request.get("method")
    req_id = request.get("id")

    if method == "tools/discover":
        return {"jsonrpc": "2.0", "id": req_id, **discover_tools()}
    elif method == "tools/call":
        params = request.get("params", {})
        result = call_tool(params.get("name"), params.get("arguments", {}))
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}
        }

def monitor_stdin():
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            request = json.loads(line.strip())
            response = process_request(request)
            send_response(response)
        except Exception as e:
            logger.error(f"Error in monitor_stdin: {e}")
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": str(e)},
                "id": None
            })

async def oneshot():
    try:
        line = sys.stdin.readline()
        request = json.loads(line.strip())
        response = process_request(request)
        send_response(response)
    except Exception as e:
        logger.error(f"One-shot mode error: {e}")
        send_response({
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(e)},
            "id": None
        })

if __name__ == "__main__":
    if "--oneshot" in sys.argv:
        asyncio.run(oneshot())
    else:
        thread = threading.Thread(target=monitor_stdin, daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Server shutting down")