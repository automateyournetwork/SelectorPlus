import os
import json
import time
import logging
import requests
import asyncio
import sys
import threading
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("selector_fastmcp")

# Environment variables
SELECTOR_URL = os.getenv("SELECTOR_URL")
SELECTOR_AI_API_KEY = os.getenv("SELECTOR_AI_API_KEY")

if not SELECTOR_URL or not SELECTOR_AI_API_KEY:
    raise ValueError("Missing SELECTOR_URL or SELECTOR_AI_API_KEY")

# API Endpoints (relative paths)
SELECTOR_CHAT = "/api/collab2-slack/copilot/v1/chat"      # For ask_selector
SELECTOR_QUERY = "/api/collab2-slack/command"          # For query_selector
SELECTOR_PHRASES = "/api/nlt2/alias"                   # For get_selector_phrases

class SelectorClient:
    def __init__(self):
        self.base_url = SELECTOR_URL
        self.chat_endpoint = SELECTOR_CHAT
        self.query_endpoint = SELECTOR_QUERY
        self.phrases_endpoint = SELECTOR_PHRASES
        
        self.headers = {
            "Authorization": f"Bearer {SELECTOR_AI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        logger.info("SelectorClient initialized")

    async def ask(self, content: str) -> Dict[str, Any]:
        logger.info(f"Calling Selector Chat API with: '{content}'")
        api_url = f"{self.base_url}{self.chat_endpoint}"
        payload = {"content": content}
        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload,
                timeout=15
            )
            if response.status_code != 200:
                logger.error(f"API error: {response.text}")
                return {"error": f"API error: {response.status_code}"}
            return response.json()
        except Exception as e:
            logger.error(f"Exception calling API: {str(e)}")
            return {"error": str(e)}
            
    async def query(self, command: str) -> Dict[str, Any]:
        logger.info(f"Calling Selector Query API with: '{command}'")
        api_url = f"{self.base_url}{self.query_endpoint}"
        payload = {"command": command}
        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload,
                timeout=15
            )
            if response.status_code != 200:
                logger.error(f"API error: {response.text}")
                return {"error": f"API error: {response.status_code}"}
            return response.json()
        except Exception as e:
            logger.error(f"Exception calling API: {str(e)}")
            return {"error": str(e)}
    
    async def get_phrases(self, source: str = None) -> Dict[str, Any]:
        logger.info("Getting Selector phrases")
        api_url = f"{self.base_url}{self.phrases_endpoint}"
        try:
            response = requests.get(api_url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                logger.error(f"API error: {response.text}")
                return {"error": f"API error: {response.status_code}"}

            all_phrases = response.json()
            if source:
                filtered = [p for p in all_phrases if p.get("source") == source]
                logger.info(f"Filtered {len(filtered)} phrases with source='{source}'")
                return filtered

            return all_phrases

        except Exception as e:
            logger.error(f"Exception calling API: {str(e)}")
            return {"error": str(e)}

selector = SelectorClient()

def send_response(response_data):
    response = json.dumps(response_data) + "\n"
    sys.stdout.write(response)
    sys.stdout.flush()

def monitor_stdin():
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                time.sleep(0.1)
                continue

            try:
                data = json.loads(line)
                if isinstance(data, dict) and data.get("method") == "tools/call":
                    tool_name = data.get("params",{}).get("name")
                    arguments = data.get("params",{}).get("arguments",{})
                    
                    if tool_name == "ask_selector":
                        content = arguments.get("content", "")
                        result = asyncio.run(selector.ask(content))
                        send_response({"result": result})
                    elif tool_name == "query_selector":
                        command = arguments.get("command", "")
                        result = asyncio.run(selector.query(command))
                        send_response({"result": result})
                    elif tool_name == "get_selector_phrases":
                        source = arguments.get("source")
                        result = asyncio.run(selector.get_phrases(source=source))
                        send_response({"result": result})

                    else:
                        send_response({"error":"tool not found"})
                        
                elif isinstance(data, dict) and data.get("method") == "tools/discover":
                    send_response({
                        "result": [
                            {
                                "name": "ask_selector",
                                "description": "Ask Selector a question",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"}
                                    },
                                    "required": ["content"]
                                }
                            },
                            {
                                "name": "query_selector",
                                "description": "Get raw data back from Selector",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"}
                                    },
                                    "required": ["command"]
                                }
                            },
                            {
                                "name": "get_selector_phrases",
                                "description": "Get the list of Selector Natural Language phrases",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        ]
                    })

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")

        except Exception as e:
            logger.error(f"Exception in monitor_stdin: {str(e)}")
            time.sleep(0.1)

if __name__ == "__main__":
    logger.info("Starting server")

    if "--oneshot" in sys.argv:
        try:
            line = sys.stdin.readline().strip()
            data = json.loads(line)

            if isinstance(data, dict) and data.get("method") == "tools/call":
                tool_name = data.get("params", {}).get("name")
                arguments = data.get("params", {}).get("arguments", {})
                
                if tool_name == "ask_selector":
                    content = arguments.get("content", "")
                    result = asyncio.run(selector.ask(content))
                    send_response({"result": result})
                elif tool_name == "query_selector":
                    command = arguments.get("command", "")
                    result = asyncio.run(selector.query(command))
                    send_response({"result": result})
                elif tool_name == "get_selector_phrases":
                    source = arguments.get("source")
                    result = asyncio.run(selector.get_phrases(source=source))
                    send_response({"result": result})

                else:
                    send_response({"error": "tool not found"})

            elif isinstance(data, dict) and data.get("method") == "tools/discover":
                send_response({
                    "result": [
                        {
                            "name": "ask_selector",
                            "description": "Ask Selector a question",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"}
                                },
                                "required": ["content"]
                            }
                        },
                        {
                            "name": "query_selector",
                            "description": "Get raw data back from Selector",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"}
                                },
                                "required": ["command"]
                            }
                        },
                        {
                            "name": "get_selector_phrases",
                            "description": "Get the list of Selector Natural Language phrases, optionally filtered by source (e.g., user, widget, s2ml)",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "description": "Optional filter to return phrases from a specific source: user, widget, s2ml"
                                    }
                                }
                            }
                        }
                    ]
                })

        except Exception as e:
            logger.error(f"Oneshot error: {e}")
            send_response({"error": str(e)})

    else:
        # Default: run as a server
        stdin_thread = threading.Thread(target=monitor_stdin, daemon=True)
        stdin_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down")