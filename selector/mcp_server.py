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

SELECTOR_API_URL = f"{SELECTOR_URL}/api/collab2-slack/copilot/v1/chat"

class SelectorClient:
    def __init__(self):
        self.api_url = SELECTOR_API_URL
        self.headers = {
            "Authorization": f"Bearer {SELECTOR_AI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        logger.info("SelectorClient initialized")

    async def ask(self, content: str) -> Dict[str, Any]:
        logger.info(f"Calling Selector API with: '{content}'")
        payload = {"content": content}
        try:
            response = requests.post(
                self.api_url,
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
                    else:
                        send_response({"error":"tool not found"})
                elif isinstance(data, dict) and data.get("method") == "tools/discover":
                    send_response({"result":[{"name":"ask_selector", "description":"ask selector a question", "parameters":{"type":"object","properties":{"content":{"type":"string"}}}}]})

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
