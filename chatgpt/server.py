import os
import json
import time
import logging
import asyncio
import sys
import threading
from typing import Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("chatgpt_fastmcp")

# Load environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized")

# ChatGPT Wrapper Class
class ChatGPTClient:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    async def ask(self, content: str) -> Dict[str, Any]:
        logger.info(f"Sending to ChatGPT: '{content}'")
        try:
            response: ChatCompletion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
            output = response.choices[0].message.content
            logger.info("Received response from ChatGPT")
            return {"output": output}
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": str(e)}

# Initialize the client
chatgpt = ChatGPTClient()

# Output helper
def send_response(response_data: Dict[str, Any]):
    response = json.dumps(response_data) + "\n"
    sys.stdout.write(response)
    sys.stdout.flush()

# stdin monitor loop for server mode
def monitor_stdin():
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                time.sleep(0.1)
                continue

            try:
                data = json.loads(line)
                handle_request(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")

        except Exception as e:
            logger.error(f"Exception in monitor_stdin: {e}")
            time.sleep(0.1)

# Request router
def handle_request(data: Dict[str, Any]):
    if not isinstance(data, dict):
        send_response({"error": "Invalid request format"})
        return

    method = data.get("method")
    if method == "tools/call":
        tool_name = data.get("params", {}).get("name")
        arguments = data.get("params", {}).get("arguments", {})
        if tool_name == "ask_chatgpt":
            content = arguments.get("content", "")
            result = asyncio.run(chatgpt.ask(content))
            send_response({"result": result})
        else:
            send_response({"error": "tool not found"})

    elif method == "tools/discover":
        send_response({
            "result": [
                {
                    "name": "ask_chatgpt",
                    # --- MODIFIED DESCRIPTION ---
                    "description": (
                        "Sends the provided text ('content') to an external ChatGPT (gpt-4o) model "
                        "for analysis, summarization, comparison, or generation tasks. Use this "
                        "ONLY when specifically asked to get ChatGPT's perspective or perform complex "
                        "analysis beyond the primary assistant's capabilities (e.g., detailed security review "
                        "of a config, summarizing a large document). Use this tool to augment your own AI capabilities as access to another external Large Language Model."
                    ),
                    # --- END MODIFIED DESCRIPTION ---
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                # You could optionally add a description for the parameter too
                                "description": "The text content (e.g., configuration, question, document snippet) to send to ChatGPT for processing."
                                }
                        },
                        "required": ["content"]
                    }
                }
            ]
        })

    else:
        send_response({"error": "unknown method"})

# Entry point
if __name__ == "__main__":
    logger.info("Starting Ask ChatGPT MCP Server")

    if "--oneshot" in sys.argv:
        try:
            line = sys.stdin.readline().strip()
            data = json.loads(line)
            handle_request(data)
        except Exception as e:
            logger.error(f"Oneshot error: {e}")
            send_response({"error": str(e)})

    else:
        stdin_thread = threading.Thread(target=monitor_stdin, daemon=True)
        stdin_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down")
