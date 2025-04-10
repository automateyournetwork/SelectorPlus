import os
import sys
import json
import time
import logging
import threading
import asyncio
import requests
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError
import argparse
from functools import partial # Keep partial, might be useful later

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SelectorMCPServer")

# Environment
SELECTOR_URL = os.getenv("SELECTOR_URL")
SELECTOR_AI_API_KEY = os.getenv("SELECTOR_AI_API_KEY")

if not SELECTOR_URL or not SELECTOR_AI_API_KEY:
    logger.critical("Missing SELECTOR_URL or SELECTOR_AI_API_KEY")
    sys.exit(1)
logger.info(f"✅ Using Selector URL: {SELECTOR_URL}")
logger.info("✅ Selector API Key is set (value hidden).")

# API Endpoints
SELECTOR_CHAT = "/api/collab2-slack/copilot/v1/chat"
SELECTOR_QUERY = "/api/collab2-slack/command"
SELECTOR_PHRASES = "/api/nlt2/alias"

# --- Pydantic Models for Input Validation ---
class AskSelectorInput(BaseModel):
    content: str = Field(..., title="Content", description="The question or content to send to Selector.")

class QuerySelectorInput(BaseModel):
    command: str = Field(..., title="Command", description="The command to query Selector.")

class GetPhrasesInput(BaseModel):
    source: Optional[str] = Field(None, title="Source", description="Optional filter for phrases by source.")

# --- Refactored SelectorClient with proper async handling ---
class SelectorClient:
    def __init__(self):
        self.base_url = SELECTOR_URL.rstrip('/') # Ensure no trailing slash initially
        self.headers = {
            "Authorization": f"Bearer {SELECTOR_AI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # It's better to create the session once if making multiple calls,
        # but for run_in_executor, simple requests calls are fine.
        # self.session = requests.Session() # Use requests.Session for potential keep-alive
        # self.session.headers.update(self.headers)

    async def _run_sync_in_executor(self, func, *args, **kwargs):
        """Helper to run synchronous blocking function in executor."""
        loop = asyncio.get_running_loop()
        # Use default executor (usually ThreadPoolExecutor)
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    async def ask(self, params: dict) -> Dict[str, Any]:
        try:
            validated_input = AskSelectorInput(**params)
            logger.info(f"Calling Selector Chat API with: '{validated_input.content}'")
            # Use helper to run blocking _post in executor
            return await self._run_sync_in_executor(
                self._post,
                self.base_url + SELECTOR_CHAT,
                {"content": validated_input.content}
            )
        except ValidationError as ve:
            logger.warning(f"Input validation failed for ask: {ve}")
            return {"status": "error", "error": f"Invalid input: {ve}"}
        except Exception as e: # Catch potential errors from _run_sync_in_executor
            logger.error(f"Error during ask execution: {e}", exc_info=True)
            return {"status": "error", "error": f"Internal error during ask: {e}"}


    async def query(self, params: dict) -> Dict[str, Any]:
        try:
            validated_input = QuerySelectorInput(**params)
            logger.info(f"Calling Selector Query API with: '{validated_input.command}'")
            # Use helper to run blocking _post in executor
            return await self._run_sync_in_executor(
                self._post,
                self.base_url + SELECTOR_QUERY,
                {"command": validated_input.command}
            )
        except ValidationError as ve:
            logger.warning(f"Input validation failed for query: {ve}")
            return {"status": "error", "error": f"Invalid input: {ve}"}
        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            return {"status": "error", "error": f"Internal error during query: {e}"}

    async def get_phrases(self, params: dict) -> Dict[str, Any]:
        try:
            validated_input = GetPhrasesInput(**params)
            source_filter = validated_input.source
            logger.info(f"Calling Selector Phrases API (Source filter: {source_filter or 'None'})")

            # Define the blocking part
            def _fetch_and_filter():
                response = requests.get(self.base_url + SELECTOR_PHRASES, headers=self.headers, timeout=15)
                response.raise_for_status()
                phrases = response.json()
                if source_filter:
                    filtered_phrases = [p for p in phrases if p.get("source") == source_filter]
                    logger.info(f"Fetched {len(phrases)} phrases, filtered to {len(filtered_phrases)}.")
                    return filtered_phrases
                else:
                     logger.info(f"Fetched {len(phrases)} phrases.")
                     return phrases

            # Run the blocking part in executor
            filtered_phrases = await self._run_sync_in_executor(_fetch_and_filter)
            return {"status": "completed", "output": filtered_phrases}

        except ValidationError as ve:
            logger.warning(f"Input validation failed for get_phrases: {ve}")
            return {"status": "error", "error": f"Invalid input: {ve}"}
        except requests.exceptions.RequestException as re:
             logger.error(f"Error fetching phrases (RequestException): {re}")
             return {"status": "error", "error": f"HTTP Request Error: {re}"}
        except Exception as e:
            logger.error(f"Error during get_phrases execution: {e}", exc_info=True)
            return {"status": "error", "error": f"Internal error during get_phrases: {e}"}

    # This remains synchronous as it's called via run_in_executor
    def _post(self, url: str, payload: dict) -> Dict[str, Any]:
        # This function now executes in a separate thread via run_in_executor
        try:
            # logger.debug(f"Executing POST to {url} in executor thread: {threading.current_thread().name}")
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            response.raise_for_status()
            return {"status": "completed", "output": response.json()}
        except requests.exceptions.RequestException as re:
             logger.error(f"POST request failed (RequestException): {re}")
             # Attempt to get more detail from response if possible
             error_detail = str(re)
             if re.response is not None:
                  try:
                       error_detail = f"HTTP {re.response.status_code}: {re.response.text[:500]}"
                  except Exception:
                       pass # Stick with original error string
             return {"status": "error", "error": f"HTTP Request Error: {error_detail}"}
        except Exception as e:
            logger.error(f"POST request failed unexpectedly in executor: {e}", exc_info=True)
            return {"status": "error", "error": f"Internal error during POST: {e}"}

selector = SelectorClient()

# --- Tool Definitions ---
# Functions should now reference the async methods directly
# Functions should now reference the async methods directly
AVAILABLE_TOOLS = {
    "ask_selector": {
        "function": selector.ask, # Reference the async method
        "description": (
            "Use this primary tool to ask general questions or give instructions to the Selector AI Assistant "
            "in natural language (e.g., 'show me device health', 'what are the top alerts?', 'summarize network status'). "
            "It interacts with the Selector Chat/Copilot API to understand intent, provide insights, or potentially execute actions. "
            "This is the default tool for most user requests unless they explicitly ask for specific raw query data or a list of phrases."
        ),
        "input_model": AskSelectorInput.model_json_schema(), # Use schema from Pydantic model
    },
    "query_selector": {
        "function": selector.query, # Reference the async method
        "description": (
            "Executes a specific, pre-defined Selector query command string (usually starting with '#') "
            "directly against the Selector Query API to retrieve structured, raw data. "
            "Use ONLY when the user provides an exact query command string. Do NOT use for general questions."
        ),
        "input_model": QuerySelectorInput.model_json_schema(), # Use schema from Pydantic model
    },
    "get_selector_phrases": {
        "function": selector.get_phrases, # Reference the async method
        "description": (
            "Retrieves the list of saved Natural Language Phrases (aliases/shortcuts) registered in the Selector system. "
            "Use ONLY when the user explicitly asks to 'list phrases', 'show aliases', 'get commands', or similar requests for the list itself. "
            "Do NOT use this to execute a phrase or ask a general question."
        ),
        "input_model": GetPhrasesInput.model_json_schema(), # Use schema from Pydantic model
    },
}

logger.info(f"🚀 Registered {len(AVAILABLE_TOOLS)} tools at startup: {list(AVAILABLE_TOOLS.keys())}")

# --- JSON-RPC Handling ---
def discover_tools() -> List[Dict[str, Any]]:
    logger.info(f"🛠 discover_tools(): AVAILABLE_TOOLS keys: {list(AVAILABLE_TOOLS.keys())}")
    tools_list = []

    for name, tool_info in AVAILABLE_TOOLS.items():
        # Clean schema for MCP protocol (remove titles)
        raw_schema = tool_info["input_model"]
        cleaned_properties = {
            k: {k2: v2 for k2, v2 in v.items() if k2 != "title"}
            for k, v in raw_schema.get("properties", {}).items()
        }
        input_schema = {
            "type": "object",
            "properties": cleaned_properties,
            "required": raw_schema.get("required", []),
            "additionalProperties": False # Often good practice
        }

        tools_list.append({
            "name": name,
            "description": tool_info["description"],
            "parameters": input_schema, # Use cleaned schema
        })

    logger.info(f"✅ discover_tools() returning {len(tools_list)} tools")
    return tools_list

# Needs to be async now to await the tool functions
async def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in AVAILABLE_TOOLS:
        logger.warning(f"Requested tool '{tool_name}' not found.")
        return {"error": {"code": -32601, "message": f"Method not found: {tool_name}"}}

    tool_info = AVAILABLE_TOOLS[tool_name]
    func = tool_info["function"] # func is now an async function

    try:
        # Await the async function
        result_data = await func(arguments)
        # Ensure result is JSON serializable (optional check)
        # json.dumps(result_data)
        return result_data
    except Exception as e:
        logger.error(f"Unexpected error calling tool '{tool_name}': {e}", exc_info=True)
        return {"error": {"code": -32603, "message": f"Internal server error during tool call: {e}"}}

# process_request remains async
async def process_request(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_id = request_data.get("id")

    if not isinstance(request_data, dict) or \
       request_data.get("jsonrpc") != "2.0" or \
       "method" not in request_data:
        logger.warning(f"Invalid JSON-RPC request received: {request_data}")
        return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": request_id}

    method = request_data["method"]
    params = request_data.get("params", {})
    logger.info(f"📥 Incoming method: {method}, params: {params}")

    if method in ("tools/discover", "tools/list"):
        tools = discover_tools() # discover_tools is sync, okay to call directly
        return {"jsonrpc": "2.0", "id": request_id, "result": tools}

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not tool_name or not isinstance(arguments, dict):
            logger.warning(f"Invalid params for tools/call: {params}")
            return {"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params"}, "id": request_id}

        # Await the now async call_tool function
        logger.info(f"Dispatching async call_tool for '{tool_name}'...")
        result_or_error = await call_tool(tool_name, arguments)
        # call_tool now returns the final result or error dict directly
        if "error" in result_or_error:
             # Assume error format is already JSON-RPC compliant
             logger.warning(f"Tool call resulted in error: {result_or_error['error']}")
             return {"jsonrpc": "2.0", "error": result_or_error["error"], "id": request_id}
        else:
             # Assume result format is already JSON-RPC compliant
             logger.info(f"Tool call successful for '{tool_name}'.")
             return {"jsonrpc": "2.0", "result": result_or_error, "id": request_id}

    logger.warning(f"Method not found: {method}")
    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": request_id}

# send_response remains synchronous
def send_response(response_data: Dict[str, Any]):
    try:
        response_string = json.dumps(response_data) + "\n"
        sys.stdout.write(response_string)
        sys.stdout.flush()
        logger.debug(f"Sent response: {response_string.strip()}")
    except Exception as e:
        logger.error(f"Failed to write response to stdout: {e}", exc_info=True)

# monitor_stdin handles the reading loop and calls asyncio.run per request
def monitor_stdin():
    logger.info("Stdin monitoring thread started.")
    while True:
        try:
            logger.debug("Waiting for input...")
            line = sys.stdin.readline()
            if not line:
                logger.warning("Stdin closed or empty line received. Exiting monitor thread.")
                break # Exit loop if stdin closes

            line = line.strip()
            if not line:
                time.sleep(0.05) # Avoid busy-waiting on empty lines
                continue

            logger.debug(f"Received line: {line}")
            try:
                request_data = json.loads(line)
                # Run the async request processing for each received line
                response = asyncio.run(process_request(request_data)) # Runs the async chain
                if response:
                    send_response(response)
                else:
                    # Handle cases where process_request might return None (e.g., notifications)
                    logger.info("No response generated for request.")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} for line: '{line}'")
                send_response({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                    "id": None # ID might not be available
                })
            except Exception as e:
                # Catch errors from process_request or asyncio.run itself
                logger.error(f"Error processing request: {e}", exc_info=True)
                # Try to get request_id if possible, might fail if parsing failed
                req_id = None
                try: req_id = json.loads(line).get("id")
                except Exception: pass
                send_response({
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal server error: {e}"},
                    "id": req_id,
                })

        except Exception as e:
            logger.error(f"Exception in monitor_stdin loop: {e}", exc_info=True)
            # Avoid rapid looping on unexpected errors
            time.sleep(1)


# --- Corrected One-Shot Logic ---
async def run_server_oneshot():
    """Reads one JSON request from stdin, processes it, writes response."""
    logger.info("Starting Selector MCP Server in one-shot mode...")
    response_sent = False
    request_id_oneshot = None # Initialize
    try:
        input_data = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.read)
        # input_data = sys.stdin.read() # Blocking read - needs executor in async
        logger.info(f"Received raw input: {input_data[:500]}{'...' if len(input_data) > 500 else ''}")

        # Find the last valid JSON object in the input (safer for one-shot)
        last_json_line = None
        potential_jsons = []
        for line in input_data.strip().splitlines():
             line = line.strip()
             if line.startswith('{') and line.endswith('}'):
                  potential_jsons.append(line)
        if potential_jsons:
            last_json_line = potential_jsons[-1] # Take the last one

        if not last_json_line:
            logger.error("No valid JSON object found in input.")
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error: No valid JSON found"},
                "id": None
            })
            response_sent = True
            return

        logger.info(f"Processing JSON: {last_json_line}")
        request_json = json.loads(last_json_line)
        request_id_oneshot = request_json.get("id") # Store ID for error handling
        response = await process_request(request_json) # Process the single request
        if response:
            send_response(response)
            response_sent = True

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error (oneshot): {e}")
        if not response_sent:
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
                "id": request_id_oneshot # Use stored ID if available
            })
    except Exception as e:
        logger.error(f"Unhandled Server Error (oneshot): {e}", exc_info=True)
        if not response_sent:
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": f"Server error: {e}"},
                "id": request_id_oneshot # Use stored ID
            })
    finally:
        logger.info("Selector MCP Server one-shot finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selector MCP Server")
    parser.add_argument("--oneshot", action="store_true", help="Run in one-shot mode")
    args = parser.parse_args()

    if args.oneshot:
        # Run the dedicated async one-shot handler
        asyncio.run(run_server_oneshot())
    else: # Continuous mode
        logger.info("Starting Selector MCP Server in continuous stdio mode...")
        # Start monitor_stdin in a separate thread for continuous operation
        stdin_thread = threading.Thread(target=monitor_stdin, name="StdinMonitorThread", daemon=True)
        stdin_thread.start()
        try:
            # Keep main thread alive
            while stdin_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
        except Exception as e:
             logger.error(f"Unexpected error in main thread: {e}", exc_info=True)
        finally:
            logger.info("Main thread exiting.")
            # Flush output just in case
            sys.stdout.flush()
            sys.stderr.flush()