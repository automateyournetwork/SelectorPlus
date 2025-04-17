import os
import json
import httpx
import uuid
import asyncio
import inspect
import logging
import importlib
import subprocess
from functools import wraps, partial
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import Tool, StructuredTool
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from typing import Dict, Any, List, Optional, Union, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode
#from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

A2A_PEER_AGENTS = os.getenv("A2A_PEER_AGENTS", "").split(",")

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State class for LangGraph."""
    messages: Annotated[list[BaseMessage], add_messages]
    context: dict
    file_path: Optional[str]  # To store the file path

def load_local_tools_from_folder(folder_path: str) -> List[Tool]:
    """Loads tools from a local folder."""
    local_tools = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"{folder_path}.{module_name}")
                for name, obj in inspect.getmembers(module):
                    if isinstance(obj, Tool):
                        wrapped = wrap_dict_input_tool(obj)
                        local_tools.append(wrapped)
                        print(f"✅ Loaded local tool: {wrapped.name}")
                    elif isinstance(obj, StructuredTool):
                        local_tools.append(obj)
                        print(f"✅ Loaded structured tool: {obj.name}")
            except Exception as e:
                print(f"❌ Failed to import {module_name}: {e}")
    return local_tools

def wrap_dict_input_tool(tool_obj: Tool) -> Tool:
    """Wraps a tool function to handle string or dict input."""
    original_func = tool_obj.func

    @wraps(original_func)
    def wrapper(input_value):
        if isinstance(input_value, str):
            input_value = {"ip": input_value}
        elif isinstance(input_value, dict) and "ip" not in input_value:
            logger.warning(f"⚠️ Missing 'ip' key in dict: {input_value}")
        return original_func(input_value)

    return Tool(
        name=tool_obj.name,
        description=tool_obj.description,
        func=wrapper,
    )

def schema_to_pydantic_model(name: str, schema: dict):
    """Dynamically creates a Pydantic model class from a JSON Schema."""
    from typing import Any, List, Dict, Optional
    namespace = {"__annotations__": {}}

    if schema.get("type") != "object":
        raise ValueError("Only object schemas are supported.")

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        is_optional = field_name not in required_fields

        if json_type == "string":
            field_type = str
        elif json_type == "boolean":
             field_type = bool
        elif json_type == "array":
            items_schema = field_schema.get("items")
            if not items_schema:
                logger.warning(f"⚠️ Skipping field '{field_name}' (array missing 'items')")
                continue
            item_type = items_schema.get("type", "string")

            if item_type == "string":
                 field_type = List[str]
            elif item_type == "integer":
                 field_type = List[int]
            elif item_type == "number":
                 field_type = List[float]
            elif item_type == "boolean":
                 field_type = List[bool]

            elif item_type == "object":
                # Check if the items schema actually defines properties
                if "properties" in items_schema and items_schema["properties"]:
                    # If properties are defined, create a specific item model
                    item_model = schema_to_pydantic_model(name + "_" + field_name + "_Item", items_schema)
                    field_type = List[item_model]
                else:
                    # If no properties defined for items, assume generic dictionaries
                    logger.warning(f"Treating array item '{field_name}' as generic List[Dict[str, Any]] due to missing/empty properties in items schema.")
                    field_type = List[Dict[str, Any]] # Use List[Dict] instead of List[EmptyModel]
            else: # Handle array of Any
                field_type = List[Any]

        elif json_type == "object":
             # Also check objects - if no properties, maybe treat as Dict[str, Any]?
             if "properties" in field_schema and field_schema["properties"]:
                   # Potentially create nested model if needed, or keep as Dict for simplicity
                   field_type = Dict[str, Any] # Keeping as Dict for now
             else:
                   field_type = Dict[str, Any] # Generic object becomes Dict

        else: # Handle Any type
            field_type = Any

        if is_optional:
            field_type = Optional[field_type]

        namespace["__annotations__"][field_name] = field_type
        if field_name in required_fields:
            namespace[field_name] = Field(...)
        else:
            namespace[field_name] = Field(default=None)

    return type(name, (BaseModel,), namespace)


# Define the input schema for the delegation tool
class DelegateToPeerSchema(BaseModel):
    peer_agent_url: str = Field(description="The base URL of the peer A2A agent to contact (e.g., http://agent2.example.com:10001).")
    task_description: str = Field(description="The specific task or question to send to the peer agent.")
    session_id: Optional[str] = Field(default=None, description="Optional session ID to maintain conversation context.")

# Define the asynchronous function for the tool
async def delegate_task_to_peer_agent(peer_agent_url: str, task_description: str, session_id: Optional[str] = None) -> str:
    """
    Sends a task to a peer A2A agent and returns its text response or an error message.
    Uses the standard A2A Task model for communication.
    """
    logger.info(f"Attempting to delegate task to peer: {peer_agent_url}")
    
    # Basic URL cleanup and ensure protocol
    peer_agent_url = peer_agent_url.strip().rstrip("/")
    if not (peer_agent_url.startswith("http://") or peer_agent_url.startswith("https://")):
         peer_agent_url = "http://" + peer_agent_url
    
    # Assume the main endpoint is at '/' relative to the base URL
    endpoint = f"{peer_agent_url}/" 
    
    request_id = str(uuid.uuid4())
    task_param_id = str(uuid.uuid4())
    current_session_id = session_id or str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send", # Standard A2A method
        "params": {
            "id": task_param_id,
            "sessionId": current_session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_description}]
            },
            "acceptedOutputModes": ["text"] # Specify desired output
            # Add other params like historyLength if needed
        },
        "id": request_id
    }

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Sending A2A task to {endpoint}: {json.dumps(payload)}")
            resp = await client.post(endpoint, json=payload, timeout=60.0) # Increased timeout
            resp.raise_for_status() # Raise exception for 4xx/5xx errors

            response_data = resp.json()
            logger.info(f"Received A2A response from {peer_agent_url}: {json.dumps(response_data)}")

            # Extract the result according to the A2A Task model
            result = response_data.get("result")
            if not result:
                return f"Error: Peer agent response missing 'result' field. Raw: {json.dumps(response_data)}"

            status = result.get("status")
            if not status:
                 return f"Error: Peer agent result missing 'status' field. Raw: {json.dumps(response_data)}"

            if status.get("state") == "failed":
                 error_message = status.get("message", {}).get("parts", [{}])[0].get("text", "Unknown error")
                 return f"Error: Peer agent failed task: {error_message}"

            # Try to get the text response
            response_message = status.get("message", {}).get("parts", [{}])[0].get("text")
            if response_message:
                return response_message
            else:
                # Handle cases where response might be elsewhere or missing
                logger.warning(f"Could not extract text response from peer agent's successful status. Raw: {json.dumps(response_data)}")
                # Fallback: return the whole result status as string
                return f"Peer agent completed task, but no standard text response found. Status: {json.dumps(status)}"

    except httpx.RequestError as e:
        logger.error(f"Network error delegating task to {peer_agent_url}: {e}")
        return f"Error: Network error connecting to peer agent {peer_agent_url}: {e}"
    except httpx.HTTPStatusError as e:
         logger.error(f"HTTP error delegating task to {peer_agent_url}: Status {e.response.status_code}, Response: {e.response.text}")
         return f"Error: Peer agent {peer_agent_url} returned HTTP status {e.response.status_code}. Response: {e.response.text}"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {peer_agent_url}: {e}. Response text: {resp.text}")
        return f"Error: Could not decode JSON response from peer agent {peer_agent_url}."
    except Exception as e:
        logger.error(f"Unexpected error delegating task to {peer_agent_url}", exc_info=True)
        return f"Error: An unexpected error occurred while delegating: {e}"

# Create the StructuredTool instance
a2a_delegation_tool = StructuredTool.from_function(
    name="delegate_task_to_peer_agent",
    description="Sends a specific task or question to another A2A-compatible agent at a given URL and returns its response. Use this when you lack the capability locally or are explicitly asked to consult another agent.",
    args_schema=DelegateToPeerSchema,
    coroutine=delegate_task_to_peer_agent # Use the async function
)

class MCPToolDiscovery:
    """Discovers and calls tools in MCP containers."""
    def __init__(self, container_name: str, command: List[str], discovery_method: str = "tools/discover",
                 call_method: str = "tools/call"):
        self.container_name = container_name
        self.command = command
        self.discovery_method = discovery_method
        self.call_method = call_method
        self.discovered_tools = []

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discovers tools from the MCP container asynchronously."""
        print(f"🔍 Async Discovering tools from container: {self.container_name}")
        print(f"🕵️ Discovery Method: {self.discovery_method}")

        try:
            discovery_payload = {
                "jsonrpc": "2.0",
                "method": self.discovery_method,
                "params": {},
                "id": "1"
            }
            payload_bytes = (json.dumps(discovery_payload) + "\n").encode('utf-8')
            print(f"Sending async discovery payload: {payload_bytes.decode('utf-8').strip()}")

            command_list = ["docker", "exec", "-i", self.container_name] + self.command

            # --- Use asyncio.create_subprocess_exec ---
            process = await asyncio.create_subprocess_exec(
                *command_list,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate(input=payload_bytes)
            # --- End asyncio subprocess usage ---

            stdout_str = stdout.decode('utf-8').strip() if stdout else ""
            stderr_str = stderr.decode('utf-8').strip() if stderr else ""

            print(f"📥 Async Raw discovery stdout:\n{stdout_str}")
            if stderr_str:
                 print(f"📥 Async Raw discovery stderr:\n{stderr_str}")

            if process.returncode != 0:
                 print(f"❌ Async Discovery subprocess failed with code {process.returncode}")
                 return []

            # Process output lines
            stdout_lines = stdout_str.strip().split("\n")
            if stdout_lines:
                last_line = None
                for line in reversed(stdout_lines):
                    # More robust check for start of JSON object or array
                    trimmed_line = line.strip()
                    if trimmed_line.startswith("{") and trimmed_line.endswith("}"):
                         last_line = trimmed_line
                         break
                    if trimmed_line.startswith("[") and trimmed_line.endswith("]"):
                         last_line = trimmed_line
                         break
                if last_line:
                    try:
                        response = json.loads(last_line)
                        # Standardize tool extraction slightly
                        tools_list = []
                        if isinstance(response.get("result"), list):
                             tools_list = response["result"]
                        elif isinstance(response.get("result"), dict) and "tools" in response["result"]:
                             tools_list = response["result"]["tools"]

                        if tools_list:
                            # Ensure names exist before logging
                            tool_names = [t.get("name", "Unnamed") for t in tools_list if isinstance(t, dict)]
                            print(f"✅ Async Discovered tools: {tool_names}")
                            return tools_list # Return the list of tool dicts
                        else:
                            print("❌ No tools found in JSON response.")
                            return []
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON Decode Error in discovery: {e} on line: {last_line}")
                        return []
                else:
                    print("❌ No valid JSON line found in discovery output.")
                    return []
            else:
                print("❌ No stdout lines received from discovery.")
                return []
        except Exception as e:
            print(f"❌ Error during async tool discovery: {e}", exc_info=True)
            return []

    @traceable
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Calls a tool in the MCP container asynchronously."""
        logger.info(f"🔍 Attempting Async call tool: {tool_name}")
        logger.info(f"📦 Arguments: {arguments}")

        # Note: Network inspection via subprocess.run is also blocking!
        # If needed, run this separately or make it async too. For now, let's comment it out
        # try:
        #     network_inspect = subprocess.run(...) # THIS IS BLOCKING
        #     logger.info(f"🌐 Network Details: {network_inspect.stdout}")
        # except Exception as e:
        #     logger.error(f"❌ Network inspection failed: {e}")

        command_list = ["docker", "exec", "-i", self.container_name] + self.command

        try:
            try:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                # 🔧 Unwrap __arg1 if it's a stringified JSON object
                if "__arg1" in arguments:
                    try:
                        unwrapped = json.loads(arguments["__arg1"])
                        if isinstance(unwrapped, dict):
                            arguments = unwrapped
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse '__arg1' as JSON", exc_info=True)
                        raise ValueError("Malformed __arg1 argument") from e

                normalized_args = arguments.copy()
            except Exception as e:
                logger.error(f"Error normalizing arguments: {e}")
                raise

            # Handle specific cases if necessary (like the 'sha' key)
            if tool_name == "create_or_update_file" and "sha" in normalized_args and normalized_args["sha"] is None:
                del normalized_args["sha"]

            payload = {
                "jsonrpc": "2.0",
                "method": self.call_method,
                "params": {"name": tool_name, "arguments": normalized_args},
                "id": "2",
            }
            payload_bytes = (json.dumps(payload) + "\n").encode('utf-8') # Encode payload

            logger.info(f"🚀 Async Payload: {payload_bytes.decode('utf-8').strip()}")

            # --- Use asyncio.create_subprocess_exec ---
            process = await asyncio.create_subprocess_exec(
                *command_list,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"} # Ensure env is passed if needed
            )

            # Send payload and receive output without blocking
            stdout, stderr = await process.communicate(input=payload_bytes)
            # --- End asyncio subprocess usage ---


            stdout_str = stdout.decode('utf-8').strip() if stdout else ""
            stderr_str = stderr.decode('utf-8').strip() if stderr else ""

            logger.info(f"🔬 Async Subprocess Exit Code: {process.returncode}")
            logger.info(f"🔬 Async Full subprocess stdout:\n{stdout_str}")
            logger.info(f"🔬 Async Full subprocess stderr:\n{stderr_str}")

            if process.returncode != 0:
                logger.error(f"❌ Async Subprocess returned non-zero exit code: {process.returncode}")
                logger.error(f"🚨 Async Error Details: {stderr_str}")
                # Maybe return more specific error based on stderr?
                return f"Subprocess Error (Exit Code {process.returncode}): {stderr_str or 'No stderr'}"

            # Process output lines (similar to before, but now on decoded strings)
            output_lines = stdout_str.strip().split("\n")
            for line in reversed(output_lines):
                try:
                    response = json.loads(line)
                    logger.info(f"✅ Parsed JSON response: {response}")
                    if "result" in response:
                        return response["result"]
                    elif "error" in response:
                        # Handle potential errors reported by the tool service
                        error_message = response["error"]
                        logger.error(f"🚨 Tool service reported error: {error_message}")
                        return f"Tool Error: {error_message}" # Return the error message from the tool

                except json.JSONDecodeError:
                    # Ignore lines that aren't valid JSON
                    logger.debug(f"Ignoring non-JSON line: {line}")
                    continue

            logger.error("❌ No valid JSON response found in stdout.")
            return "Error: No valid JSON response found in tool output."

        except Exception as e:
            # Catch potential exceptions during subprocess creation or communication
            logger.critical(f"🔥 Critical async tool call error for {tool_name}", exc_info=True)
            return f"Critical Error during tool call: {str(e)}"

# Add these helper functions somewhere *before* get_tools_for_service in selectorplus.py
# --- Base async functions OUTSIDE the loop ---
async def _base_mcp_call(tool_name_to_call: str, service_discovery_instance: MCPToolDiscovery, args_dict: Union[str, dict]):
    """Generic async function to call an MCP tool."""
    logger.info(f"PARTIAL_TRACE: _base_mcp_call invoked for '{tool_name_to_call}' with args {args_dict}")

    # 🛡️ Ensure it's a dict
    try:
        if isinstance(args_dict, str):
            args_dict = json.loads(args_dict)
    except Exception as e:
        logger.error(f"❌ Failed to parse args_dict string as JSON: {e}")
        return json.dumps({"status": "error", "error": f"Invalid input: {e}"})

    # ✅ Normalize for send-email
    if tool_name_to_call == "send-email":
        if "body" in args_dict and "text" not in args_dict:
            args_dict["text"] = args_dict.pop("body")

    return await service_discovery_instance.call_tool(tool_name_to_call, args_dict)


async def _structured_mcp_call(tool_name_to_call: str, service_discovery_instance: MCPToolDiscovery, pydantic_model: type[BaseModel], **kwargs):
    """Generic async function for structured MCP tools with Pydantic validation."""
    logger.info(f"PARTIAL_TRACE: _structured_mcp_call invoked for '{tool_name_to_call}' with raw kwargs {kwargs}")
    try:
        # Validate and clean args using Pydantic model
        validated_args = pydantic_model(**kwargs).model_dump(exclude_unset=True) # Use model_dump for Pydantic v2+
        logger.info(f"PARTIAL_TRACE: Validation successful for '{tool_name_to_call}', validated args: {validated_args}")
        # Directly call the instance method
        return await service_discovery_instance.call_tool(tool_name_to_call, validated_args)
    except ValidationError as ve:
        logger.warning(f"PARTIAL_TRACE: Pydantic validation failed for {tool_name_to_call}: {ve}")
        # Return an error structure ToolNode can process (e.g., JSON string in content)
        # Ensure the content is a string for ToolMessage
        return json.dumps({"status": "error", "error": f"Input validation failed: {ve}"})
    except Exception as e:
        logger.error(f"PARTIAL_TRACE: Unexpected error during structured call for {tool_name_to_call}: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": f"Unexpected error during tool execution: {e}"})


# --- Replace the existing get_tools_for_service function in selectorplus.py with this ---
async def get_tools_for_service(service_name, command, discovery_method, call_method, service_discoveries):
    """Enhanced tool discovery using functools.partial for safe coroutine binding."""
    print(f"🕵️ Discovering tools for: {service_name}")
    discovery = MCPToolDiscovery(
        container_name=service_name,
        command=command,
        discovery_method=discovery_method,
        call_method=call_method
    )
    service_discoveries[service_name] = discovery  # Store for future tool calls

    tools = []
    try:
        # Make sure discovery actually returns something
        discovered_tools_list = await discovery.discover_tools()
        if not discovered_tools_list: # Handle case where discovery fails or returns empty
             print(f"⚠️ No tools discovered for {service_name}")
             return []

        # Use .get() for safer access to tool names in the log message
        print(f"🛠️ Tools for {service_name}: {[t.get('name', 'Unnamed') for t in discovered_tools_list]}")

        for tool_info in discovered_tools_list: # Use a different variable name 'tool_info'
            tool_name = tool_info.get("name")
            if not tool_name:
                logger.warning(f"⚠️ Skipping tool with no name for service {service_name}: {tool_info}")
                continue

            tool_description = tool_info.get("description", f"Tool {tool_name} from {service_name}") # Default desc
            # Use 'parameters' as the key based on your MCP server output schema
            tool_schema = tool_info.get("parameters", {})

            # Use functools.partial to create the coroutine safely binding current values
            specific_service_discovery = service_discoveries[service_name] # Get instance for this service

            # Check if schema exists, is an object, and has properties for StructuredTool
            if tool_schema and tool_schema.get("type") == "object" and tool_schema.get("properties"):
                try:
                    input_model = schema_to_pydantic_model(f"{service_name.replace('-', '_')}_{tool_name}_Input", tool_schema)

                    # Create the partial function for the structured call
                    # This binds the *current* values of tool_name, specific_service_discovery, and input_model
                    tool_coroutine = partial(
                        _structured_mcp_call,
                        tool_name,
                        specific_service_discovery,
                        input_model
                    )

                    structured_tool = StructuredTool.from_function( # Use class method for easier creation
                        name=tool_name,
                        description=tool_description,
                        args_schema=input_model,
                        coroutine=tool_coroutine # Pass the partial coroutine
                    )
                    tools.append(structured_tool)
                    logger.debug(f"✅ Added StructuredTool (via partial): {tool_name}")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to build structured tool {tool_name} via partial: {e}. Adding as simple tool.")
                    # Fallback: Create partial for the simple base call
                    tool_coroutine = partial(
                         _base_mcp_call,
                         tool_name,
                         specific_service_discovery
                     )
                    simple_tool = Tool.from_function( # Use class method
                        name=tool_name,
                        description=tool_description,
                        func=None, # No sync implementation
                        coroutine=tool_coroutine
                    )
                    tools.append(simple_tool)
                    logger.debug(f"✅ Added Simple Tool (fallback via partial): {tool_name}")
            else:
                # Simple Tool Path (No args/schema or non-object schema)
                # Create partial for the simple base call
                tool_coroutine = partial(
                    _base_mcp_call,
                    tool_name,
                    specific_service_discovery
                )
                simple_tool = Tool.from_function( # Use class method
                    name=tool_name,
                    description=tool_description,
                    func=None, # No sync implementation
                    coroutine=tool_coroutine # Use the partial coroutine
                )
                tools.append(simple_tool)
                logger.debug(f"✅ Added Simple Tool (via partial): {tool_name}")

    except Exception as e:
        # Log any error during the overall discovery/processing for the service
        logger.error(f"❌ Tool discovery/processing error in {service_name}: {e}", exc_info=True)

    return tools

async def discover_agent(url: str) -> Optional[dict]:
    """
    Discovers metadata from a peer agent by fetching its /.well-known/agent.json file.

    Args:
        url (str): Base URL of the peer agent, e.g. http://agent2:10001

    Returns:
        dict: The parsed agent metadata if successful, or None if not.
    """
    cleaned_url = url.strip().rstrip("/")
    if not cleaned_url.startswith("http://") and not cleaned_url.startswith("https://"):
        cleaned_url = "http://" + cleaned_url

    discovery_url = f"{cleaned_url}/.well-known/agent.json"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(discovery_url, timeout=10.0)
            response.raise_for_status()
            agent_data = response.json()
            return agent_data
    except httpx.RequestError as e:
        logger.error(f"Network error discovering peer agent at {discovery_url}: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} when accessing {discovery_url}: {e.response.text}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {discovery_url}")
    except Exception as e:
        logger.error(f"Unexpected error discovering peer agent at {discovery_url}: {e}", exc_info=True)

    return None

def make_delegation_coroutine(peer_agent_url: str):
    async def wrapper(**kwargs):
        return await delegate_task_to_peer_agent(peer_agent_url=peer_agent_url, **kwargs)
    return wrapper

async def load_delegated_tools(peer_agents: Dict[str, dict]) -> List[Tool]:
    """Creates delegation tools and wraps each peer agent's skills."""
    delegated_tools = []

    for url, agent_card in peer_agents.items():
        agent_name = agent_card.get("name", "peer").replace(" ", "_").lower()

        for skill in agent_card.get("skills", []):
            tool_name = skill["id"]
            tool_description = skill.get("description", "")
            tool_schema = skill.get("parameters", {})

            try:
                InputModel = schema_to_pydantic_model(f"{tool_name}_Input", tool_schema)

                async def make_delegate(peer_url=url, skill_id=tool_name):
                    async def delegate(**kwargs):
                        # Filter out None values from kwargs just like you do for local tools
                        filtered_tool_input = {k: v for k, v in kwargs.items() if v is not None}

                        return await delegate_task_to_peer_agent(
                            peer_agent_url=peer_url,
                            task_description=f"Call remote tool '{skill_id}' with args: {json.dumps(filtered_tool_input)}"
                        )
                    return delegate

                # Important: Await and assign the coroutine before tool construction
                delegate_coroutine = await make_delegate(url, tool_name)

                tool = StructuredTool.from_function(
                    name=f"{tool_name}_via_{agent_name}",
                    description=f"[Remote] {tool_description}",
                    args_schema=InputModel,
                    coroutine=delegate_coroutine
                )
                delegated_tools.append(tool)
                print(f"✅ Wrapped remote tool: {tool.name}")
            except Exception as e:
                print(f"⚠️ Could not wrap tool {tool_name} from {url}: {e}")

        # Also create a delegation tool (explicit peer task delegation)
        delegation_tool = StructuredTool.from_function(
            name=f"delegate_to_{agent_name}",
            description=f"Delegate task directly to {agent_name} at {url}",
            args_schema=DelegateToPeerSchema,
            coroutine=lambda **kwargs: delegate_task_to_peer_agent(peer_agent_url=url, **kwargs)
        )
        delegated_tools.append(delegation_tool)

    return delegated_tools

embedding = OpenAIEmbeddings()

vector_store = InMemoryVectorStore(embedding=embedding)

async def load_all_tools():
    """Async function to load tools from different MCP services and local files."""
    print("🚨 COMPREHENSIVE TOOL DISCOVERY STARTING 🚨")

    tool_services = [
        ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("github-mcp", ["node", "dist/index.js"], "list_tools", "call_tool"),
        # ("google-maps-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("sequentialthinking-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("slack-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("excalidraw-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("filesystem-mcp", ["node", "/app/dist/index.js", "/projects"], "tools/list", "tools/call"),
        # ("netbox-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("google-search-mcp", ["node", "/app/build/index.js"], "tools/list", "tools/call"),
        # ("servicenow-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("pyats-mcp", ["python3", "pyats_mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("email-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),
        # ("chatgpt-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("quickchart-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),
        # ("vegalite-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("mermaid-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("rfc-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),    
        # ("nist-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),        
    ]

    try:
        # Run docker ps to verify containers
        docker_ps_result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        print(docker_ps_result.stdout)

        service_discoveries = {}
        local_tools_lists  = await asyncio.gather(
            *[get_tools_for_service(service, command, discovery_method, call_method, service_discoveries)
              for service, command, discovery_method, call_method in tool_services]
        )
    
        all_tools = []
        for tools_list in local_tools_lists :
            all_tools.extend(tools_list)
    
        # ✅ Peer discovery inside this function
        peer_agents = {}
        for url in A2A_PEER_AGENTS:
            url = url.strip()
            if not url:
                continue
            agent = await discover_agent(url)
            if agent:
                peer_agents[url] = agent
                print(f"✅ Discovered peer: {url}")
            else:
                print(f"⚠️ Failed peer discovery: {url}")

        # ✅ Load delegated tools separately
        delegated_tools = await load_delegated_tools(peer_agents)

        # ✅ Finalize tool sets
        local_tools = []
        for tools_list in local_tools_lists:
            local_tools.extend(tools_list)

        all_tools = local_tools + delegated_tools + [a2a_delegation_tool]

        # ✅ Index only local tools for tool selection
        tool_documents = [
            Document(
                page_content=f"Tool name: {tool.name}. Tool purpose: {tool.description}",
                metadata={"tool_name": tool.name}
            )
            for tool in local_tools if hasattr(tool, "description")
        ]
        vector_store.add_documents(tool_documents)

        return all_tools, local_tools

    except Exception as e:
        print(f"❌ CRITICAL TOOL DISCOVERY ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

# Load tools
all_tools, local_tools = asyncio.run(load_all_tools())


def format_tool_descriptions(tools: List[Tool]) -> str:
    return "\n".join(
        f"- `{tool.name}`: {tool.description or 'No description provided.'}"
        for tool in tools
    )

tool_documents = [
    Document(
        page_content=f"Tool name: {tool.name}. Tool purpose: {tool.description}",
        metadata={"tool_name": tool.name}
    )
    for tool in all_tools if hasattr(tool, "description")
]

document_ids = vector_store.add_documents(tool_documents)

print("🔧 All bound tools:", [t.name for t in all_tools])

AGENT_CARD_OUTPUT_DIR = os.getenv("AGENT_CARD_OUTPUT_DIR", "/a2a/.well-known")
AGENT_CARD_PATH = os.path.join(AGENT_CARD_OUTPUT_DIR, "agent.json")

# Environment variables or defaults
AGENT_NAME = os.getenv("A2A_AGENT_NAME", "Selector Agent")
AGENT_DESCRIPTION = os.getenv("A2A_AGENT_DESCRIPTION", "LangGraph-based MCP agent for Selector AI and other MCPs.")
AGENT_HOST = os.getenv("A2A_AGENT_HOST", "70.53.207.50")
AGENT_PORT = os.getenv("A2A_AGENT_PORT", "10000")

AGENT_URL = f"http://{AGENT_HOST}:{AGENT_PORT}"

# ✅ Use standards-compliant fields
agent_card = {
    "name": AGENT_NAME,
    "description": AGENT_DESCRIPTION,
    "version": "1.0",
    "url": AGENT_URL,
    "endpoint": AGENT_URL,  # ✅ Essential for downstream routing
    "methods": {
        "send": f"{AGENT_URL}/"  # ✅ A2A-compatible 'send' route
    },
    "capabilities": {
        "a2a": True,
        "tool-use": True,
        "chat": True
    },
    "skills": []
}

# Populate skills from your discovered tools
for tool in local_tools:
    skill = {
        "id": tool.name,  
        "name": tool.name,
        "description": tool.description or "No description provided.",
    }

    if hasattr(tool, "args_schema") and tool.args_schema:
        try:
            skill["parameters"] = tool.args_schema.schema()
        except Exception:
            skill["parameters"] = {"type": "object", "properties": {}}

    agent_card["skills"].append(skill)

os.makedirs(AGENT_CARD_OUTPUT_DIR, exist_ok=True)
with open(AGENT_CARD_PATH, "w") as f:
    json.dump(agent_card, f, indent=2)

print(f"✅ A2A agent card written to {AGENT_CARD_PATH}")
print(f"🌐 Agent is reachable at: {AGENT_URL}")
print("DEBUG: Listing contents of AGENT_CARD_OUTPUT_DIR")
print(os.listdir(AGENT_CARD_OUTPUT_DIR))
print("DEBUG: Full absolute path check:", os.path.abspath(AGENT_CARD_PATH))

#llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)
llm = ChatOpenAI(model_name="gpt-4o", temperature="0.1")

llm_with_tools = llm.bind_tools(all_tools)

def format_tool_descriptions(tools: List[Tool]) -> str:
    """Formats the tool descriptions into a string."""
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


@traceable
class ContextAwareToolNode(ToolNode):
    """
    A specialized ToolNode that handles tool execution and updates the graph state
    based on the tool's response. It assumes that tools return a dictionary.
    """

    async def ainvoke(
        self, state: GraphState, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> GraphState:
        """
        Executes the tool call specified in the last AIMessage and updates the state.
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            # No tool calls in the last message, or it's not an AIMessage
            # This might happen if the assistant decided not to call a tool
            # Decide how to handle this - maybe just pass through or raise specific error
            logger.warning("ContextAwareToolNode: Last message is not an AIMessage with tool_calls.")
            # Depending on your graph logic, you might just return the state
            # or raise a more specific error if this state is unexpected.
            # Let's assume for now it should proceed to 'handle_tool_results' which returns to assistant
            return {"messages": messages, "context": state.get("context", {}), "__next__": "handle_tool_results"}


        tool_calls = last_message.tool_calls
        context = state.get("context", {})
        used = set(context.get("used_tools", []))
        tool_messages = [] # Store new tool messages here

        logger.info(f"🛠️ Processing {len(tool_calls)} tool calls from AIMessage {last_message.id}")
        for i, tool_call in enumerate(tool_calls):
            logger.info(f"  -> Processing tool_call #{i+1}")

            # --- FIX: Check structure before access ---
            if not isinstance(tool_call, dict) or 'name' not in tool_call or 'args' not in tool_call or 'id' not in tool_call:
                # LangChain ToolCall objects are dict-like, but let's be safe
                # Or if it IS a dict but missing keys (original error case)
                logger.error(f"❌ Invalid tool_call structure found: Type={type(tool_call)}, Value={repr(tool_call)}. Skipping.")
                # Create an error message to send back to the LLM
                # We need an ID. If it's missing, generate a placeholder or skip. Let's try getting it if possible.
                error_tool_call_id = getattr(tool_call, 'id', f"invalid_call_{i}") if not isinstance(tool_call, dict) else tool_call.get('id', f"invalid_call_{i}")
                tool_messages.append(ToolMessage(
                    tool_call_id=error_tool_call_id,
                    content=f"Error: Invalid tool call structure received: {repr(tool_call)}",
                    # Provide a dummy name if unavailable
                    name=getattr(tool_call, 'name', 'unknown_tool') if not isinstance(tool_call, dict) else tool_call.get('name', 'unknown_tool')
                ))
                continue # Skip this malformed call
            # --- End FIX ---

            tool_name = tool_call['name'] # Use dictionary access now that we know it's dict-like
            tool_args = tool_call['args']
            # Normalize tool arguments for known patterns
            if tool_name == "send-email":
               # Normalize 'body' to 'text'
               if "body" in tool_args and "text" not in tool_args:
                   tool_args["text"] = tool_args.pop("body")
            tool_id = tool_call['id']
            logger.info(f"     Tool Name: {tool_name}")
            logger.debug(f"     Tool Args: {tool_args}")
            logger.debug(f"     Tool ID: {tool_id}")

            if not (tool := self.tools_by_name.get(tool_name)):
                logger.warning(
                    f"Tool '{tool_name}' requested by LLM not found in available tools. Skipping."
                )
                tool_messages.append(ToolMessage(
                    tool_call_id=tool_id,
                    content=f"Error: Tool '{tool_name}' not found.",
                    name=tool_name,
                ))
                continue

            # Filter out None values AFTER getting the arguments
            filtered_tool_input = {k: v for k, v in tool_args.items() if v is not None}
            logger.debug(f"Calling tool: {tool.name} with filtered args: {filtered_tool_input}")

            try:
                # Execute the tool
                tool_response = await tool.ainvoke(filtered_tool_input, config=config) # Pass config

                logger.debug(f"Raw tool response for {tool.name}: Type={type(tool_response)}, Value={repr(tool_response)}")

                # --- Start Refined Logic ---
                # First, check if tool_response is already a string (e.g., from an error during the call)
                if isinstance(tool_response, str):
                    try:
                        # Try parsing it as JSON in case the tool returned a JSON string error
                        tool_data = json.loads(tool_response)
                    except json.JSONDecodeError:
                        # If it's not JSON, use the string directly (might be a simple error message)
                        tool_data = tool_response
                        logger.debug(f"Tool response is a non-JSON string for {tool.name}: {tool_data}")
                else:
                    # Assume it's likely a dict if not a string
                    tool_data = tool_response

                response_content = "" # Default

                if isinstance(tool_data, dict):
                    # Check for standard success structure
                    if tool_data.get("status") == "completed" and "output" in tool_data:
                        output_data = tool_data["output"]
                        # ** Specific handling for ask_selector's expected output **
                        if tool_name == "ask_selector" and isinstance(output_data, dict) and "content" in output_data:
                            response_content = output_data["content"] # Extract the natural language answer
                            logger.debug(f"Extracted NL content for {tool_name}: {response_content[:100]}...") # Log snippet
                        else:
                            # General case for other tools or if ask_selector format changes
                            response_content = json.dumps(output_data) # Serialize just the 'output' part
                            logger.debug(f"Serialized output part for {tool_name}: {response_content[:100]}...")
                    # Check for standard error structure
                    elif "error" in tool_data:
                        error_info = tool_data['error']
                        response_content = f"Tool Error: {json.dumps(error_info)}"
                        logger.debug(f"Serialized error part for {tool_name}: {response_content[:100]}...")
                    else:
                        # Fallback if structure is unexpected
                        response_content = json.dumps(tool_data)
                        logger.debug(f"Serialized whole unexpected dict for {tool_name}: {response_content[:100]}...")
                elif isinstance(tool_data, str):
                     # If tool_data ended up being a string (e.g., simple error before JSON parsing)
                     response_content = tool_data
                     logger.debug(f"Using direct string response for {tool.name}: {response_content[:100]}...")
                else:
                    # If tool_response wasn't a dict or string originally
                    response_content = str(tool_data) # Stringify directly
                    logger.debug(f"Stringified unexpected type response for {tool.name}: {response_content[:100]}...")

                # Create ToolMessage with the refined content
                tool_messages.append(ToolMessage(
                    tool_call_id=tool_id,
                    content=response_content, # Use the refined content
                    name=tool_name,
                ))
                used.add(tool.name) # Add to used tools AFTER successful processing
                # --- End Refined Logic ---

            except Exception as e:
                # Keep existing error handling for exceptions during tool execution *itself*
                logger.error(f"🔥 Error executing/processing tool '{tool_name}': {e}", exc_info=True)
                tool_messages.append(ToolMessage(
                    tool_call_id=tool_id,
                    content=f"Error executing tool {tool_name}: {str(e)}",
                    name=tool_name,
                ))
    
        # Update state AFTER the loop
        context["used_tools"] = list(used)
        # Add all generated tool messages to the main message list
        new_messages = messages + tool_messages

        # Return state, rely on graph definition for next step ('handle_tool_results')
        return {"messages": new_messages, "context": context}

async def select_tools(state: GraphState):
    messages = state.get("messages", [])
    context = state.get("context", {})
    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_user_message:
        logger.warning("select_tools: No user message found.")
        state["selected_tools"] = []
        return {"messages": messages, "context": context}

    query = last_user_message.content
    selected_tool_names = []

    try:
        # Step 1: Vector search
        scored_docs = vector_store.similarity_search_with_score(query, k=35)

        # Step 2: Apply threshold with fallback
        threshold = 0.50
        relevant_docs = [doc for doc, score in scored_docs if score >= threshold]

        if not relevant_docs:
            logger.warning(f"⚠️ No tools above threshold {threshold}. Falling back to top 15 by score.")
            relevant_docs = [doc for doc, _ in scored_docs[:15]]

        logger.info(f"✅ Selected {len(relevant_docs)} tools after filtering/fallback.")

        # Step 3: Build tool info for LLM
        tool_infos = {
            doc.metadata["tool_name"]: doc.page_content
            for doc in relevant_docs if "tool_name" in doc.metadata
        }

        if not tool_infos:
            logger.warning("select_tools: No valid tool_name metadata found.")
            state["selected_tools"] = []
            return {"messages": messages, "context": context}

        # Log top tools and scores for debugging
        logger.info("Top tools with scores:")
        for doc, score in scored_docs[:10]:
            if "tool_name" in doc.metadata:
                logger.info(f"- {doc.metadata['tool_name']}: {score}")

        tool_descriptions_for_prompt = "\n".join(
            f"- {name}: {desc}" for name, desc in tool_infos.items()
        )

        # Step 4: LLM refinement
        tool_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise Tool Selector Assistant. Your task is to choose the most relevant tools from the provided list to fulfill the user's request.

Consider these guidelines:
- Match tools to the *exact* user intent.
- Refer to tool descriptions to understand their purpose.
- Prefer specific tools over general ones if applicable.
- If multiple tools seem relevant for sequential steps *explicitly requested*, list them.
- If no tool is a good fit, output "None".
- Output *only* a comma-separated list of the chosen tool names (e.g., tool_a,tool_b) or the word "None"."""),

            ("human", "User request:\n---\n{query}\n---\n\nAvailable tools:\n---\n{tools}\n---\n\nBased *only* on the tools listed above, which are the best fit for the request? Output only the comma-separated tool names or 'None'.")
        ])

        selection_prompt_messages = tool_prompt.format_messages(
            query=query,
            tools=tool_descriptions_for_prompt
        )

        logger.info("🤖 Invoking LLM for tool selection...")
        tool_selection_response = await llm.ainvoke(selection_prompt_messages)
        raw_selection = tool_selection_response.content.strip()

        logger.info(f"📝 LLM raw tool selection: '{raw_selection}'")

        if raw_selection.lower() == "none" or not raw_selection:
            selected_tool_names = []
        else:
            potential_names = [name.strip() for name in raw_selection.split(',')]

            # 🔧 Normalize selection for delegated tools (e.g., ask_selector → ask_selector_via_xxx)
            normalized_tool_names = {}
            for name in tool_infos.keys():
                base_name = name.split("_via_")[0] if "_via_" in name else name
                normalized_tool_names[base_name] = name  # Always map shortest name → full name

            # Map LLM-chosen names to full tool names
            selected_tool_names = [
                normalized_tool_names.get(name, name)
                for name in potential_names
                if name in normalized_tool_names
            ]

            if len(selected_tool_names) != len(potential_names):
                logger.warning(f"⚠️ LLM selected invalid tools: {set(potential_names) - set(selected_tool_names)}")

    except Exception as e:
        logger.error(f"🔥 Error during tool selection: {e}", exc_info=True)
        selected_tool_names = []

    # Final: Update context
    context["selected_tools"] = list(set(context.get("selected_tools", [])) | set(selected_tool_names))
    logger.info(f"✅ Final selected tools: {context['selected_tools']}")
    return {
        "messages": messages,
        "context": context
    }


system_msg = """You are a computer networking expert at the CCIE level. You are a precise and helpful assistant with access to a wide range of tools for networking, GitHub automation, Slack notifications, file system operations, and ServiceNow ticketing. You must follow strict guidelines before choosing and using tools.

AVAILABLE TOOL CATEGORIES:
{tool_descriptions}

📌 TOOL USAGE GUIDELINES:

GENERAL RULES:
1. THINK step-by-step about what the user wants.
2. MATCH tools to the *exact* user intent.
3. DO NOT guess. Only use tools when the user explicitly requests an action that matches the tools purpose.
4. NEVER call a tool without all required parameters.
5. NEVER call a tool just because the output of another tool suggests a next step — unless the user explicitly asked for that.

🔄 **AFTER A TOOL RUNS:**
- When you receive information back from a tool in a `ToolMessage`, your **only** goal is to synthesize this information into a final, natural language answer for the user.
- Present the key findings from the `ToolMessage` clearly and concisely.
- **CRITICAL:** Do **NOT** repeat your decision to call the tool. Do **NOT** explain that you will use the tool again.
- **CRITICAL:** Do **NOT** include `executable_code` blocks or `tool_code` blocks in your final synthesized answer to the user unless specifically asked to generate code. Focus on the natural language explanation.


✅ WHEN TO USE SELECTOR TOOLS:

🤖 SELECTOR TOOLS:
- Use `ask_selector` as the **default tool** for general user queries in natural language. This includes requests for network summaries, health overviews, alert insights, or when the user provides instructions like “check the status of my environment” or “summarize network issues.”
- Use `query_selector` **only** when the user provides a **valid Selector command string** (typically starting with `#`, like `#alerts.recent` or `#device.status.all`). This tool bypasses natural language processing and runs the exact query directly.
- Use `get_selector_phrases` when the user explicitly asks to “list phrases,” “show available aliases,” or “see available commands.” This tool is for discovering registered NL phrases, not executing them.

*** IF you need to find the best way to ask selector you can use the get selector phrases to get the list of supported natural language phrases and then use the ask selector to ask the question.**

🧠 PYATS NETWORK AUTOMATION TOOLS:
- Use `pyATS_show_running_config`, `pyATS_run_show_command`, `pyATS_ping_from_network_device`, or `pyATS_configure_device` ONLY if the user requests network validation, inspection, or configuration of Cisco-style network devices.
- Do NOT use these tools for cloud or filesystem tasks.

📁 FILESYSTEM TOOLS:
- Use `write_file`, `edit_file`, `read_file`, or `create_directory` when the user asks to **create, modify, save, or read from files** in a local or mounted directory.
- Example: “Save the config to a markdown file” → `write_file`

🐙 GITHUB TOOLS:
- Use GitHub tools ONLY when the user explicitly asks to:
  - Push files
  - Create or update code or documentation in a repo
  - Open or manage GitHub issues or PRs
- Required for all GitHub actions: `owner`, `repo`, `branch`, and `commit message`
- NEVER use GitHub tools for local file management or Slack-style notifications.

💬 SLACK TOOLS:
- Use `slack_post_message`, `slack_reply_to_thread`, or `slack_add_reaction` only when the user asks to send messages to a Slack channel or thread.
- Example: “Notify the team” or “Send a message to #NOC” → `slack_post_message`

🗺️ MAPS TOOLS:
- Use `maps_geocode`, `maps_elevation`, etc., ONLY when the user asks for location-based data.
- NEVER use for IP addresses or configs.

📐 DIAGRAMMING TOOLS:
- Use `create_drawing`, `update_drawing`, `export_to_json` only when the user wants a network diagram or visual model.
- Do NOT export a drawing unless the user explicitly says so.

🧜 MERMAID DIAGRAM TOOLS:
- Use `mermaid_generate` ONLY when the user asks to create a PNG image from **Mermaid diagram code**.
  - **Purpose**: Converts Mermaid diagram code text into a PNG image file.
  - **Parameters:**
    - `code` (string): The Mermaid diagram code to render (required).
    - `theme` (string, optional): Theme for the diagram. Options: default, forest, dark, neutral. Defaults to default.
    - `backgroundColor` (string, optional): Background color for the generated PNG, e.g., white, transparent, #F0F0F0. Defaults to transparent or theme-based.
    - `name` (string): The filename for the generated PNG image (e.g., network_topology.png). **Required only if the tools environment is configured to save files to disk (CONTENT_IMAGE_SUPPORTED=false).**
    - `folder` (string): The absolute path *inside the container* where the image should be saved (e.g., /output). **Required only if the tools environment is configured to save files to disk (CONTENT_IMAGE_SUPPORTED=false).**
  - **Behavior Note:** This tools behavior depends on the `CONTENT_IMAGE_SUPPORTED` environment variable of the running container.
    - If `true` (default): The PNG image data is returned directly in the API response. `name` and `folder` parameters are ignored.
    - If `false`: The PNG image is saved to the specified `folder` with the specified `name`. The API response will contain the path to the saved file (e.g., /output/network_topology.png). `name` and `folder` parameters are **mandatory** in this mode.
    
🛠️ SERVICE NOW TOOLS:
- ONLY use ServiceNow tools if the user explicitly says things like:
  - “Create a problem ticket in ServiceNow”
  - “Get the state of a ServiceNow problem”
  - if asked to create a problem in service now - only call the create service now problem tool; not the other service now problem tools. You only need 1 tool to create a problem.
- NEVER use ServiceNow tools to write files, notify teams, or log internal info.
- NEVER assume a ServiceNow ticket is needed unless the user says so.
- ⚠️ If the user does NOT mention “ServiceNow” or “ticket,” DO NOT CALL ANY ServiceNow tool.

📧 EMAIL TOOLS:
- Use email tools (like `email_send_message`) ONLY when the user explicitly asks to send an email.
- Examples: "Send an email to team@example.com with the results", "Email the configuration to the network admin".
- Required: Recipient email address(es), subject line, and the body content for the email.
- Specify clearly who the email should be sent to and what information it should contain.
- DO NOT use email tools for Slack notifications, saving files, or internal logging unless specifically instructed to email that information.

🤖 CHATGPT ANALYSIS TOOLS:
- Use the `ask_chatgpt` tool ONLY when the user explicitly asks you to leverage an external ChatGPT model for specific analysis, summarization, comparison, or generation tasks that go beyond your primary function or require a separate perspective.
- Examples: "Analyze this Cisco config for security best practices using ChatGPT", "Ask ChatGPT to summarize this document", "Get ChatGPTs explanation for this routing behavior".
- Required: The `content` (e.g., configuration text, document snippet, specific question) that needs to be sent to the external ChatGPT tool.
- Clearly state *why* you are using the external ChatGPT tool (e.g., "To get a detailed security analysis from ChatGPT...").
- Do NOT use this tool for tasks you are expected to perform directly based on your core instructions or other available tools (like running a show command or saving a file). Differentiate between *your* analysis/response and the output requested *from* the external ChatGPT tool.

📊 VEGALITE VISUALIZATION TOOLS (Requires 2 Steps: Save then Visualize):
- Use these tools to create PNG charts from structured data (like parsed command output) using the Vega-Lite standard.

1.  **vegalite_save_data**
    - **Purpose**: Stores structured data under a unique name so it can be visualized later. This MUST be called *before* vegalite_visualize_data.
    - **Parameters**:
        - name (string): A unique identifier for this dataset (e.g., R1_interface_stats, packet_comparison). Choose a descriptive name.
        - data (List[Dict]): The actual structured data rows, formatted as a list of dictionaries. **CRITICAL: Ensure this data argument contains the *actual, non-empty* data extracted from previous steps (like pyATS output). Do NOT pass empty lists or lists of empty dictionaries.**
    - **Returns**: Confirmation that the data was saved successfully.

2.  **vegalite_visualize_data**
    - **Purpose**: Generates a PNG image visualization from data previously saved using vegalite_save_data. It uses a provided Vega-Lite JSON specification *template* and saves the resulting PNG to the /output directory.
    - **Parameters**:
        - data_name (string): The *exact* unique name that was used when calling vegalite_save_data.
        - vegalite_specification (string): A valid Vega-Lite v5 JSON specification string that defines the desired chart (marks, encodings, axes, etc.). **CRITICAL: This JSON string MUST NOT include the top-level data key.** The tool automatically loads the data referenced by data_name and injects it. The encodings within the spec (e.g., field, packets) must refer to keys present in the saved data.
    - **Returns**: Confirmation message including the container path where the PNG file was saved (e.g., /output/R1_interface_stats.png).

📈 QUICKCHART TOOLS (Generates Standard Chart Images/URLs):
- Use these tools for creating common chart types (bar, line, pie, etc.) using the QuickChart.io service. This requires constructing a valid Chart.js configuration object.

1.  **generate_chart**
    - **Purpose**: Creates a chart image hosted by QuickChart.io and returns a publicly accessible URL to that image. Use this when the user primarily needs a *link* to the visualization.
    - **Parameters**:
        - chart_config (dict or JSON string): A complete configuration object following the **Chart.js structure**. This object must define the chart type (e.g., bar, line, pie), the data (including labels and datasets with their values), and any desired options. Refer to Chart.js documentation for details on structuring this object. **CRITICAL: You must construct the full, valid Chart.js configuration based on the users request and available data.**
    - **Returns**: A string containing the URL pointing to the generated chart image.

2.  **download_chart**
    - **Purpose**: Creates a chart image using QuickChart.io and saves it directly as an image file (e.g., PNG) to the /output directory on the server. Use this when the user explicitly asks to **save the chart as a file**.
    - **Parameters**:
        - chart_config (dict or JSON string): The *same* complete Chart.js configuration object structure required by generate_chart. It defines the chart type, data, and options. **CRITICAL: You must construct the full, valid Chart.js configuration.**
        - file_path (string): The desired filename for the output image within the /output directory (e.g., interface_pie_chart.png, device_load.png). The tool automatically saves to the /output path.
    - **Returns**: Confirmation message including the container path where the chart image file was saved (e.g., /output/interface_pie_chart.png).

📜 RFC DOCUMENT TOOLS:
- Use `get_rfc`, `search_rfcs`, or `get_rfc_section` ONLY when the user explicitly asks to find, retrieve, or examine Request for Comments (RFC) documents.
- **Trigger Examples**:
    - Search for RFCs about HTTP/3 → `search_rfcs`
    - Get RFC 8446 or Show me the document for RFC 8446 → `get_rfc`
    - What's the metadata for RFC 2616?" → `get_rfc` with `format=metadata`
    - Find section 4.2 in RFC 791 or Get the 'Security Considerations' section of RFC 3550 → `get_rfc_section`
- **Constraints**:
    - Requires the specific RFC `number` for `get_rfc` and `get_rfc_section`.
    - Requires a `query` string for `search_rfcs`.
    - For `get_rfc_section`, requires a `section` identifier (title or number).
    - Do NOT use these tools for general web searches, code lookup, configuration files, or non-RFC standards documents. ONLY use for retrieving information directly related to official RFCs.

🛡️ NIST CVE VULNERABILITY TOOLS:
- Use `get_cve` or `search_cve` ONLY when the user explicitly asks to find or retrieve information about Common Vulnerabilities and Exposures (CVEs) from the NIST National Vulnerability Database (NVD).
- **Trigger Examples**:
    - Get details for CVE-2021-44228 or Tell me about the Log4Shell vulnerability CVE-2021-44228 → `get_cve` with `cve_id=CVE-2021-44228`
    - Search the NVD for vulnerabilities related to Apache Struts → `search_cve` with `keyword="Apache Struts"`
    - Find CVEs mentioning 'Microsoft Exchange Server' exactly → `search_cve` with `keyword="Microsoft Exchange Server"` and `exact_match=True`
    - Give me a concise summary of CVE-2019-1010218 → `get_cve` with `cve_id="CVE-2019-1010218"` and `concise=True`
    - Show me the latest 5 vulnerabilities for 'Cisco IOS XE' → `search_cve` with `keyword=Cisco IOS XE` and `results=5`
- **Constraints**:
    - Requires a valid CVE ID (e.g., `CVE-YYYY-NNNNN`) for `get_cve`.
    - Requires a `keyword` string for `search_cve`.
    - Use the `concise` parameter only if the user asks for summary information.
    - Use the `exact_match` parameter for `search_cve` only if the user specifies needing an exact phrase match.
    - Do NOT use these tools for general security advice, threat hunting outside of NVD, retrieving non-CVE vulnerability info, or fetching software patches. They are ONLY for interacting with the NIST NVD CVE database.

🎯 TOOL CHAINING:
- Do NOT chain tools together unless the user clearly describes multiple steps.
  - Example: “Save the config to GitHub and notify Slack” → You may use two tools.
- Otherwise, assume single-tool usage unless explicitly stated.

🧠 BEFORE YOU ACT:
- Pause and explain your thought process.
- Say WHY the tool youre selecting is the best fit.
- If unsure, respond with a clarification question instead of calling a tool.

🤝 A2A PEER AGENT DELEGATION TOOL:
- Use the `delegate_task_to_peer_agent` tool ONLY in the following situations:
    1. You determine you lack the necessary local tool to fulfill a specific user request (e.g., asked about a system you don't manage).
    2. A local tool execution fails with an error indicating the target is not found or not managed by this agent.
    3. The user explicitly asks you to consult or delegate the task to another specific agent.
- **Required Parameters:**
    - `peer_agent_url`: The full base URL of the agent to contact (e.g., "http://agent2-hostname:10001"). You might need to ask the user for this if it's not obvious or provided. **Known Peer Agents: [List known peer agent URLs here if applicable, e.g., Agent2 at http://<agent2-ip-or-hostname>:<port>]**
    - `task_description`: The precise question or task description to send to the peer agent. Usually, this will be the relevant part of the user's original request.
- **Example:** If the user asks "Ask selector device health for device S3" and your local `pyATS_...` tool fails because S3 is unknown, you should explain this and then call `delegate_task_to_peer_agent` with `peer_agent_url='http://<agent2_url>'` and `task_description='Ask selector device health for device S3'`.
- **DO NOT** use this tool for tasks you *can* perform locally.

"""


@traceable
async def assistant(state: GraphState):
    """Handles assistant logic and LLM interaction, with support for sequential tool calls."""
    messages = state.get("messages", [])
    context = state.get("context", {})
    selected_tool_names = context.get("selected_tools", [])
    run_mode = context.get("run_mode", "start")
    used = set(context.get("used_tools", []))

    # If selected_tool_names is empty, fall back to ALL tools not already used
    if selected_tool_names:
        tools_to_use = [
            tool for tool in all_tools
            if tool.name in selected_tool_names and tool.name not in used
        ]
    else:
        tools_to_use = [
            tool for tool in all_tools
            if tool.name not in used
        ]

    # If we're in continuous mode, don't re-select tools
    if run_mode == "continue":
        last_tool_message = None
        # Find the last tool message
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break

        if last_tool_message:
            # Add the tool message to ensure proper conversation context
            new_messages = [SystemMessage(content=system_msg)] + messages

            llm_with_tools = llm.bind_tools(tools_to_use)
            response = await llm_with_tools.ainvoke(new_messages, config={"tool_choice": "auto"})

            if hasattr(response, "tool_calls") and response.tool_calls:
                # Continue using tools
                return {"messages": [response], "context": context, "__next__": "tools"}
            else:
                # No more tools to use, return to user
                return {"messages": [response], "context": context, "__next__": "__end__"}

    # Initial processing or starting a new sequence
    llm_with_tools = llm.bind_tools(tools_to_use)
    formatted_tool_descriptions = format_tool_descriptions(tools_to_use)
    formatted_system_msg = system_msg.format(tool_descriptions=formatted_tool_descriptions)
    new_messages = [SystemMessage(content=formatted_system_msg)] + messages

    try:
        logger.info(f"assistant: Invoking LLM with new_messages: {new_messages}")
        # Always use auto tool choice to allow model to decide which tools to use
        response = await llm_with_tools.ainvoke(new_messages, config={"tool_choice": "auto"})
        logger.info(f"Raw LLM Response: {response}")

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        response = AIMessage(content=f"LLM Error: {e}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        context["run_mode"] = "continue"
        return {"messages": [response], "context": context, "__next__": "tools"}
    else:
        context["run_mode"] = "start"
        return {"messages": [response], "context": context, "__next__": "__end__"}

@traceable
async def handle_tool_results(state: GraphState):
    """Handles tool results and determines the next step."""
    messages = state.get("messages", [])
    context = state.get("context", {})

    # Always reset run_mode after tool execution
    context["run_mode"] = "start"

    return {
        "messages": messages,
        "context": context,
        "__next__": "assistant"  # Go back to the assistant to process tool results
    }

# Graph setup
graph_builder = StateGraph(GraphState)

# Define core nodes
graph_builder.add_node("select_tools", select_tools)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools", ContextAwareToolNode(tools=all_tools))
graph_builder.add_node("handle_tool_results", handle_tool_results)

# Define clean and minimal edges
# Start flow
graph_builder.add_edge(START, "select_tools")

# After tool selection, go to assistant
graph_builder.add_edge("select_tools", "assistant")

# Assistant decides: use tool or end
graph_builder.add_conditional_edges(
    "assistant",
    lambda state: state.get("__next__", "__end__"),
    {
        "tools": "tools",
        "__end__": END,
    }
)

# Tools always go to handler
graph_builder.add_edge("tools", "handle_tool_results")

# Tool results always return to assistant
graph_builder.add_edge("handle_tool_results", "assistant")

# Compile graph
compiled_graph = graph_builder.compile()

async def run_cli_interaction():
    """Runs the CLI interaction loop."""
    state = {"messages": [], "context": {"used_tools": []}}
    print("🛠️ Available tools:", [tool.name for tool in all_tools])
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        user_message = HumanMessage(content=user_input)
        state["messages"].append(user_message)
        state["context"]["used_tools"] = [] # Reset used tools for each new user turn

        print("🚀 Invoking graph...")
        result = await compiled_graph.ainvoke(state, config={"recursion_limit": 100})
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                print("Assistant:", message.content)
                break

if __name__ == "__main__":
    asyncio.run(run_cli_interaction())