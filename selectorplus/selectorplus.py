import os
import json
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
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableConfig

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
        elif json_type == "integer":
            field_type = int
        elif json_type == "number":
            field_type = float
        elif json_type == "boolean":
            field_type = bool
        elif json_type == "array":
            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "string":
                field_type = List[str]
            elif items_schema.get("type") == "integer":
                field_type = List[int]
            elif items_schema.get("type") == "number":
                field_type = List[float]
            elif items_schema.get("type") == "boolean":
                field_type = List[bool]
            elif items_schema.get("type") == "object":
                item_model = schema_to_pydantic_model(name + "_" + field_name + "_Item", items_schema)
                field_type = List[Any]
        elif json_type == "object":
            field_type = Dict[str, Any]
        else:
            field_type = Any

        if is_optional:
            field_type = Optional[field_type]

        namespace["__annotations__"][field_name] = field_type
        if field_name in required_fields:
            namespace[field_name] = Field(...)
        else:
            namespace[field_name] = Field(default=None)

    return type(name, (BaseModel,), namespace)


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
            normalized_args = arguments.copy() # Avoid modifying original dict

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
async def _base_mcp_call(tool_name_to_call: str, service_discovery_instance: MCPToolDiscovery, args_dict: dict):
    """Generic async function to call an MCP tool."""
    logger.info(f"PARTIAL_TRACE: _base_mcp_call invoked for '{tool_name_to_call}' with args {args_dict}")
    # Directly call the instance method
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


async def load_all_tools():
    """Async function to load tools from different MCP services and local files."""
    print("🚨 COMPREHENSIVE TOOL DISCOVERY STARTING 🚨")

    tool_services = [
        ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        ("github-mcp", ["node", "dist/index.js"], "list_tools", "call_tool"),
        ("google-maps-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("sequentialthinking-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("slack-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("excalidraw-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("filesystem-mcp", ["node", "/app/dist/index.js", "/projects"], "tools/list", "tools/call"),
        ("netbox-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        ("google-search-mcp", ["node", "/app/build/index.js"], "tools/list", "tools/call"),
        ("servicenow-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        ("pyats-mcp", ["python3", "pyats_mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        ("email-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),
        ("chatgpt-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
    ]

    try:
        # Run docker ps to verify containers
        docker_ps_result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        print(docker_ps_result.stdout)

        service_discoveries = {}

        # Gather tools from all services
        all_service_tools = await asyncio.gather(
            *[get_tools_for_service(service, command, discovery_method, call_method, service_discoveries)
              for service, command, discovery_method, call_method in tool_services]
        )

        # Add local tools
        print("🔍 Loading Local Tools:")
        local_tools = load_local_tools_from_folder("tools")
        print(f"🧰 Local Tools Found: {[tool.name for tool in local_tools]}")

        # Combine all tools
        all_tools = []
        for tools_list in all_service_tools:
            if tools_list:
                all_tools.extend(tools_list)
        all_tools.extend(local_tools)

        print("🔧 Comprehensive Tool Discovery Results:")
        print("✅ All Discovered Tools:", [t.name for t in all_tools])

        if not all_tools:
            print("🚨 WARNING: NO TOOLS DISCOVERED 🚨")
            print("Potential Issues:")
            print("1. Docker containers not running")
            print("2. Incorrect discovery methods")
            print("3. Network/communication issues")
            print("4. Missing tool configuration")

        return all_tools

    except Exception as e:
        print(f"❌ CRITICAL TOOL DISCOVERY ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

# Load tools
valid_tools = asyncio.run(load_all_tools())

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = InMemoryVectorStore(embedding=embedding)

tool_documents =[
    Document(
        page_content=f"Tool name: {tool.name}. Tool purpose: {tool.description}",
        metadata={"tool_name": tool.name}
    )
    for tool in valid_tools if hasattr(tool, "description")
]

document_ids = vector_store.add_documents(tool_documents)

print("🔧 All bound tools:", [t.name for t in valid_tools])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)

llm_with_tools = llm.bind_tools(valid_tools)

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

@traceable
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
            selected_tool_names = [name for name in potential_names if name in tool_infos]
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


system_msg = """You are a helpful file system and diagramming assistant.

*Available Tools:
{tool_descriptions}

IMPORTANT TOOL USAGE GUIDELINES:
1. GitHub tools require specific parameters:
    - For creating/updating files, you MUST include: owner, repo, path, content, branch, AND message (for commit message)
    - Example: create_or_update_file(owner="MyOrg", repo="MyRepo", path="file.md", content="Content", branch="main", message="Commit message")

IMPORTANT: When selecting a tool, follow these strict guidelines:
1. ALWAYS think step-by-step about what the user is asking for
2. ONLY use tools that match the user's exact intention
3. Do NOT call tools unless the user explicitly asks for it. Creating a drawing (via `create_drawing`) is a separate action from exporting it (e.g., `export_to_json`). Do NOT chain or follow up one with the other unless the user clearly requests it.
4. NEVER call a tool without all required parameters

THOUGHT PROCESS: Before taking any action, clearly explain your thought process and why you're choosing a specific tool.
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
            tool for tool in valid_tools
            if tool.name in selected_tool_names and tool.name not in used
        ]
    else:
        tools_to_use = [
            tool for tool in valid_tools
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
graph_builder.add_node("tools", ContextAwareToolNode(tools=valid_tools))
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