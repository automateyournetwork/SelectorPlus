import os
import json
import time
import asyncio
import inspect
import logging
import importlib
import subprocess
from functools import wraps
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.messages import BaseMessage
from langchain.tools import Tool, StructuredTool
from langgraph.graph.message import add_messages
from langchain_core.vectorstores import InMemoryVectorStore
from typing import Dict, Any, List, Optional, Union, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MessagesStateWithSelection = Dict[str, Union[List[BaseMessage], List[str]]]

def load_local_tools_from_folder(folder_path: str) -> List[Tool]:
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
    original_func = tool_obj.func

    @wraps(original_func)
    def wrapper(input_value):
        if isinstance(input_value, str):
            input_value = {"ip": input_value}
        elif isinstance(input_value, dict) and "ip" not in input_value:
            # You could log or raise a warning here if needed
            logger.warning(f"⚠️ Missing 'ip' key in dict: {input_value}")
        return original_func(input_value)

    return Tool(
        name=tool_obj.name,
        description=tool_obj.description,
        func=wrapper,
    )

def schema_to_pydantic_model(name: str, schema: dict):
    """
    Dynamically creates a Pydantic model class from a JSON Schema object.
    Compatible with Pydantic v2.
    """
    from typing import Any, List, Dict
    namespace = {"__annotations__": {}}

    if schema.get("type") != "object":
        raise ValueError("Only object schemas are supported.")

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")

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
                # Handle array of objects recursively
                item_model = schema_to_pydantic_model(name + "_" + field_name + "_Item", items_schema)
                field_type = List[item_model]
            else:
                field_type = List[Any]
        elif json_type == "object":
            field_type = Dict[str, Any]
        else:
            field_type = Any

        namespace["__annotations__"][field_name] = field_type
        if field_name in required_fields:
            namespace[field_name] = Field(...)
        else:
            namespace[field_name] = Field(default=None)

    return type(name, (BaseModel,), namespace)

class MessagesStateWithSelection(dict):
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    selected_tools: List[str]

    def __init__(self, messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = [],
                 selected_tools: List[str] = []):
        super().__init__(messages=messages, selected_tools=selected_tools)

class MCPToolDiscovery:
    def __init__(self, container_name: str, command: List[str], discovery_method: str = "tools/discover",
                 call_method: str = "tools/call"):
        self.container_name = container_name
        self.command = command
        self.discovery_method = discovery_method
        self.call_method = call_method
        self.discovered_tools = []

    async def discover_tools(self) -> List[Dict[str, Any]]:
        print(f"🔍 Discovering tools from container: {self.container_name}")
        print(f"🕵️ Discovery Method: {self.discovery_method}")

        try:
            discovery_payload = {
                "jsonrpc": "2.0",
                "method": self.discovery_method,
                "params": {},
                "id": "1"
            }
            print(f"Sending discovery payload: {discovery_payload}")  # added log
            command = ["docker", "exec", "-i", self.container_name] + self.command
            process = subprocess.run(
                command,
                input=json.dumps(discovery_payload) + "\n",
                capture_output=True,
                text=True,
            )
            stdout_lines = process.stdout.strip().split("\n")
            print("📥 Raw discovery response:", stdout_lines)
            if stdout_lines:
                last_line = None
                for line in reversed(stdout_lines):
                    if line.startswith("{") or line.startswith("["):
                        last_line = line
                        break
                if last_line:
                    try:
                        response = json.loads(last_line)
                        # Correctly handle both response structures
                        if "result" in response:
                            if isinstance(response["result"], list):
                                tools = response["result"]
                            elif isinstance(response["result"], dict) and "tools" in response["result"]:
                                tools = response["result"]["tools"]
                            else:
                                print("❌ Unexpected 'result' structure.")
                                return []
                        else:
                            tools = []  # Assuming no tools if result is not present
                        if tools:
                            print("✅ Discovered tools:", [tool["name"] for tool in tools])
                            return tools
                        else:
                            print("❌ No tools found in response.")
                            return []
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON Decode Error: {e}")
                        return []
                    else:
                        print("❌ No valid JSON response found.")
                        return []
                else:
                    print("❌ No response lines received.")
                    return []
        except Exception as e:
            print(f"❌ Error discovering tools: {e}")
            return []

    @traceable
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Enhanced tool calling method with comprehensive logging and error handling.
        """
        logger.info(f"🔍 Attempting to call tool: {tool_name}")
        logger.info(f"📦 Arguments: {arguments}")

        # Detailed logging of containers and network
        try:
            network_inspect = subprocess.run(
                ["docker", "network", "inspect", "bridge"],
                capture_output=True,
                text=True,
            )
            logger.info(f"🌐 Network Details: {network_inspect.stdout}")
        except Exception as e:
            logger.error(f"❌ Network inspection failed: {e}")

        command = ["docker", "exec", "-i", self.container_name] + self.command

        try:
            # Initialize normalized_args with the original arguments
            normalized_args = arguments

            payload = {
                "jsonrpc": "2.0",
                "method": self.call_method,
                "params": {"name": tool_name, "arguments": normalized_args},
                "id": "2",
            }

            logger.info(f"🚀 Full Payload: {json.dumps(payload)}")

            process = subprocess.run(
                command,
                input=json.dumps(payload) + "\n",
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            logger.info(f"🔬 Subprocess Exit Code: {process.returncode}")
            logger.info(f"🔬 Full subprocess stdout: {process.stdout}")
            logger.info(f"🔬 Full subprocess stderr: {process.stderr}")

            if process.returncode != 0:
                logger.error(f"❌ Subprocess returned non-zero exit code: {process.returncode}")
                logger.error(f"🚨 Error Details: {process.stderr}")
                return f"Subprocess Error: {process.stderr}"

            # Enhanced JSON parsing with fallback and explicit tool name checking
            output_lines = process.stdout.strip().split("\n")
            for line in reversed(output_lines):
                try:
                    response = json.loads(line)
                    logger.info(f"✅ Parsed JSON response: {response}")

                    if "result" in response:
                        return response["result"]
                    elif "error" in response:
                        error_message = response["error"]
                        if "tool not found" in str(error_message).lower():
                            logger.error(f"🚨 Tool '{tool_name}' not found by service.")
                            return f"Tool Error: Tool '{error_message}"
                    else:
                        logger.warning("⚠️ Unexpected response structure")
                        return response
                except json.JSONDecodeError:
                    continue

            logger.error("❌ No valid JSON response found")
            return "Error: No valid JSON response"

        except Exception:
            logger.critical(f"🔥 Critical tool call error", exc_info=True)
            return "Critical Error: tool call failure"

def load_mcp_tools() -> List[Tool]:
    """
    Asynchronously load tools from different MCP services.

    :return: Consolidated list of tools
    """

    async def gather_tools():
        tool_services = [
            ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
            ("github-mcp", ["node", "dist/index.js"], "list_tools", "call_tool"),
            ("google-maps-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
            ("sequentialthinking-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
            ("slack-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
            ("excalidraw-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
            ("filesystem-mcp", ["node", "/app/dist/index.js", "/projects"], "tools/list", "tools/call"),
            ("brave-search-mcp", ["node", "dist/index.js"], "tools/list", "tools/call")
        ]

        dynamic_tools = []
        for service, command in tool_services:
            discovery = MCPToolDiscovery(service, command)
            tools = await discovery.discover_tools()
            dynamic_tools.extend(tools)

        # Load local tools
        local_tools = load_local_tools_from_folder("tools")

        # Combine MCP and local tools
        all_tools = dynamic_tools + local_tools

        # Filter out None tools
        valid_tools = [t for t in all_tools if t is not None]

        print("🔧 All bound tools:", [t.name for t in valid_tools])

        return valid_tools

    # Run the async function and get tools
    return asyncio.run(gather_tools())

async def get_tools_for_service(service_name, command, discovery_method, call_method, service_discoveries):
    """Enhanced tool discovery for each service."""
    print(f"🕵️ Discovering tools for: {service_name}")
    discovery = MCPToolDiscovery(
        service_name,
        command,
        discovery_method=discovery_method,
        call_method=call_method
    )
    service_discoveries[service_name] = discovery  # Store the instance

    tools = []
    try:
        discovered_tools = await discovery.discover_tools()
        print(f"🛠️ Tools for {service_name}: {discovered_tools}")

        for tool in discovered_tools:
            tool_name = tool['name']
            tool_description = tool.get('description', '')
            tool_schema = tool.get('inputSchema', {}) or tool.get('parameters', {})  # Some servers call it 'parameters'

            logger.info(f"🔧 Processing tool '{tool_name}' from service '{service_name}'")
            logger.debug(f"📝 Raw tool data for '{tool_name}': {json.dumps(tool, indent=2)}")

            if tool_schema and tool_schema.get("type") == "object":
                logger.info(f"🧬 Found schema for tool '{tool_name}': {json.dumps(tool_schema, indent=2)}")

                try:
                    input_model = schema_to_pydantic_model(tool_name + "_Input", tool_schema)

                    async def tool_call_wrapper(**kwargs):
                        try:
                            validated_args = input_model(**kwargs).dict()
                            result = await service_discoveries[service_name].call_tool(tool_name, validated_args)
                            return result
                        except Exception as e:
                            logger.error(f"Tool call error: {e}")
                            return f"Tool call error: {e}"

                    structured_tool = StructuredTool.from_function(
                        name=tool_name,
                        description=tool_description,
                        args_schema=input_model,
                        func=lambda **kwargs: asyncio.run(tool_call_wrapper(**kwargs))  # added asyncio.run
                    )

                    tools.append(structured_tool)
                    logger.info(f"✅ Loaded StructuredTool: {tool_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to build StructuredTool for '{tool_name}': {e}")
            else:
                logger.info(f"📦 No schema found for '{tool_name}'. Falling back to generic Tool with __arg1")

                async def fallback_tool_call_wrapper(x):
                    try:
                        result = await service_discoveries[service_name].call_tool(tool_name, {"__arg1": x})
                        return result
                    except Exception as e:
                        logger.error(f"Fallback tool call error: {e}")
                        return f"Fallback tool call error: {e}"

                fallback_tool = Tool(
                    name=tool_name,
                    description=tool_description,
                    func=lambda x: asyncio.run(fallback_tool_call_wrapper(x))  # added asyncio.run
                )

                tools.append(fallback_tool)
                logger.info(f"✅ Loaded fallback Tool: {tool_name}")

    except Exception as e:
        logger.error(f"❌ Tool Discovery Error for {service_name}: {e}")
    finally:
        logger.info(f"🏁 Finished processing tools for service: {service_name}")
        return tools

async def load_all_tools():
    """Async function to load tools from all services with comprehensive logging."""
    print("🚨 COMPREHENSIVE TOOL DISCOVERY STARTING 🚨")

    tool_services = [
        ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        ("github-mcp", ["node", "dist/index.js"], "list_tools", "call_tool"),
        ("google-maps-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("sequentialthinking-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("slack-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("excalidraw-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("filesystem-mcp", ["node", "/app/dist/index.js", "/projects"], "tools/list", "tools/call"),
        ("brave-search-mcp", ["node", "dist/index.js"], "tools/list", "tools/call")
    ]

    try:
        # Run docker ps to verify containers
        docker_ps_result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        print(docker_ps_result.stdout)

        service_discoveries = {}

        # Gather tools from all services
        # 🛠️ Pass service_discoveries in the call
        all_service_tools = await asyncio.gather(
            *[get_tools_for_service(service, command, discovery_method, call_method, service_discoveries)
              for service, command, discovery_method, call_method in tool_services]
        )

        # Add local tools
        print("🔍 Loading Local Tools:")
        local_tools = load_local_tools_from_folder("tools")
        print(f"🧰 Local Tools Found: {[tool.name for tool in local_tools]}")

        # Define all_tools before using it
        all_tools = []

        # Combine all tools
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

# Use asyncio to run the async function and get tools
valid_tools = asyncio.run(load_all_tools())

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # or "text-embedding-005"
vector_store = InMemoryVectorStore(embedding=embedding)

tool_documents = [
    Document(page_content=tool.description or "", metadata={"tool_name": tool.name})
    for tool in valid_tools if hasattr(tool, "description")
]

document_ids = vector_store.add_documents(tool_documents)

print("🔧 All bound tools:", [t.name for t in valid_tools])

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

llm_with_tools = llm.bind_tools(valid_tools)

# System Message
system_msg = """You are a helpful file system and diagramming assistant.

IMPORTANT: When selecting a tool, follow these strict guidelines:
1. ALWAYS think step-by-step about what the user is asking for
2. ONLY use tools that match the user's exact intention
3. Do NOT call tools unless the user explicitly asks for it. Creating a drawing (via `create_drawing`) is a separate action from exporting it (e.g., `export_to_json`). Do NOT chain or follow up one with the other unless the user clearly requests it.
4. NEVER call a tool without all required parameters

THOUGHT PROCESS: Before taking any action, clearly explain your thought process and why you're choosing a specific tool.
"""

@traceable
def select_tools(state: MessagesStateWithSelection):
    messages = state.get("messages", [])
    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if last_user_message:
        query = last_user_message.content
        relevant_docs = vector_store.similarity_search(query, k=8)
        selected_tool_names = [doc.metadata["tool_name"] for doc in relevant_docs]
        logger.info(f"🔍 Tool selection for '{query}': {selected_tool_names}")

        return {
            "messages": messages,
            "selected_tools": selected_tool_names
        }

@traceable
def assistant(state: MessagesStateWithSelection):
    messages = state.get("messages", [])
    selected_tool_names = state.get("selected_tools", [])

    tools_to_use = [tool for tool in valid_tools if tool.name in selected_tool_names]
    if not tools_to_use:
        logger.warning("🤷 No tools selected, using all tools")
        tools_to_use = valid_tools

    llm_with_selected_tools = llm.bind_tools(tools_to_use)
    new_messages = [SystemMessage(content=system_msg)] + messages
    try:
        logger.info(f"assistant: Invoking LLM with new_messages: {new_messages}")  # CRUCIAL LOG
        response = llm_with_selected_tools.invoke(new_messages)
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        response = AIMessage(content=f"LLM Error: {e}")

    if response and hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"🛠️ Tool Calls Detected: {response.tool_calls}")
        logger.info(f"Response: {response}")
        return {"messages": [response]}
    elif any(isinstance(msg, ToolMessage) for msg in messages):
        logger.info("🔁 Processing follow-up after tool result")
        try:
            followup_messages = [SystemMessage(content=system_msg)] + messages
            logger.info(f"assistant: Invoking LLM for followup with followup_messages: {followup_messages}")  # CRUCIAL LOG
            followup = llm_with_selected_tools.invoke(followup_messages)
            if not isinstance(followup, AIMessage) or not followup.content:
                followup = AIMessage(content="LLM response to tool output was empty. Returning to user.")
        except Exception as e:
            logger.error(f"Error invoking LLM for followup: {e}", exc_info=True)
            followup = AIMessage(content=f"LLM Error: {e} while processing tool output. Returning to user.")
        return {"messages": [followup]}
    elif response and hasattr(response, "content") and response.content:
        logger.info(f"🧠 Assistant response: {response.content}")
        return {"messages": [response]}
    else:
        logger.warning("⚠️ Empty response from LLM")
        return {"messages": [AIMessage(content="No response from LLM. Returning to user.")]}
    
class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]

def tools_condition(state: State) -> str:
    messages = state.get("messages", [])
    if not messages:
        logger.info("tools_condition: No messages, going to START")
        return START

    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        if getattr(last_message, "tool_calls", None):
            logger.info("tools_condition: AIMessage with tool_calls → tools")
            return "tools"
        elif last_message.content:
            logger.info("tools_condition: AIMessage with content → __end__")
            return "__end__"
        else:
            logger.info("tools_condition: AIMessage without content → assistant")
            return "assistant"

    elif isinstance(last_message, ToolMessage):
        logger.info("tools_condition: ToolMessage → handle_tool_response")
        return "handle_tool_response"

    logger.info("tools_condition: Default → assistant")
    return "assistant"

@traceable
async def handle_tool_response(state: State):
    messages = state.get("messages", [])
    last_message = messages[-1]

    logger.info("Entering handle_tool_response node")
    logger.info(f"Input messages to handle_tool_response: {messages}")

    if isinstance(last_message, ToolMessage):
        tool_name = last_message.name
        tool_output = last_message.content

        logger.info(f"Handling tool response from {tool_name}: {tool_output}")

        try:
            # Parse the JSON output from the tool
            parsed_output = json.loads(tool_output)
            content = parsed_output.get("content", "").strip()

            if not content:
                logger.warning("⚠️ Tool response content is empty.")
                content = "The tool returned no useful content."

            # Build final AI message directly from tool output content
            llm_response = AIMessage(content=content)

            logger.info(f"✅ Returning parsed tool response as AIMessage: {llm_response.content}")
            return {"messages": [llm_response]}  # This lets tools_condition() exit cleanly to __end__

        except Exception as e:
            logger.error(f"❌ Error in handle_tool_response: {e}", exc_info=True)
            error_message = AIMessage(content=f"Tool call error: {e}. Returning to user.")
            return {"messages": [error_message], "error": True}

    else:
        logger.warning("handle_tool_response received non-ToolMessage")
        empty_response = AIMessage(content="Unexpected input to handle_tool_response. Returning to user.")
        return {"messages": [empty_response]}

    
graph_builder = StateGraph(State)
graph_builder.add_node("select_tools", select_tools)
graph_builder.add_node("assistant", assistant)
tool_node = ToolNode(tools=valid_tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("handle_tool_response", handle_tool_response)

graph_builder.add_conditional_edges(
    "assistant",
    tools_condition,
    path_map={
        "tools": "tools",
        "__end__": END
    }
)

graph_builder.add_edge("select_tools", "assistant")
graph_builder.add_edge("assistant", "tools")
graph_builder.add_edge("tools", "handle_tool_response")
graph_builder.add_edge(START, "select_tools")

compiled_graph = graph_builder.compile()
logger.info("🚀 Selector Plus LangGraph compiled successfully")

# CLI Loop
async def run_cli_interaction():
    state = {"messages": []}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        user_message = HumanMessage(content=user_input)
        state["messages"].append(user_message)

        print("🚀 Invoking graph...")
        result = await compiled_graph.ainvoke(state, config={"recursion_limit": 100})  # ainvoke for async
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and (not hasattr(message, "tool_calls") or not message.tool_calls):
                print("Assistant:", message.content)
                break


if __name__ == "__main__":
    asyncio.run(run_cli_interaction())