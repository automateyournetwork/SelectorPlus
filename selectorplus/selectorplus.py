import os
import json
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from typing import Dict, Any, List, Optional, Union, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import tools_condition, ToolNode
from langgraph.pregel.retry import RetryPolicy
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State class for LangGraph."""
    messages: Annotated[list[BaseMessage], add_messages]
    # selected_tools: list[str]  # Removed selected_tools
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
        """Discovers tools from the MCP container."""
        print(f"🔍 Discovering tools from container: {self.container_name}")
        print(f"🕵️ Discovery Method: {self.discovery_method}")

        try:
            discovery_payload = {
                "jsonrpc": "2.0",
                "method": self.discovery_method,
                "params": {},
                "id": "1"
            }
            print(f"Sending discovery payload: {discovery_payload}")
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
                        if "result" in response:
                            if isinstance(response["result"], list):
                                tools = response["result"]
                            elif isinstance(response["result"], dict) and "tools" in response["result"]:
                                tools = response["result"]["tools"]
                            else:
                                print("❌ Unexpected 'result' structure.")
                                return []
                        else:
                            tools = []
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
        """Calls a tool in the MCP container with logging and error handling."""
        logger.info(f"🔍 Attempting to call tool: {tool_name}")
        logger.info(f"📦 Arguments: {arguments}")

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
            normalized_args = arguments

            if tool_name == "create_or_update_file" and isinstance(normalized_args, dict) and "sha" in normalized_args and normalized_args["sha"] is None:
                del normalized_args["sha"]

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


async def get_tools_for_service(service_name, command, discovery_method, call_method, service_discoveries):
    """Enhanced tool discovery for each service."""
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
        discovered_tools = await discovery.discover_tools()
        print(f"🛠️ Tools for {service_name}: {[t['name'] for t in discovered_tools]}")

        for tool in discovered_tools:
            tool_name = tool["name"]
            tool_description = tool.get("description", "")
            tool_schema = tool.get("inputSchema") or tool.get("parameters", {})

            if tool_schema and tool_schema.get("type") == "object":
                try:
                    input_model = schema_to_pydantic_model(tool_name + "_Input", tool_schema)

                    async def tool_call_wrapper(**kwargs):
                        validated_args = input_model(**kwargs).dict()
                        return await service_discoveries[service_name].call_tool(tool_name, validated_args)

                    structured_tool = StructuredTool.from_function(
                        name=tool_name,
                        description=tool_description,
                        args_schema=input_model,
                        func=(lambda tool_name=tool_name, input_model=input_model:
                            lambda **kwargs: asyncio.run(
                                service_discoveries[service_name].call_tool(tool_name, input_model(**kwargs).dict())
                            ))()
                    )

                    tools.append(structured_tool)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to build structured tool {tool_name}: {e}")
            else:
                async def fallback_tool_call_wrapper(x):
                    return await service_discoveries[service_name].call_tool(tool_name, {"__arg1": x})

                fallback_tool = Tool(
                    name=tool_name,
                    description=tool_description,
                    func=lambda x: asyncio.run(fallback_tool_call_wrapper(x))
                )
                tools.append(fallback_tool)

    except Exception as e:
        logger.error(f"❌ Tool discovery error in {service_name}: {e}", exc_info=True)

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
        ("brave-search-mcp", ["node", "dist/index.js"], "tools/list", "tools/call")
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

def format_tool_descriptions(tools: List[Tool]) -> str:
    return "\n".join(
        f"- `{tool.name}`: {tool.description or 'No description provided.'}"
        for tool in tools
    )

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = InMemoryVectorStore(embedding=embedding)

tool_documents = [
    Document(page_content=tool.description or "", metadata={"tool_name": tool.name})
    for tool in valid_tools if hasattr(tool, "description")
]

document_ids = vector_store.add_documents(tool_documents)

print("🔧 All bound tools:", [t.name for t in valid_tools])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)
llm_with_tools = llm.bind_tools(valid_tools)

def format_tool_descriptions(tools: List[Tool]) -> str:
    """Formats the tool descriptions into a string."""
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


class ContextAwareToolNode(ToolNode):
    """
    A specialized ToolNode that handles tool execution and updates the graph state
    based on the tool's response.  It assumes that tools return a dictionary.
    """

    def invoke(self, state: GraphState) -> GraphState:
        """
        Executes the tool call specified in the last AIMessage and updates the state.

        Args:
            state: The current graph state.

        Returns:
            The updated graph state.

        Raises:
            ValueError: If the last message is not an AIMessage with tool calls.
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            raise ValueError("Expected an AIMessage with tool_calls")

        tool_calls = last_message.tool_calls
        context = state.get("context", {})

        for tool_call in tool_calls:
            tool = self.tools_by_name[tool_call.name]  # Corrected attribute access
            tool_input = tool_call.args

            logger.info(f"Tool Input (Before Filtering): {tool_input}")  # Logging for debugging

            # Filter out null values from tool_input
            filtered_tool_input = {k: v for k, v in tool_input.items() if v is not None}

            logger.info(f"Calling tool: {tool.name} with args: {filtered_tool_input}")
            tool_response = tool.invoke(filtered_tool_input)  # Execute the tool

            if not isinstance(tool_response, dict):
                raise ValueError(
                    f"Tool {tool.name} should return a dictionary, but returned {type(tool_response)}"
                )

            logger.info(f"Tool {tool.name} returned: {tool_response}")

            # Update the context with the tool's output
            context.update(tool_response)

            # Create a ToolMessage and add it to the message history
            tool_message = ToolMessage(
                tool_call_id=tool_call.id,
                content=tool_response.get("content", str(tool_response)),  # Ensure content is always a string
                name=tool_call.name,
            )
            messages.append(tool_message)

        return {"messages": messages, "context": context}
        

@traceable
def select_tools(state: GraphState):
    messages = state.get("messages", [])
    context = state.get("context", {})
    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_user_message:
        return {"messages": messages, "context": context}

    query = last_user_message.content
    relevant_docs = vector_store.similarity_search(query, k=8)

    # Gather tool names and descriptions
    tool_infos = [
        (doc.metadata["tool_name"], doc.page_content)
        for doc in relevant_docs
    ]

    # LLM prompt: choose best tool
    tool_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent tool selector. Based on the user request and the available tools, pick the name of the best matching tool ONLY. Do not explain. Just output the tool name."),
        ("human", "User request:\n{query}\n\nAvailable tools:\n{tools}\n\nPick the BEST tool name:")
    ])

    tool_descriptions = "\n".join(f"- {name}: {desc}" for name, desc in tool_infos)
    
    selection_prompt = tool_prompt.format_messages(query=query, tools=tool_descriptions)

    try:
        tool_selection_response = llm.invoke(selection_prompt)
        selected_tool_name = tool_selection_response.content.strip().split()[0]  # be strict
        logger.info(f"🔍 LLM selected tool: {selected_tool_name}")
    except Exception as e:
        logger.error(f"⚠️ Tool selection LLM failed: {e}")
        selected_tool_name = None

    if selected_tool_name:
        context["selected_tools"] = [selected_tool_name]
    else:
        context["selected_tools"] = [t[0] for t in tool_infos]  # fallback to vector results

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
def assistant(state: GraphState):
    """Handles assistant logic and LLM interaction, with support for sequential tool calls."""
    messages = state.get("messages", [])
    context = state.get("context", {})
    selected_tool_names = context.get("selected_tools", [])
    run_mode = context.get("run_mode", "start")

    tools_to_use = [tool for tool in valid_tools if tool.name in selected_tool_names] if selected_tool_names else valid_tools

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
            response = llm_with_tools.invoke(new_messages, config={"tool_choice": "auto"})

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
        response = llm_with_tools.invoke(new_messages, config={"tool_choice": "auto"})
        logger.info(f"Raw LLM Response: {response}")

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        response = AIMessage(content=f"LLM Error: {e}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        # Update context to indicate we're in a tool sequence
        context["run_mode"] = "continue"
        return {"messages": [response], "context": context, "__next__": "tools"}
    else:
        # Reset mode if no tools are called
        context["run_mode"] = "start"
        return {"messages": [response], "context": context, "__next__": "__end__"}

@traceable
def handle_tool_results(state: GraphState):
    """Handles tool results and determines if more tools should be used."""
    messages = state.get("messages", [])
    context = state.get("context", {})
    
    # Check if we should continue with more tools or go back to the LLM
    return {"messages": messages, "context": context, "__next__": "assistant"}

# Graph setup
graph_builder = StateGraph(GraphState)
graph_builder.add_node("select_tools", select_tools)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools", ContextAwareToolNode(tools=valid_tools))
graph_builder.add_node("handle_tool_results", handle_tool_results)

# Updated edges
graph_builder.add_conditional_edges(
    "assistant",
    lambda state: state.get("__next__", "__end__"),
    {
        "tools": "tools",
        "select_tools": "select_tools",
        "__end__": END,
    }
)

# After tools execute, go to the handler
graph_builder.add_edge("tools", "handle_tool_results")
# After handling results, go back to assistant for potential more tool calls
graph_builder.add_edge("handle_tool_results", "assistant")
graph_builder.add_edge("select_tools", "assistant")
graph_builder.add_edge(START, "select_tools")

compiled_graph = graph_builder.compile()

async def run_cli_interaction():
    """Runs the CLI interaction loop."""
    state = {"messages": [], "context": {}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        user_message = HumanMessage(content=user_input)
        state["messages"].append(user_message)

        print("🚀 Invoking graph...")
        result = await compiled_graph.ainvoke(state, config={"recursion_limit": 100})
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                print("Assistant:", message.content)
                break

if __name__ == "__main__":
    asyncio.run(run_cli_interaction())
