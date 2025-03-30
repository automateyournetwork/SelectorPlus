import os
import json
import copy
import asyncio
import inspect
import logging
import importlib
import subprocess
import uuid
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
from langchain.prebuilt.tool_node import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
                item_model = schema_to_pydantic_model(name + "_" + field_name + "_Item", items_schema)
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

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embedding=embedding)

tool_documents = [
    Document(page_content=tool.description or "", metadata={"tool_name": tool.name})
    for tool in valid_tools if hasattr(tool, "description")
]

document_ids = vector_store.add_documents(tool_documents)

print("🔧 All bound tools:", [t.name for t in valid_tools])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
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
            logger.info(f"Calling tool: {tool.name} with args: {tool_input}")
            tool_response = tool.invoke(tool_input)  # Execute the tool

            if not isinstance(tool_response, dict):
                raise ValueError(f"Tool {tool.name} should return a dictionary, but returned {type(tool_response)}")
            
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
    """Selects tools based on user query."""
    messages = state.get("messages", [])
    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if last_user_message:
        query = last_user_message.content
        relevant_docs = vector_store.similarity_search(query, k=8)
        selected_tool_names = [doc.metadata["tool_name"] for doc in relevant_docs]
        logger.info(f"🔍 Tool selection for '{query}': {selected_tool_names}")

        return {
            "messages": messages,
            "context": state.get("context", {}),  # Pass the context
        }

@traceable
def assistant(state: GraphState):
    """Handles assistant logic and LLM interaction."""
    messages = state.get("messages", [])
    context = state.get("context", {})
    tools_to_use = [tool for tool in valid_tools if tool.name in state.get("selected_tools", [])]
    if not tools_to_use:
        logger.warning("🤷 No tools selected, using all tools")
        tools_to_use = valid_tools

    llm_with_selected_tools = llm.bind_tools(tools_to_use)
    formatted_tool_descriptions = format_tool_descriptions(tools_to_use)
    formatted_system_msg = system_msg.format(tool_descriptions=formatted_tool_descriptions)
    new_messages = [SystemMessage(content=formatted_system_msg)] + messages

    try:
        logger.info(f"assistant: Invoking LLM with new_messages: {new_messages}")
        response = llm_with_selected_tools.invoke(
            new_messages,
            config={"tool_choice": "auto"}  #  "auto"
        )
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        response = AIMessage(content=f"LLM Error: {e}")

    if response and hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"🛠️ Tool Calls Detected: {response.tool_calls}")
        logger.info(f"Response: {response}")
        return {"messages": [response], "context": context}
    elif response and hasattr(response, "content") and response.content:
        logger.info(f"🧠 Assistant response: {response.content}")
        return {"messages": [response], "context": context}
    else:
        logger.warning("⚠️ Empty response from LLM")
        return {"messages": [AIMessage(content="No response from LLM. Returning to user.")]}



def handle_tool_response(state: GraphState) -> GraphState:
    """Handles responses from tool calls and updates state."""
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, ToolMessage):
        raise ValueError(f"Expected ToolMessage, got {type(last_message)}")

    tool_name = last_message.name
    tool_output = last_message.content
    context = state.get("context", {})

    logger.info(f"Handling tool response from {tool_name}: {tool_output}")

    # Update the context with the tool's output.  Assume tool returns a dict.
    try:
        tool_output_dict = json.loads(tool_output)
        if isinstance(tool_output_dict, dict):
            context.update(tool_output_dict)
            state["context"] = context #update the context
        else:
            context["tool_response"] = tool_output
            state["context"] = context
    except json.JSONDecodeError:
        context["tool_response"] = tool_output
        state["context"] = context

    # Add the ToolMessage to the message history.
    updated_messages = messages + [last_message]
    return {"messages": updated_messages, "context": context}



def tools_condition(state: GraphState) -> str:
    """Determines which node to go to after the assistant node."""
    messages = state.get("messages", [])
    if not messages:
        logger.info("tools_condition: No messages, going to START")
        return START
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("tools_condition: AIMessage with tool_calls, go to tools")
            return "tools"
        else:
            logger.info("tools_condition: AIMessage with content, go to __end__")
            return "__end__"
    elif isinstance(last_message, ToolMessage):
        logger.info("tools_condition: ToolMessage, go to handle_tool_response")
        return "handle_tool_response"
    else:
        logger.info("tools_condition: Default, go to assistant")
        return "assistant"


graph_builder = StateGraph(GraphState)
graph_builder.add_node("select_tools", select_tools)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools", ContextAwareToolNode(tools=valid_tools))
graph_builder.add_node("handle_tool_response", handle_tool_response)

graph_builder.add_conditional_edges(
    "assistant",
    tools_condition,
    path_map={
        "tools": "tools",
        "__end__": END,
    }
)

graph_builder.add_edge("select_tools", "assistant")
graph_builder.add_edge("tools", "handle_tool_response")
graph_builder.add_conditional_edges(
    "handle_tool_response",
    tools_condition,
    path_map={
        "assistant": "assistant",
        "__end__": END,
    }
)
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
