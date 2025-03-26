import os
import re
import ast
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
from langchain_openai import ChatOpenAI
from mcp.client.stdio import stdio_client
from langchain.tools import Tool, StructuredTool
from typing import Dict, Any, List, Optional, Union
from mcp import ClientSession, StdioServerParameters, types
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt.tool_node import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

logger = logging.getLogger(__name__)

class MCPToolDiscovery:
    def __init__(self, container_name: str, command: List[str], discovery_method: str = "tools/discover", call_method: str = "tools/call"):
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
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Call a specific tool with given arguments.

        :param tool_name: Name of the tool to call
        :param arguments: Arguments for the tool
        :return: Tool execution result
        """
        command = ["docker", "exec", "-i", self.container_name] + self.command
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": self.call_method,
                "params": {"name": tool_name, "arguments": arguments},
                "id": "2",
            }

            process = subprocess.run(
                command,
                input=json.dumps(payload) + "\n",
                capture_output=True,
                text=True,
            )

            logger.info(f"📥 STDOUT: {process.stdout}")
            if process.stderr:
                logger.error(f"🚨 STDERR: {process.stderr}")

            # Parse the output
            if process.stdout:
                output = process.stdout.strip()
                # Find the last valid JSON object
                json_lines = []
                for line in reversed(output.splitlines()):
                    line = line.strip()
                    if line.startswith("{") or line.startswith("["):
                        try:
                            json_lines.append(json.loads(line))
                            break
                        except json.JSONDecodeError:
                            logger.warning(f"Ignoring invalid JSON line: {line}")
                    if json_lines:
                        response = json_lines[0]
                        logger.info(f"Parsed JSON response: {response}")
                        return response.get("result")  # Or whatever field you need
                    else:
                        logger.error("No valid JSON found in output")
                        return None
                else:
                    logger.error("No output from subprocess")
                    return None
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return None

def load_mcp_tools() -> List[Tool]:
    """
    Asynchronously load tools from different MCP services.

    :return: Consolidated list of tools
    """
    async def gather_tools():
        tool_services = [
            ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"]),
            ("github-mcp", ["node", "dist/index.js"]),
            ("google-maps-mcp", ["node", "dist/index.js"]),
            ("sequentialthinking-mcp", ["node", "dist/index.js"]),
            ("slack-mcp", ["node", "dist/index.js"]),
            ("excalidraw-mcp", ["node", "dist/index.js"])
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

async def get_tools_for_service(service_name, command):
    """Helper function to discover tools for a specific service."""
    discovery = MCPToolDiscovery(service_name, command)
    return await discovery.discover_tools()

async def load_all_tools():
    """Async function to load tools from all services."""
    tool_services = [
        ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        ("github-mcp", ["node", "dist/index.js"], "list_tools", "tools/call"),
        ("google-maps-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("sequentialthinking-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("slack-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        ("excalidraw-mcp", ["node", "dist/index.js"], "tools/list", "tools/call")
    ]

    async def get_tools_for_service(service_name, command, discovery_method, call_method):
        """Helper function to discover tools for a specific service."""
        discovery = MCPToolDiscovery(
            service_name,
            command,
            discovery_method=discovery_method,
            call_method=call_method
        )
        discovered_tools = await discovery.discover_tools()

        # Convert discovered tools to Tool objects
        return [
            Tool(
                name=tool['name'],
                description=tool.get('description', ''),
                func=lambda x: x  # Placeholder function, replace with actual implementation
            ) for tool in discovered_tools
        ]

    # Gather tools from all services
    all_service_tools = await asyncio.gather(
        *[get_tools_for_service(service, command, discovery_method, call_method)
          for service, command, discovery_method, call_method in tool_services]
    )

    # Flatten the list of tools
    dynamic_tools = [tool for service_tools in all_service_tools for tool in service_tools]

    # Add local tools
    local_tools = load_local_tools_from_folder("tools")

    # Combine all tools
    all_tools = dynamic_tools + local_tools

    # Filter out None tools
    valid_tools = [t for t in all_tools if t is not None]

    print("🔧 All bound tools:", [t.name for t in valid_tools])

    return valid_tools

# Use asyncio to run the async function and get tools
valid_tools = asyncio.run(load_all_tools())

print("🔧 All bound tools:", [t.name for t in valid_tools])

# LLM
llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools(valid_tools, parallel_tool_calls=False)

# System Message
sys_msg = SystemMessage(content="You are an AI assistant with dynamically discovered tools.")

@traceable
def assistant(state: MessagesState):
    """Handles user questions and dynamically invokes tools when needed."""
    messages = state.get("messages", [])
    latest_user_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    if not latest_user_message:
        return {"messages": [AIMessage(content="⚠️ No valid question detected.")]}

    logger.info(f"🛠️ Processing Message: {latest_user_message.content}")
    new_messages = [sys_msg] + messages
    response = llm_with_tools.invoke(new_messages)

    tool_call_messages = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"🛠️ Tool Calls Detected: {response.tool_calls}")

        # Create an async function to handle tool calls
        async def process_tool_calls():
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args'].copy()
                logger.info(f"🛠️ Calling tool '{tool_name}' with args: {tool_args}")
                # Find the corresponding MCP discovery instance
                discovery = next((
                    MCPToolDiscovery(service, command)
                    for service, command in [
                        ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"]),
                        ("github-mcp", ["node", "dist/index.js"]),
                        ("google-maps-mcp", ["node", "dist/index.js"]),
                        ("sequentialthinking-mcp", ["node", "dist/index.js"]),
                        ("slack-mcp", ["node", "dist/index.js"]),
                        ("excalidraw-mcp", ["node", "dist/index.js"])
                    ] if service in tool_name
                ), None)

                if discovery:
                    try:
                        tool_result = await discovery.call_tool(tool_name, tool_args)
                        logger.info(f"🛠️ Tool '{tool_name}' result: {tool_result}")  # ADDED
                        tool_results.append((tool_name, tool_result))
                    except Exception as e:
                        logger.error(f"❌ Error calling tool '{tool_name}': {e}", exc_info=True)  # ADDED
                        tool_results.append((tool_name, f"Error: {e}"))
                else:
                    logger.warning(f"⚠️ Tool '{tool_name}' not found.")  # ADDED
                    tool_results.append((tool_name, "Tool not found"))

            return tool_results

        # Run the async tool processing
        tool_call_results = asyncio.run(process_tool_calls())

        # Convert results to messages
        for tool_name, result in tool_call_results:
            tool_call_messages.append(
                AIMessage(content=f"✅ `{tool_name}` executed with result:\n```\n{result}\n```")
            )

    final_messages = [response] if not tool_call_messages else tool_call_messages
    return {"messages": final_messages}

logging.info("Script started, LangGraph setup commented out.")

# ✅ Build the LangGraph
builder = StateGraph(MessagesState)

# ✅ Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([]))

# ✅ Define Edges (Matches Space Graph)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to "tools" if tools are needed, else to END
)
builder.add_edge("tools", "assistant")  # ✅ Tools always return to assistant

# ✅ Compile the Graph
compiled_graph = builder.compile()

logger.info("🚀 Packet Copilot LangGraph compiled successfully")

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
        result = await compiled_graph.ainvoke(state)  # ainvoke for async
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and (not hasattr(message, "tool_calls") or not message.tool_calls):
                print("Assistant:", message.content)
                break

if __name__ == "__main__":
    asyncio.run(run_cli_interaction())