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

            # Standardized argument handling: Try to parse __arg1
            if '__arg1' in arguments:
                try:
                    normalized_args = json.loads(arguments['__arg1'])
                except json.JSONDecodeError:
                    normalized_args = arguments['__arg1']  # Use the raw string if parsing fails

            # Tool-specific argument formatting (if needed)
            if tool_name == 'ask_selector':
                normalized_args = {"content": normalized_args}

            elif tool_name == 'sequentialthinking':
                # Format arguments for sequentialthinking
                if isinstance(normalized_args, dict):
                    tool_args = normalized_args
                else:
                    tool_args = {
                        "thought": normalized_args,
                        "nextThoughtNeeded": True,  # Or False if it's the final thought
                        "thoughtNumber": 1,
                        "totalThoughts": 6
                    }
                # Ensure thoughtNumber is an integer
                if "thoughtNumber" in tool_args:
                    tool_args["thoughtNumber"] = int(tool_args["thoughtNumber"])
                normalized_args = tool_args

            elif tool_name == 'create_or_update_file':
                # Format arguments for GitHub
                # Assuming the LLM provides these fields in normalized_args
                required_fields = ["owner", "repo", "path", "content", "message", "branch"]
                github_args = {}
                valid_args = True
                for field in required_fields:
                    if field not in normalized_args:
                        logger.error(f"GitHub tool requires field: {field}")
                        valid_args = False
                        break
                    github_args[field] = normalized_args[field]
                if valid_args:
                    normalized_args = github_args
                    # Decode content from base64
                    if "content" in normalized_args:
                        import base64
                        normalized_args["content"] = base64.b64decode(normalized_args["content"]).decode()
                    else:
                        return "Error: Missing required field content"
                else:
                    return "Error: Missing required fields for GitHub tool"

            elif tool_name == 'create_drawing' or tool_name == 'get_drawing' or tool_name == 'update_drawing' or tool_name == 'delete_drawing' or tool_name == 'list_drawings' or tool_name == 'export_to_json':
                # Format arguments for Excalidraw
                if "content" in normalized_args:
                    normalized_args["content"] = json.dumps(normalized_args["content"])
                if tool_name == 'export_to_json':
                    # Ensure 'id' is passed correctly
                    if isinstance(normalized_args, str):
                        normalized_args = {"id": normalized_args}

            elif tool_name in ['read_file', 'read_multiple_files', 'write_file', 'edit_file', 'create_directory', 'list_directory', 'move_file', 'search_files', 'get_file_info', 'directory_tree']:
                # Format arguments for filesystem tools
                normalized_args = normalized_args if isinstance(normalized_args, dict) else {"path": normalized_args}
                
                # Correct paths for filesystem access
                def ensure_projects_path(path):
                    """Ensure path starts with '/projects/'"""
                    if not path:
                        return path
                    return path if path.startswith('/projects/') else f'/projects/{path.lstrip("/")}'
                
                # Apply path correction to different argument types
                if 'path' in normalized_args:
                    normalized_args['path'] = ensure_projects_path(normalized_args['path'])
                
                if 'source' in normalized_args:
                    normalized_args['source'] = ensure_projects_path(normalized_args['source'])
                
                if 'destination' in normalized_args:
                    normalized_args['destination'] = ensure_projects_path(normalized_args['destination'])
                
                if 'paths' in normalized_args:
                    normalized_args['paths'] = [ensure_projects_path(path) for path in normalized_args['paths']]
            
            elif tool_name == 'list_allowed_directories':
                # Correctly format arguments as an empty object
                normalized_args = {}

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
            ("selector-mcp", ["python3", "mcp_server.py", "--oneshot"]),
            ("github-mcp", ["node", "dist/index.js"]),
            ("google-maps-mcp", ["node", "dist/index.js"]),
            ("sequentialthinking-mcp", ["node", "dist/index.js"]),
            ("slack-mcp", ["node", "dist/index.js"]),
            ("excalidraw-mcp", ["node", "dist/index.js"]),
            ("filesystem-mcp", ["node", "dist/index.js"])
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

    try:
        discovered_tools = await discovery.discover_tools()
        print(f"🛠️ Tools for {service_name}: {discovered_tools}")

        # Convert discovered tools to Tool objects
        tools = []
        for tool in discovered_tools:
            tool_name = tool['name']
            tool_description = tool.get('description', '')

            # Create a Tool that calls call_tool
            def tool_wrapper(input_arg, tool_name=tool_name):
                return asyncio.run(service_discoveries[service_name].call_tool(tool_name, {"__arg1": input_arg}))

            tools.append(
                Tool(
                    name=tool_name,
                    description=tool_description,
                    func=tool_wrapper,
                )
            )
        return tools
    except Exception as e:
        print(f"❌ Tool Discovery Error for {service_name}: {e}")
        return []

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
        ("filesystem-mcp", ["node", "dist/index.js", "/projects"], "tools/list", "tools/call")
    ]

    service_discoveries = {}  # Store MCPToolDiscovery instances

    async def get_tools_for_service(service_name, command, discovery_method, call_method):
        """Enhanced tool discovery for each service."""
        print(f"🕵️ Discovering tools for: {service_name}")
        discovery = MCPToolDiscovery(
            service_name,
            command,
            discovery_method=discovery_method,
            call_method=call_method
        )
        service_discoveries[service_name] = discovery  # Store the instance

        try:
            discovered_tools = await discovery.discover_tools()
            print(f"🛠️ Tools for {service_name}: {discovered_tools}")

            # Convert discovered tools to Tool objects
            tools = []
            for tool in discovered_tools:
                tool_name = tool['name']
                tool_description = tool.get('description', '')

                # Create a Tool that calls call_tool
                def tool_wrapper(input_arg, tool_name=tool_name):
                    return asyncio.run(service_discoveries[service_name].call_tool(tool_name, {"__arg1": input_arg}))

                tools.append(
                    Tool(
                        name=tool_name,
                        description=tool_description,
                        func=tool_wrapper,
                    )
                )
            return tools
        except Exception as e:
            print(f"❌ Tool Discovery Error for {service_name}: {e}")
            return []

    try:
        # Run docker ps to verify containers
        docker_ps_result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        print(docker_ps_result.stdout)

        # Gather tools from all services
        all_service_tools = await asyncio.gather(
            *[get_tools_for_service(service, command, discovery_method, call_method)
              for service, command, discovery_method, call_method in tool_services]
        )

        # Flatten the list of tools
        dynamic_tools = [tool for service_tools in all_service_tools for tool in all_service_tools]

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

print("🔧 All bound tools:", [t.name for t in valid_tools])

# LLM
llm = ChatOpenAI(model="gpt-4o")

llm_with_tools = llm.bind_tools(valid_tools, parallel_tool_calls=False)

# System Message
sys_msg = SystemMessage(content="You are an AI assistant with dynamically discovered tools.")

@traceable
def assistant(state: MessagesState):
    """Handles user questions, detects PCAP files, and dynamically invokes tools when needed."""
    
    messages = state.get("messages", [])

    # ✅ Extract latest user message
    latest_user_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
    if not latest_user_message:
        return {"messages": [AIMessage(content="⚠️ No valid question detected.")]}

    logger.info(f"🛠️ Processing Message: {latest_user_message.content}")

    # ✅ Preserve existing messages, including previous PCAP analysis
    new_messages = [sys_msg] + messages  

    # ✅ Invoke LLM with tools
    response = llm_with_tools.invoke(new_messages)

    # ✅ Log tool calls before returning response
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"🛠️ Tool Calls Detected: {response.tool_calls}")

    return {"messages": response}

# ✅ Build the LangGraph
builder = StateGraph(MessagesState)

# ✅ Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(valid_tools))

# ✅ Define Edges (Matches Space Graph)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # Routes to "tools" if tools are needed, else to END
)
builder.add_edge("tools", "assistant")  # ✅ Tools always return to assistant

# ✅ Compile the Graph
compiled_graph = builder.compile()

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
        result = await compiled_graph.ainvoke(state)  # ainvoke for async
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and (not hasattr(message, "tool_calls") or not message.tool_calls):
                print("Assistant:", message.content)
                break

if __name__ == "__main__":
    asyncio.run(run_cli_interaction())