import os
import re
import ast
import json
import inspect
import logging
import importlib
import subprocess
from functools import wraps
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from typing import Dict, Any, List, Optional, Union
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

class MCPToolDiscovery:
    def __init__(self, container_name: str, command: List[str], discovery_method: str = "tools/discover", call_method: str = "tools/call"):
        self.container_name = container_name
        self.command = command
        self.discovery_method = discovery_method
        self.call_method = call_method
        self.discovered_tools = []

    def discover_tools(self) -> List[Dict[str, Any]]:
        print(f"🔍 Discovering tools from container: {self.container_name}")
        
        if self.container_name == "google-maps-mcp":
            logger.info("🔧 Using manual MAPS tool definitions as fallback")
            return [
                {"name": "maps_geocode", "description": "Convert an address into geographic coordinates"},
                {"name": "maps_reverse_geocode", "description": "Convert coordinates into an address"},
                {"name": "maps_search_places", "description": "Search for places using Google Places API"},
                {"name": "maps_place_details", "description": "Get detailed information about a specific place"},
                {"name": "maps_distance_matrix", "description": "Calculate travel distance and time for multiple origins and destinations"},
                {"name": "maps_elevation", "description": "Get elevation data for locations on the earth"},
                {"name": "maps_directions", "description": "Get directions between two points"},
            ]

        try:
            discovery_payload = {
                "jsonrpc": "2.0",
                "method": self.discovery_method,
                "params": {},
                "id": "1"
            }
            process = subprocess.run(
                ["docker", "exec", "-i", self.container_name] + self.command,
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
        
    def create_dynamic_tool(self, tool_info: Dict[str, Any]) -> Tool:
        tool_name = tool_info["name"]   

        class FlexibleInput(BaseModel):
            content: Union[str, Dict[str, Any]] = Field(
                default=None,
                description="Flexible input that can be a string or dictionary, used to ask Selector something."
            )

        # Specific input model for GitHub file creation
        class GitHubFileInput(BaseModel):
            """
            Structured input model for GitHub file creation tool.

            Requires specific fields for creating or updating a file in a GitHub repository.
            """
            owner: str = Field(..., description="Repository owner (username or organization)")
            repo: str = Field(..., description="Repository name")
            path: str = Field(..., description="Path where to create/update the file")
            content: str = Field(default="Default content", description="Content of the file")
            message: str = Field(..., description="Commit message")
            branch: str = Field(default="main", description="Branch to create/update the file in")
            sha: Optional[str] = Field(default=None, description="SHA of the file being replaced (required when updating existing files)")  

        class SequentialThinkingInput(BaseModel):
            thought: str = Field(..., description="Your current thinking step.")
            nextThoughtNeeded: bool = Field(..., description="Whether another thought step is needed.")
            thoughtNumber: int = Field(..., description="Current thought number", ge=1)
            totalThoughts: int = Field(..., description="Estimated total thoughts needed", ge=1)
            isRevision: bool = Field(default=False, description="Whether this revises previous thinking.")
            revisesThought: Optional[int] = Field(default=None, description="Which thought number is being reconsidered.")
            branchFromThought: Optional[int] = Field(default=None, description="Branching point thought number.")
            branchId: Optional[str] = Field(default=None, description="Branch identifier.")
            needsMoreThoughts: Optional[bool] = Field(default=None, description="If more thoughts are needed.")

        class SlackPostMessageInput(BaseModel):
            """
            Structured input for sending a message to a Slack channel.
            """
            channel_id: str = Field(..., description="The ID of the Slack channel to post to")
            text: str = Field(..., description="The message text to send")

        class IPInput(BaseModel):
            ip: str

        class CreateDrawingInput(BaseModel):
            name: str = Field(..., description="Name of the Excalidraw drawing")
            content: str = Field(..., description="Instructions or content for the drawing")

        class ExportToJsonInput(BaseModel):
            """
            Structured input for exporting an Excalidraw drawing to JSON.
            """
            id: str = Field(..., description="The unique identifier of the Excalidraw drawing to export")
        
        @traceable(name=f"Tool - {tool_name}")
        def dynamic_tool_function(content: Union[str, Dict[str, Any]] = None, **kwargs):
            logger.info(f"⚙️ Calling tool: {tool_name}")
            logger.info(f"🔍 Full kwargs received: {kwargs}")

            try:
                payload_arguments = {}  # Initialize as empty dict
                if content:
                  payload_arguments['content'] = content

                # Special handling for create_or_update_file
                if tool_name == "create_or_update_file":
                    # Check if content exists in the original tool calls
                    if 'content' in kwargs:
                        payload_arguments['content'] = kwargs['content']
                    elif content:
                        payload_arguments['content'] = content

                    # Fallback to default if no content
                    if 'content' not in payload_arguments or not payload_arguments['content']:
                        logger.warning("No content found. Using default content.")
                        payload_arguments['content'] = "Default health report content for Selector Device S3"

                    # Ensure 'branch' is always provided
                    if 'branch' not in payload_arguments:
                        payload_arguments['branch'] = 'main'  # Default branch

                    logger.info(f"🚀 Final payload_arguments for file creation: {payload_arguments}")
                elif tool_name == "sequentialthinking":
                    # ✅ Use all structured fields already passed through args
                    if content and isinstance(content, dict):
                        payload_arguments.update(content)
                    elif isinstance(content, str):
                        payload_arguments["thought"] = content
                    # Ensure required fields if missing (for robustness)
                    if "thoughtNumber" not in payload_arguments:
                        payload_arguments["thoughtNumber"] = 1
                    if "totalThoughts" not in payload_arguments:
                        payload_arguments["totalThoughts"] = 5
                    if "nextThoughtNeeded" not in payload_arguments:
                        payload_arguments["nextThoughtNeeded"] = True
                    if "isRevision" not in payload_arguments:
                        payload_arguments["isRevision"] = False

                elif tool_name == "create_drawing":
                    if content:
                        payload_arguments["content"] = content
                    if "name" not in payload_arguments:
                        payload_arguments["name"] = "AI Drawing"
                    logger.info(f"🖌️ Final payload_arguments for drawing: {payload_arguments}")

                elif tool_name == "ask_selector":
                    if content:
                        payload_arguments["content"] = content
                    else:
                      logger.warning(f"⚠️ No content provided for ask_selector. Using default empty string")
                      payload_arguments['content'] = ""

                    logger.info(f"🔍 ask_selector arguments: {payload_arguments}")

                payload = {
                    "jsonrpc": "2.0",
                    "method": self.call_method,
                    "params": {"name": tool_name, "arguments": payload_arguments},
                    "id": "2",
                }

                logger.info(f"📤 Payload: {json.dumps(payload, indent=2)}")

                process = subprocess.run(
                    ["docker", "exec", "-i", self.container_name] + self.command,
                    input=json.dumps(payload) + "\n",
                    capture_output=True,
                    text=True,
                )

                logger.info(f"📥 STDOUT: {process.stdout}")
                if process.stderr:
                    logger.error(f"🚨 STDERR: {process.stderr}")

                stdout_lines = process.stdout.strip().split("\n")
                if stdout_lines:
                    last_line = stdout_lines[-1]
                    try:
                        response = json.loads(last_line)
                        logger.info(f"📝 Full Response: {json.dumps(response, indent=2)}")

                        result = response.get("result", {})
                        content = result.get("content", str(result))

                        logger.info(f"✅ Result content: {content}")
                        return content
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON Decode Error: {e}")
                        return f"JSON Decode Error: {e}"
                else:
                    logger.warning("❌ No response lines received.")
                    return "No response from tool"

            except Exception as e:
                logger.error(f"❌ Execution error: {e}")
                return f"Execution error: {e}"
        
        return StructuredTool.from_function(
            name=tool_name,
            description=tool_info.get("description", f"Tool from {self.container_name}"),
            func=dynamic_tool_function,
            args_schema=(
                GitHubFileInput if tool_name == "create_or_update_file" else
                SequentialThinkingInput if tool_name == "sequentialthinking" else
                SlackPostMessageInput if tool_name == "slack_post_message" else
                CreateDrawingInput if tool_name == "create_drawing" else
                ExportToJsonInput if tool_name == "export_to_json" else
                IPInput if tool_name in {
                    "bgp_lookup_tool",
                    "curl_lookup_tool",
                    "dig_tool",
                    "nslookup_tool",
                    "ping_tool",
                    "whois_tool",
                    "get_location_tool",
                    "traceroute_tool",
                    "threat_check_tool"
                } else
                FlexibleInput
                )
            )
    
    def get_tools(self) -> List[Tool]:
        if not self.discovered_tools:
            discovered_tool_info = self.discover_tools()
            self.discovered_tools = [self.create_dynamic_tool(tool_info) for tool_info in discovered_tool_info]
        return self.discovered_tools
    
# Python-based Selector
selector_discovery = MCPToolDiscovery("selector-mcp", ["python3", "mcp_server.py", "--oneshot"])
selector_tools = selector_discovery.get_tools()

# Node.js-based GitHub
github_discovery = MCPToolDiscovery(
    "github-mcp",
    ["node", "dist/index.js"],
    discovery_method="list_tools",
    call_method="call_tool"
)
github_tools = github_discovery.get_tools()


# Node.js-based Google Maps
maps_discovery = MCPToolDiscovery(
    container_name="google-maps-mcp",
    command=["node", "dist/index.js"],
    discovery_method="list_tools",
    call_method="tools/call"
)
maps_tools = maps_discovery.get_tools()

# Node.js-based Google Maps
sequentialthinking_discovery = MCPToolDiscovery(
    container_name="sequentialthinking-mcp",
    command=["node", "dist/index.js"],
    discovery_method="tools/list",
    call_method="tools/call"
)
sequentialthinking_tools = sequentialthinking_discovery.get_tools()

# Node.js-based Google Maps
slack_discovery = MCPToolDiscovery(
    container_name="slack-mcp",
    command=["node", "dist/index.js"],
    discovery_method="tools/list",
    call_method="tools/call"
)
slack_tools = slack_discovery.get_tools()

# Node.js-based Google Maps
excalidraw_discovery = MCPToolDiscovery(
    container_name="excalidraw-mcp",
    command=["node", "dist/index.js"],
    discovery_method="tools/list",
    call_method="tools/call"
)
excalidraw_tools = excalidraw_discovery.get_tools()

# Local tools from ./tools folder
local_tools = load_local_tools_from_folder("tools")

# Merge all tools
# Merge all MCP and local tools
dynamic_tools = (
    selector_tools +
    github_tools +
    maps_tools +
    sequentialthinking_tools +
    slack_tools +
    excalidraw_tools +
    local_tools  # 👈 your local ping/dig/curl/etc.
)

valid_tools = [t for t in dynamic_tools if t is not None]

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
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']

            tool_args = tool_call['args'].copy()
            logger.info(f"🔍 Tool {tool_name} args from LLM: {tool_args}")
            tool = next((t for t in valid_tools if t.name == tool_name), None)

            # ✅ Handle __arg1 mapping
            if '__arg1' in tool_args and tool:
                if tool_name in {
                    "bgp_lookup_tool", "curl_lookup_tool", "dig_tool", 
                    "nslookup_tool", "ping_tool", "whois_tool", 
                    "get_location_tool", "traceroute_tool", "threat_check_tool"
                }:
                    tool_args = {"ip": tool_args.pop('__arg1')}
                    logger.info(f"🛠️ Mapped __arg1 to {{'ip': ...}} for tool {tool_name}")
                elif tool_name in {
                    "get_drawing", "update_drawing", "delete_drawing", 
                    "export_to_svg", "export_to_png", "export_to_json"
                }:
                    tool_args = {"id": tool_args.pop('__arg1')}
                    logger.info(f"🛠️ Mapped __arg1 to {{'id': ...}} for tool {tool_name}")

            # ✅ Handle content-based parsing
            if "content" in tool_args and tool_name in {
                "get_drawing", "update_drawing", "delete_drawing", 
                "export_to_svg", "export_to_png", "export_to_json"
            }:
                content_raw = tool_args.get("content")
                try:
                    # Try parsing as a dictionary or extracting the ID
                    if isinstance(content_raw, str):
                        try:
                            parsed = ast.literal_eval(content_raw)
                            tool_args["id"] = parsed.get("id", parsed) if isinstance(parsed, (dict, str)) else content_raw
                        except (ValueError, SyntaxError):
                            tool_args["id"] = content_raw

                    # Validate the input using Pydantic
                    ExportToJsonInput(id=tool_args["id"])
                    logger.info(f"🛠️ Validated ID for {tool_name}")

                except Exception as e:
                    logger.error(f"🚨 Failed to process ID for {tool_name}: {e}")
                    # Handle error or raise as needed

            tool = next((t for t in valid_tools if t.name == tool_name), None)

            if tool:
                try:
                    tool_result = tool.run(tool_args)
                    logger.info(f"Tool {tool_name} result: {tool_result}")

                    # ✅ Clean result message without tool_calls (prevents infinite loop)
                    tool_call_messages.append(
                        AIMessage(content=f"✅ `{tool_name}` executed with result:\n```\n{tool_result}\n```")
                    )

                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    tool_call_messages.append(
                        AIMessage(content=f"❌ Error executing `{tool_name}`: {e}")
                    )

    final_messages = [response] if not tool_call_messages else tool_call_messages
    return {"messages": final_messages}

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

logger.info("🚀 Packet Copilot LangGraph compiled successfully")

# CLI Loop
def run_cli_interaction():
    state = {"messages": []}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        user_message = HumanMessage(content=user_input)
        state["messages"].append(user_message)

        print("🚀 Invoking graph...")
        result = compiled_graph.invoke(state)
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage) and (not hasattr(message, "tool_calls") or not message.tool_calls):
                print("Assistant:", message.content)
                break

if __name__ == "__main__":
    run_cli_interaction()
