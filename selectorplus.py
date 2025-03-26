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

from typing import List, Dict, Any
from anthropic.mcp import MCPClient, Tool

class MCPToolDiscovery:
    def __init__(self, service_name: str):
        """
        Initialize MCP Tool Discovery with a service name.
        
        :param service_name: Name of the MCP service to discover tools from
        """
        self.client = MCPClient()
        self.service_name = service_name
        self.discovered_tools = []

    def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover tools for the specified service using MCP Client.
        
        :return: List of discovered tool definitions
        """
        try:
            # Use MCP Client to discover tools for the service
            tools = self.client.discover_tools(self.service_name)
            
            # Log discovered tools
            print(f"🔍 Discovered {len(tools)} tools for {self.service_name}")
            for tool in tools:
                print(f"✅ Tool: {tool['name']} - {tool.get('description', 'No description')}")
            
            return tools
        
        except Exception as e:
            print(f"❌ Error discovering tools for {self.service_name}: {e}")
            return []

    def get_tools(self) -> List[Tool]:
        """
        Get discovered tools, caching them for subsequent calls.
        
        :return: List of Tool objects
        """
        if not self.discovered_tools:
            discovered_tool_info = self.discover_tools()
            self.discovered_tools = [
                self.client.create_tool(tool_info) 
                for tool_info in discovered_tool_info
            ]
        
        return self.discovered_tools

# Example usage
def load_mcp_tools() -> List[Tool]:
    """
    Load tools from different MCP services.
    
    :return: Consolidated list of tools
    """
    tool_services = [
        "selector-mcp",
        "github-mcp", 
        "google-maps-mcp", 
        "sequentialthinking-mcp", 
        "slack-mcp", 
        "excalidraw-mcp"
    ]
    
    dynamic_tools = []
    for service in tool_services:
        discovery = MCPToolDiscovery(service)
        dynamic_tools.extend(discovery.get_tools())
    
    # Load local tools as before
    local_tools = load_local_tools_from_folder("tools")
    
    # Combine MCP and local tools
    all_tools = dynamic_tools + local_tools
    
    # Filter out None tools
    valid_tools = [t for t in all_tools if t is not None]
    
    print("🔧 All bound tools:", [t.name for t in valid_tools])
    
    return valid_tools

# Python-based Selector
selector_discovery = MCPToolDiscovery("selector-mcp")
selector_tools = selector_discovery.get_tools()

# Node.js-based GitHub
github_discovery = MCPToolDiscovery("github-mcp")
github_tools = github_discovery.get_tools()

# Node.js-based Google Maps
maps_discovery = MCPToolDiscovery("google-maps-mcp")
maps_tools = maps_discovery.get_tools()

# Node.js-based Sequential Thinking
sequentialthinking_discovery = MCPToolDiscovery("sequentialthinking-mcp")
sequentialthinking_tools = sequentialthinking_discovery.get_tools()

# Node.js-based Slack
slack_discovery = MCPToolDiscovery("slack-mcp")
slack_tools = slack_discovery.get_tools()

# Node.js-based Excalidraw
excalidraw_discovery = MCPToolDiscovery("excalidraw-mcp")
excalidraw_tools = excalidraw_discovery.get_tools()

# Local tools from ./tools folder
local_tools = load_local_tools_from_folder("tools")

# Merge all tools
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
    
    try:
        response = llm_with_tools.invoke(new_messages)
    except Exception as e:
        logger.error(f"❌ LLM Invocation Error: {e}")
        return {"messages": [AIMessage(content=f"🚨 Error processing your request: {e}")]}

    tool_call_messages = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"🛠️ Tool Calls Detected: {response.tool_calls}")
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']

            tool_args = tool_call['args'].copy()
            logger.info(f"🔍 Tool {tool_name} args from LLM: {tool_args}")
            
            # Simplified tool finding (assuming MCP SDK provides a consistent tool interface)
            tool = next((t for t in valid_tools if t.name == tool_name), None)

            if tool:
                try:
                    # Use the tool's run method directly
                    tool_result = tool.run(tool_args)
                    logger.info(f"Tool {tool_name} result: {tool_result}")

                    tool_call_messages.append(
                        AIMessage(content=f"✅ `{tool_name}` executed with result:\n```\n{tool_result}\n```")
                    )

                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    tool_call_messages.append(
                        AIMessage(content=f"❌ Error executing `{tool_name}`: {e}")
                    )
            else:
                logger.warning(f"🚨 Tool {tool_name} not found in valid tools")

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
