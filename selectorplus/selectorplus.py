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
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from mcp.client.stdio import stdio_client
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.messages import BaseMessage
from langchain.tools import Tool, StructuredTool
from langgraph.graph.message import add_messages
from langchain_core.vectorstores import InMemoryVectorStore
from mcp import ClientSession, StdioServerParameters, types
from typing import Dict, Any, List, Optional, Union, Annotated
from langgraph.graph import MessagesState, StateGraph, START, END
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


logger = logging.getLogger(__name__)


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


READ ME HELP FOR EACH TOOL: 

# Brave Search MCP Server
## Tools

- **brave_web_search**
  - Execute web searches with pagination and filtering
  - Inputs:
    - `query` (string): Search terms
    - `count` (number, optional): Results per page (max 20)
    - `offset` (number, optional): Pagination offset (max 9)

- **brave_local_search**
  - Search for local businesses and services
  - Inputs:
    - `query` (string): Local search terms
    - `count` (number, optional): Number of results (max 20)
  - Automatically falls back to web search if no local results found

# Excalidraw 
## Tools
### API Endpoints

The server provides the following tools:

#### Drawing Management

- `create_drawing`: Create a new Excalidraw drawing
- `get_drawing`: Get an Excalidraw drawing by ID
- `update_drawing`: Update an Excalidraw drawing by ID
- `delete_drawing`: Delete an Excalidraw drawing by ID
- `list_drawings`: List all Excalidraw drawings

#### Export Operations

- `export_to_svg`: Export an Excalidraw drawing to SVG
- `export_to_png`: Export an Excalidraw drawing to PNG
- `export_to_json`: Export an Excalidraw drawing to JSON

# FileSystem
## Tools
### Tools

- **read_file**
  - Read complete contents of a file
  - Input: `path` (string)
  - Reads complete file contents with UTF-8 encoding

- **read_multiple_files**
  - Read multiple files simultaneously
  - Input: `paths` (string[])
  - Failed reads won't stop the entire operation

- **write_file**
  - Create new file or overwrite existing (exercise caution with this)
  - Inputs:
    - `path` (string): File location
    - `content` (string): File content

- **edit_file**
  - Make selective edits using advanced pattern matching and formatting
  - Features:
    - Line-based and multi-line content matching
    - Whitespace normalization with indentation preservation
    - Multiple simultaneous edits with correct positioning
    - Indentation style detection and preservation
    - Git-style diff output with context
  - Inputs:
    - `path` (string): File to edit
    - `edits` (array): List of edit operations
      - `oldText` (string): Text to search for (can be substring)
      - `newText` (string): Text to replace with
    - `dryRun` (boolean): Preview changes without applying (default: false)
  - Returns detailed diff and match information for dry runs, otherwise applies changes
  - Best Practice: Always use dryRun first to preview changes before applying them

- **create_directory**
  - Create new directory or ensure it exists
  - Input: `path` (string)
  - Creates parent directories if needed
  - Succeeds silently if directory exists

- **list_directory**
  - List directory contents with [FILE] or [DIR] prefixes
  - Input: `path` (string)

- **move_file**
  - Move or rename files and directories
  - Inputs:
    - `source` (string)
    - `destination` (string)
  - Fails if destination exists

- **search_files**
  - Recursively search for files/directories
  - Inputs:
    - `path` (string): Starting directory
    - `pattern` (string): Search pattern
    - `excludePatterns` (string[]): Exclude any patterns. Glob formats are supported.
  - Case-insensitive matching
  - Returns full paths to matches

- **get_file_info**
  - Get detailed file/directory metadata
  - Input: `path` (string)
  - Returns:
    - Size
    - Creation time
    - Modified time
    - Access time
    - Type (file/directory)
    - Permissions

- **list_allowed_directories**
  - List all directories the server is allowed to access
  - No input required
  - Returns:
    - Directories that this server can read/write from

# GitHub
## Tools
## Tools

1.  `create_or_update_file`
    -   Create or update a single file in a repository
    -   Inputs:
        -   `owner` (string): Repository owner (username or organization)
        -   `repo` (string): Repository name
        -   `path` (string): Path where to create/update the file
        -   `content` (string): Content of the file
        -   `message` (string): Commit message
        -   `branch` (string): Branch to create/update the file in
        -   `sha` (optional string): SHA of file being replaced (for updates)
    -   Returns: File content and commit details

2.  `push_files`
    -   Push multiple files in a single commit
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `branch` (string): Branch to push to
        -   `files` (array): Files to push, each with `path` and `content`
        -   `message` (string): Commit message
    -   Returns: Updated branch reference

3.  `search_repositories`
    -   Search for GitHub repositories
    -   Inputs:
        -   `query` (string): Search query
        -   `page` (optional number): Page number for pagination
        -   `perPage` (optional number): Results per page (max 100)
    -   Returns: Repository search results

4.  `create_repository`
    -   Create a new GitHub repository
    -   Inputs:
        -   `name` (string): Repository name
        -   `description` (optional string): Repository description
        -   `private` (optional boolean): Whether repo should be private
        -   `autoInit` (optional boolean): Initialize with README
    -   Returns: Created repository details

5.  `get_file_contents`
    -   Get contents of a file or directory
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `path` (string): Path to file/directory
        -   `branch` (optional string): Branch to get contents from
    -   Returns: File/directory contents

6.  `create_issue`
    -   Create a new issue
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `title` (string): Issue title
        -   `body` (optional string): Issue description
        -   `assignees` (optional string[]): Usernames to assign
        -   `labels` (optional string[]): Labels to add
        -   `milestone` (optional number): Milestone number
    -   Returns: Created issue details

7.  `create_pull_request`
    -   Create a new pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `title` (string): PR title
        -   `body` (optional string): PR description
        -   `head` (string): Branch containing changes
        -   `base` (string): Branch to merge into
        -   `draft` (optional boolean): Create as draft PR
        -   `maintainer_can_modify` (optional boolean): Allow maintainer edits
    -   Returns: Created pull request details

8.  `fork_repository`
    -   Fork a repository
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `organization` (optional string): Organization to fork to
    -   Returns: Forked repository details

9.  `create_branch`
    -   Create a new branch
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `branch` (string): Name for new branch
        -   `from_branch` (optional string): Source branch (defaults to repo default)
    -   Returns: Created branch reference

10. `list_issues`
    -   List and filter repository issues
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `state` (optional string): Filter by state ('open', 'closed', 'all')
        -   `labels` (optional string[]): Filter by labels
        -   `sort` (optional string): Sort by ('created', 'updated', 'comments')
        -   `direction` (optional string): Sort direction ('asc', 'desc')
        -   `since` (optional string): Filter by date (ISO 8601 timestamp)
        -   `page` (optional number): Page number
        -   `per_page` (optional number): Results per page
    -   Returns: Array of issue details

11. `update_issue`
    -   Update an existing issue
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `issue_number` (number): Issue number to update
        -   `title` (optional string): New title
        -   `body` (optional string): New description
        -   `state` (optional string): New state ('open' or 'closed')
        -   `labels` (optional string[]): New labels
        -   `assignees` (optional string[]): New assignees
        -   `milestone` (optional number): New milestone number
    -   Returns: Updated issue details

12. `add_issue_comment`
    -   Add a comment to an issue
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `issue_number` (number): Issue number to comment on
        -   `body` (string): Comment text
    -   Returns: Created comment details

13. `search_code`
    -   Search for code across GitHub repositories
    -   Inputs:
        -   `q` (string): Search query using GitHub code search syntax
        -   `sort` (optional string): Sort field ('indexed' only)
        -   `order` (optional string): Sort order ('asc' or 'desc')
        -   `per_page` (optional number): Results per page (max 100)
        -   `page` (optional number): Page number
    -   Returns: Code search results with repository context

14. `search_issues`
    -   Search for issues and pull requests
    -   Inputs:
        -   `q` (string): Search query using GitHub issues search syntax
        -   `sort` (optional string): Sort field (comments, reactions, created, etc.)
        -   `order` (optional string): Sort order ('asc' or 'desc')
        -   `per_page` (optional number): Results per page (max 100)
        -   `page` (optional number): Page number
    -   Returns: Issue and pull request search results

15. `search_users`
    -   Search for GitHub users
    -   Inputs:
        -   `q` (string): Search query using GitHub users search syntax
        -   `sort` (optional string): Sort field (followers, repositories, joined)
        -   `order` (optional string): Sort order ('asc' or 'desc')
        -   `per_page` (optional number): Results per page (max 100)
        -   `page` (optional number): Page number
    -   Returns: User search results

16. `list_commits`
    -   Gets commits of a branch in a repository
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `page` (optional string): page number
        -   `per_page` (optional string): number of record per page
        -   `sha` (optional string): branch name
    -   Returns: List of commits

17. `get_issue`
    -   Gets the contents of an issue within a repository
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `issue_number` (number): Issue number to retrieve
    -   Returns: Github Issue object & details

18. `get_pull_request`
    -   Get details of a specific pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
    -   Returns: Pull request details including diff and review status

19. `list_pull_requests`
    -   List and filter repository pull requests
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `state` (optional string): Filter by state ('open', 'closed', 'all')" \
-   `head` (optional string): Filter by head user/org and branch
        -   `base` (optional string): Filter by base branch
        -   `sort` (optional string): Sort by ('created', 'updated', 'popularity', 'long-running')
        -   `direction` (optional string): Sort direction ('asc', 'desc')
        -   `per_page` (optional number): Results per page (max 100)
        -   `page` (optional number): Page number
    -   Returns: Array of pull request details

20. `create_pull_request_review`
    -   Create a review on a pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
        -   `body` (string): Review comment text
        -   `event` (string): Review action ('APPROVE', 'REQUEST_CHANGES', 'COMMENT')
        -   `commit_id` (optional string): SHA of commit to review
        -   `comments` (optional array): Line-specific comments, each with:
            -   `path` (string): File path
            -   `position` (number): Line position in diff
            -   `body` (string): Comment text
    -   Returns: Created review details

21. `merge_pull_request`
    -   Merge a pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
        -   `commit_title` (optional string): Title for merge commit
        -   `commit_message` (optional string): Extra detail for merge commit
        -   `merge_method` (optional string): Merge method ('merge', 'squash', 'rebase')
    -   Returns: Merge result details

22. `get_pull_request_files`
    -   Get the list of files changed in a pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
    -   Returns: Array of changed files with patch and status details

23. `get_pull_request_status`
    -   Get the combined status of all status checks for a pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
    -   Returns: Combined status check results and individual check details

24. `update_pull_request_branch`
    -   Update a pull request branch with the latest changes from the base branch (equivalent to GitHub's "Update branch" button)
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
        -   `expected_head_sha` (optional string): The expected SHA of the pull request's HEAD ref
    -   Returns: Success message when branch is updated

25. `get_pull_request_comments`
    -   Get the review comments on a pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
    -   Returns: Array of pull request review comments with details like the comment text, author, and location in the diff

26. `get_pull_request_reviews`
    -   Get the reviews on a pull request
    -   Inputs:
        -   `owner` (string): Repository owner
        -   `repo` (string): Repository name
        -   `pull_number` (number): Pull request number
    -   Returns: Array of pull request reviews with details like the review state (APPROVED, CHANGES_REQUESTED, etc.), reviewer, and review body

# Google Maps
## Tools
## Tools

1.  `maps_geocode`
    -   Convert address to coordinates
    -   Input: `address` (string)
    -   Returns: location, formatted_address, place_id

2.  `maps_reverse_geocode`
    -   Convert coordinates to address
    -   Inputs:
        -   `latitude` (number)
        -   `longitude` (number)
    -   Returns: formatted_address, place_id, address_components

3.  `maps_elevation`
    -   Get elevation data for locations
    -   Input: `locations` (array of {latitude, longitude})
    -   Returns: elevation data for each point

# Selector
## Tools

1.  ask_selector
    {
      "method": "tools/call",
      "tool_name": "ask_selector",
      "content": "What can you tell me about device S6?"
    }

# Sequential Thinking
## Tools
## Tool

### sequential_thinking

Facilitates a detailed, step-by-step thinking process for problem-solving and analysis.

**Inputs:**
-   `thought` (string): The current thinking step
-   `nextThoughtNeeded` (boolean): Whether another thought step is needed
-   `thoughtNumber` (integer): Current thought number
-   `totalThoughts` (integer): Estimated total thoughts needed
-   `isRevision` (boolean, optional): Whether this revises previous thinking
-   `revisesThought` (integer, optional): Which thought is being reconsidered
-   `branchFromThought` (integer, optional): Branching point thought number
-   `branchId` (string, optional): Branch identifier
-   `needsMoreThoughts` (boolean, optional): If more thoughts are needed

# Slack
## Tools
1.  `slack_list_channels`
    -   List public channels in the workspace
    -   Optional inputs:
        -   `limit` (number, default: 100, max: 200): Maximum number of channels to return
        -   `cursor` (string): Pagination cursor for next page
    -   Returns: List of channels with their IDs and information

2.  `slack_post_message`
    -   Post a new message to a Slack channel
    -   Required inputs:
        -   `channel_id` (string): The ID of the channel to post to
        -   `text` (string): The message text to post
    -   Returns: Message posting confirmation and timestamp

3.  `slack_reply_to_thread`
    -   Reply to a specific message thread
    -   Required inputs:
        -   `channel_id` (string): The channel containing the thread
        -   `thread_ts` (string): Timestamp of the parent message
        -   `text` (string): The reply text
    -   Returns: Reply confirmation and timestamp

4.  `slack_add_reaction`
    -   Add an emoji reaction to a message
    -   Required inputs:
        -   `channel_id` (string): The channel containing the message
        -   `timestamp` (string): Message timestamp to react to
        -   `reaction` (string): Emoji name without colons
    -   Returns: Reaction confirmation

5.  `slack_get_channel_history`
    -   Get recent messages from a channel
    -   Required inputs:
        -   `channel_id` (string): The channel ID
    -   Optional inputs:
        -   `limit` (number, default: 10): Number of messages to retrieve
    -   Returns: List of messages with their content and metadata

6.  `slack_get_thread_replies`
    -   Get all replies in a message thread
    -   Required inputs:
        -   `channel_id` (string): The channel containing the thread
        -   `thread_ts` (string): Timestamp of the parent message
    -   Returns: List of replies with their content and metadata

7.  `slack_get_users`
    -   Get list of workspace users with basic profile information
    -   Optional inputs:
        -   `cursor` (string): Pagination cursor for next page
        -   `limit` (number, default: 100, max: 200): Maximum users to return
    -   Returns: List of users with their basic profiles

8.  `slack_get_user_profile`
    -   Get detailed profile information for a specific user
    -   Required inputs:
        -   `user_id` (string): The user's ID
    -   Returns: Detailed user profile information

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
    response = llm_with_selected_tools.invoke(new_messages)

    # 🛠️ Tool call path (first pass)
    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info(f"🛠️ Tool Calls Detected: {response.tool_calls}")
        logger.info(f"Response: {response}")  # Log the entire response

        tool_messages = []
        for call in response.tool_calls:
            logger.info(f"Tool Call: {call}")  # Log each tool call
            try:
                args_str = json.dumps(call["args"]) if isinstance(call["args"], dict) else str(call["args"])
            except Exception as e:
                logger.error(f"Error dumping args: {e}")
                args_str = str(call["args"])
            tool_message = ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                content=args_str,
            )
            logger.info(f"Tool Message: {tool_message}")  # Log the ToolMessage
            tool_messages.append(tool_message)
        logger.info(f"Tool Messages: {tool_messages}")  # Log the list of ToolMessages
        return {"messages": messages + [response] + tool_messages}

    # 🧠 Second pass: LLM follow-up after tool response
    if any(isinstance(msg, ToolMessage) for msg in messages):
        logger.info("🔁 Processing follow-up after tool result")
        followup = llm_with_selected_tools.invoke([SystemMessage(content=system_msg)] + messages)
        if hasattr(followup, "content"):
            return {"messages": messages + [followup]}

    # ✅ Final response from LLM (no tools)
    if hasattr(response, "content") and response.content:
        logger.info(f"🧠 Assistant response: {response.content}")
        return {"messages": [response]}

    # 🚨 Fallback
    logger.warning("⚠️ Empty response from LLM")
    return {"messages": []}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]

@traceable
def tools_condition(state: State) -> str:
    messages = state.get("messages", [])
    if any(isinstance(m, ToolMessage) for m in messages):
        return "tools"
    return "__end__"


graph_builder = StateGraph(State)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("select_tools", select_tools)

tool_node = ToolNode(tools=valid_tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "assistant",
    tools_condition,
    path_map={
        "tools": "tools",
        "__end__": END
    }
)

graph_builder.add_edge("tools", "select_tools")
graph_builder.add_edge("select_tools", "assistant")
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