import httpx
import json
import uuid
import os
import traceback
from datetime import datetime # Import datetime for timestamp
from fastapi import FastAPI, Request
# Using JSONResponse as we will return standard JSON structure
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Environment Variables ---
A2A_PORT = int(os.getenv("A2A_PORT", 10000))
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://host.docker.internal:2024")
AGENT_ID = os.getenv("AGENT_ID", "SelectorPlus")
AGENT_CARD_PATH = os.getenv("AGENT_CARD_PATH", "/a2a/.well-known/agent.json")

app = FastAPI(
    title="LangGraph A2A Adapter",
    description="Adapts LangGraph agent interactions to the A2A protocol (Conforming to common/types Task model).",
    version="1.2.0", # Bump version
)

threads = {}

app.mount("/.well-known", StaticFiles(directory="/a2a/.well-known"), name="well-known")

@app.get("/.well-known/agent.json", tags=["A2A Discovery"])
async def agent_card():
    # Returns standard JSON, no changes needed
    try:
        with open(AGENT_CARD_PATH) as f:
            content = json.load(f)
            return JSONResponse(content=content)
    except FileNotFoundError:
        print(f"ERROR: Agent card not found at {AGENT_CARD_PATH}")
        return JSONResponse(status_code=404, content={"error": "Agent configuration file not found."})
    except json.JSONDecodeError:
        print(f"ERROR: Agent card at {AGENT_CARD_PATH} is not valid JSON.")
        return JSONResponse(status_code=500, content={"error": "Agent configuration file is corrupted."})
    except Exception as e:
        print(f"ERROR: Failed to serve agent card: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error serving agent card."})


@app.post("/", tags=["A2A Task Execution"])
async def send_task(request: Request):
    """
    Receives task, interacts with LangGraph, returns standard JSONRPCResponse
    with result conforming to the Task model from common/types.py.
    The AI's final answer is placed in result.status.message.
    """
    task_param_id = None
    request_id = None
    conversation_id = None # Will be mapped to sessionId

    # --- Basic Request Parsing and Validation ---
    try:
        payload = await request.json()
        print("🟡 Incoming Payload:", json.dumps(payload, indent=2))
        request_id = payload.get("id")
    except Exception as e:
        print(f"ERROR: Failed to parse incoming request: {e}")
        return JSONResponse(status_code=400, content={"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": request_id})

    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "params" not in payload:
         return JSONResponse(status_code=400, content={"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request: Missing 'params' or invalid structure"}, "id": request_id})

    params = payload.get("params", {})
    task_param_id = params.get("id") # ID of the task being sent

    # Use sessionId from input, or conversation_id, or generate new
    session_id = params.get("sessionId") # Match the Task model field name
    if not session_id:
        session_id = params.get("conversation_id", str(uuid.uuid4().hex)) # Fallback

    message_content = None
    message_object = params.get("message")
    if isinstance(message_object, dict):
        message_parts = message_object.get("parts")
        if isinstance(message_parts, list) and len(message_parts) > 0:
            first_part = message_parts[0]
            if isinstance(first_part, dict) and first_part.get("type") == "text":
                message_content = first_part.get("text")

    # Prepare failed status structure conforming to TaskStatus model
    failed_status = {
        "state": "failed", # Use the TaskState enum value string
        "timestamp": datetime.now().isoformat() # Add timestamp
        # message could be added here too for errors
    }

    if not message_content:
        print(f"⚠️ Warning: Could not extract text message content for task {task_param_id}.")
        failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": "Invalid params: Missing or invalid 'message.parts[0].text'"}]}
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Invalid params: Missing or invalid 'message.parts[0].text'"},
                 # Include result object matching Task model on error
                "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None},
                "id": request_id
            }
        )

    print(f"Received task {task_param_id} for session {session_id} (Request ID: {request_id})")

    # --- Thread Management ---
    # Use session_id for tracking LangGraph threads now
    thread_id = threads.get(session_id)
    if not thread_id:
        print(f"Creating new LangGraph thread for session {session_id}, task {task_param_id}")
        async with httpx.AsyncClient(base_url=LANGGRAPH_URL) as client:
            try:
                thread_payload = {"assistant_id": AGENT_ID} if AGENT_ID else {}
                response = await client.post("/threads", json=thread_payload, timeout=20.0)
                response.raise_for_status()
                thread_data = response.json()
                thread_id = thread_data.get("thread_id")
                if not thread_id:
                     print(f"ERROR: LangGraph thread creation failed for task {task_param_id}.")
                     failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": "LangGraph thread creation failed: Invalid response format"}]}
                     return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "error": {"code": -32000, "message": "LangGraph thread creation failed: Invalid response format"}, "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None}, "id": request_id})
                threads[session_id] = thread_id # Store thread_id against session_id
                print(f"Created LangGraph thread {thread_id} for session {session_id}, task {task_param_id}")
            except Exception as e:
                 error_msg = f"Error during LangGraph thread creation: {e}"
                 print(f"ERROR: {error_msg} for task {task_param_id}")
                 failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": error_msg}]}
                 # Determine appropriate status code based on error type if possible
                 status_code = 503 if isinstance(e, httpx.RequestError) else 500
                 return JSONResponse(status_code=status_code, content={"jsonrpc": "2.0", "error": {"code": -32000, "message": error_msg}, "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None}, "id": request_id})
    else:
         print(f"Using existing LangGraph thread {thread_id} for session {session_id}, task {task_param_id}")


    # --- Call LangGraph Run Stream Endpoint ---
    try:
        async with httpx.AsyncClient(base_url=LANGGRAPH_URL) as client:
            langgraph_payload = {"input": {"messages": [{"role": "user", "type": "human", "content": message_content}]}}
            if AGENT_ID: langgraph_payload["assistant_id"] = AGENT_ID

            print(f"Calling LangGraph for task {task_param_id}: POST /threads/{thread_id}/runs/stream")
            resp = await client.post(f"/threads/{thread_id}/runs/stream", json=langgraph_payload, timeout=90.0)
            resp.raise_for_status()

            text = resp.text.strip()
            # print(f"🔥 Full LangGraph stream response task {task_param_id}:\n{text}")

            # --- Process Stream Data (same logic as before to find the final string) ---
            final_response_content = None
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("data:"):
                    try:
                        data_content = line[5:].strip();
                        if not data_content: continue
                        json_data = json.loads(data_content); current_content = None
                        if isinstance(json_data, dict) and "content" in json_data and isinstance(json_data["content"], str) and json_data["content"].strip(): current_content = json_data["content"]
                        elif isinstance(json_data, dict) and "content" in json_data and isinstance(json_data["content"], list) and len(json_data["content"]) > 0 and isinstance(json_data["content"][0], str) and json_data["content"][0].strip(): current_content = json_data["content"][0]
                        elif isinstance(json_data, dict) and "messages" in json_data:
                           for msg in reversed(json_data.get("messages", [])):
                               is_ai = msg.get("type") == "ai" or msg.get("role") == "assistant";
                               if is_ai and "content" in msg and isinstance(msg["content"], str) and msg["content"].strip(): current_content = msg["content"]; break
                        elif isinstance(json_data, dict) and json_data.get("event") == "on_chat_model_stream":
                             chunk = json_data.get("data", {}).get("chunk");
                             if chunk and isinstance(chunk, dict) and "content" in chunk and chunk["content"].strip(): current_content = chunk["content"]
                        if current_content: final_response_content = current_content
                    except Exception as parse_err: print(f"⚠️ Warning [Task {task_param_id}]: Error processing stream line: '{line}'. Error: {parse_err}")


            # --- Format and Return SUCCESS Response CONFORMING TO Task MODEL ---
            final_status_object = {
                "state": "completed",
                "timestamp": datetime.now().isoformat()
            }
            result_payload = {
                 "id": task_param_id,
                 "status": final_status_object,
                 "sessionId": session_id, # Use the correct field name
                 "artifacts": None,       # Explicitly include optional fields as None
                 "history": None,
                 "metadata": None
             }

            if final_response_content:
                print(f"✅ Successfully processed stream for task {task_param_id}. Placing answer in status.message.")
                # Package final AI response into the status message
                final_status_object["message"] = {
                     "role": "agent",
                     "parts": [{"type": "text", "text": final_response_content}]
                }
            else:
                print(f"⚠️ Warning [Task {task_param_id}]: No final AI message content captured. Sending default status message.")
                # Include a default message in status if no specific AI response found
                final_status_object["message"] = {
                     "role": "agent",
                     "parts": [{"type": "text", "text": "Agent processed the request but no text content was extracted from the final response."}]
                }

            # Construct the final JSON-RPC response payload
            response_payload_to_send = {
                "jsonrpc": "2.0",
                "result": result_payload, # The result object conforms to the Task model
                "id": request_id
            }

            # Debug print the final payload
            print(f"🔵 DEBUG: Adapter sending success payload (conforming to Task): {json.dumps(response_payload_to_send)}")
            return JSONResponse(content=response_payload_to_send)

    # --- Handle Exceptions during LangGraph RUN ---
    # Return standard JSON-RPC errors, including result object where possible
    except Exception as e:
        error_message = f"Error during LangGraph run/processing: {e}"
        error_code = -32000 # Internal error default
        status_code = 500   # Internal error default

        if isinstance(e, httpx.RequestError):
            error_message = f"LangGraph connection error during run: {e}"
            status_code = 503
        elif isinstance(e, httpx.HTTPStatusError):
            error_message = f"LangGraph run failed (HTTP {e.response.status_code})"
            try: detail = e.response.json().get("detail", e.response.text); error_message += f": {detail}"
            except Exception: error_message += f": {e.response.text}"
            # status_code = e.response.status_code # Or keep 500? Let's keep 500 for internal failure indication
        else:
             traceback.print_exc() # Log unexpected errors fully

        print(f"ERROR: {error_message} for task {task_param_id}")
        failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": error_message}]}
        return JSONResponse(
            status_code=status_code,
            content={
                "jsonrpc": "2.0",
                "error": {"code": error_code, "message": error_message},
                "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None},
                "id": request_id
            }
        )


async def discover_agent(agent_url: str) -> dict | None:
    """Fetch another agent's card from the /.well-known/ endpoint."""

    # Strip invisible characters (like U+200B, U+2060, etc.)
    agent_url = re.sub(r"[\u200B-\u200F\u2060-\u206F]", "", agent_url.strip())

    # First ensure the URL has a protocol
    if not (agent_url.startswith("http://") or agent_url.startswith("https://")):
        agent_url = "http://" + agent_url

    # Then ensure it ends with the correct path
    if not agent_url.endswith("/.well-known/agent.json"):
        agent_url = agent_url.rstrip("/") + "/.well-known/agent.json"

    try:
        print(f"🔍 Discovering agent at {agent_url}")
        resp = await httpx.AsyncClient().get(agent_url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to discover agent at {agent_url}: {e}")
        return None


async def send_message_to_agent(agent_card: dict, content: str, session_id: str | None = None):
    """Send a standard A2A message to another agent conforming to Google A2A Task model."""
    endpoint = agent_card.get("methods", {}).get("send")
    if not endpoint:
        base_url = agent_card.get("endpoint", "").rstrip("/")
        if not base_url:
            print("❌ Error: No endpoint found in agent card")
            return None
            
        # Fix: More robust protocol checking and addition
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            base_url = "http://" + base_url
        endpoint = base_url + "/"
    
    # Additional check to ensure endpoint has a protocol
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        endpoint = "http://" + endpoint

    payload = {
        "jsonrpc": "2.0",
        "method": "send",
        "params": {
            "id": str(uuid.uuid4()),
            "sessionId": session_id or str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": content}]
            }
        },
        "id": str(uuid.uuid4())
    }

    try:
        print(f"📨 Sending message to {endpoint}: {content}")
        resp = await httpx.AsyncClient().post(endpoint, json=payload, timeout=20.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to send message to agent: {e}")
        return None


@app.post("/send_to_peer", tags=["Agent Communication"])
async def send_to_peer(request: Request):
    """
    Sends a message to another public A2A agent using discovery + A2A Task model.
    Expects JSON with 'agent_url' and 'message'.
    """
    try:
        data = await request.json()
        agent_url = data["agent_url"]
        message = data["message"]
        session_id = data.get("session_id")  # Optional

        agent_card = await discover_agent(agent_url)
        if not agent_card:
            return JSONResponse(status_code=400, content={"error": "Agent discovery failed."})

        response = await send_message_to_agent(agent_card, message, session_id=session_id)
        return JSONResponse(content=response)
    except Exception as e:
        print(f"ERROR in /send_to_peer: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal error: {e}"})

# --- Health Check ---
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "A2A Adapter is running"}

import httpx
import json
import uuid
import os
import re
import traceback
from datetime import datetime # Import datetime for timestamp
from fastapi import FastAPI, Request
# Using JSONResponse as we will return standard JSON structure
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Environment Variables ---
A2A_PORT = int(os.getenv("A2A_PORT", 10000))
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://host.docker.internal:2024")
AGENT_ID = os.getenv("AGENT_ID", "MCpyATS")
AGENT_CARD_PATH = os.getenv("AGENT_CARD_PATH", "/a2a/.well-known/agent.json")

app = FastAPI(
    title="LangGraph A2A Adapter",
    description="Adapts LangGraph agent interactions to the A2A protocol (Conforming to common/types Task model).",
    version="1.2.0", # Bump version
)

threads = {}

app.mount("/.well-known", StaticFiles(directory="/a2a/.well-known"), name="well-known")

@app.get("/.well-known/agent.json", tags=["A2A Discovery"])
async def agent_card():
    # Returns standard JSON, no changes needed
    try:
        with open(AGENT_CARD_PATH) as f:
            content = json.load(f)
            return JSONResponse(content=content)
    except FileNotFoundError:
        print(f"ERROR: Agent card not found at {AGENT_CARD_PATH}")
        return JSONResponse(status_code=404, content={"error": "Agent configuration file not found."})
    except json.JSONDecodeError:
        print(f"ERROR: Agent card at {AGENT_CARD_PATH} is not valid JSON.")
        return JSONResponse(status_code=500, content={"error": "Agent configuration file is corrupted."})
    except Exception as e:
        print(f"ERROR: Failed to serve agent card: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error serving agent card."})


@app.post("/", tags=["A2A Task Execution"])
async def send_task(request: Request):
    """
    Receives task, interacts with LangGraph, returns standard JSONRPCResponse
    with result conforming to the Task model from common/types.py.
    The AI's final answer is placed in result.status.message.
    """
    task_param_id = None
    request_id = None
    conversation_id = None # Will be mapped to sessionId

    # --- Basic Request Parsing and Validation ---
    try:
        payload = await request.json()
        print("🟡 Incoming Payload:", json.dumps(payload, indent=2))
        request_id = payload.get("id")
    except Exception as e:
        print(f"ERROR: Failed to parse incoming request: {e}")
        return JSONResponse(status_code=400, content={"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": request_id})

    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "params" not in payload:
         return JSONResponse(status_code=400, content={"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request: Missing 'params' or invalid structure"}, "id": request_id})

    params = payload.get("params", {})
    task_param_id = params.get("id") # ID of the task being sent

    # Use sessionId from input, or conversation_id, or generate new
    session_id = params.get("sessionId") # Match the Task model field name
    if not session_id:
        session_id = params.get("conversation_id", str(uuid.uuid4().hex)) # Fallback

    message_content = None
    message_object = params.get("message")
    if isinstance(message_object, dict):
        message_parts = message_object.get("parts")
        if isinstance(message_parts, list) and len(message_parts) > 0:
            first_part = message_parts[0]
            if isinstance(first_part, dict) and first_part.get("type") == "text":
                message_content = first_part.get("text")

    # Prepare failed status structure conforming to TaskStatus model
    failed_status = {
        "state": "failed", # Use the TaskState enum value string
        "timestamp": datetime.now().isoformat() # Add timestamp
        # message could be added here too for errors
    }

    if not message_content:
        print(f"⚠️ Warning: Could not extract text message content for task {task_param_id}.")
        failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": "Invalid params: Missing or invalid 'message.parts[0].text'"}]}
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Invalid params: Missing or invalid 'message.parts[0].text'"},
                 # Include result object matching Task model on error
                "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None},
                "id": request_id
            }
        )

    print(f"Received task {task_param_id} for session {session_id} (Request ID: {request_id})")

    # --- Thread Management ---
    # Use session_id for tracking LangGraph threads now
    thread_id = threads.get(session_id)
    if not thread_id:
        print(f"Creating new LangGraph thread for session {session_id}, task {task_param_id}")
        async with httpx.AsyncClient(base_url=LANGGRAPH_URL) as client:
            try:
                thread_payload = {"assistant_id": AGENT_ID} if AGENT_ID else {}
                response = await client.post("/threads", json=thread_payload, timeout=20.0)
                response.raise_for_status()
                thread_data = response.json()
                thread_id = thread_data.get("thread_id")
                if not thread_id:
                     print(f"ERROR: LangGraph thread creation failed for task {task_param_id}.")
                     failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": "LangGraph thread creation failed: Invalid response format"}]}
                     return JSONResponse(status_code=500, content={"jsonrpc": "2.0", "error": {"code": -32000, "message": "LangGraph thread creation failed: Invalid response format"}, "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None}, "id": request_id})
                threads[session_id] = thread_id # Store thread_id against session_id
                print(f"Created LangGraph thread {thread_id} for session {session_id}, task {task_param_id}")
            except Exception as e:
                 error_msg = f"Error during LangGraph thread creation: {e}"
                 print(f"ERROR: {error_msg} for task {task_param_id}")
                 failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": error_msg}]}
                 # Determine appropriate status code based on error type if possible
                 status_code = 503 if isinstance(e, httpx.RequestError) else 500
                 return JSONResponse(status_code=status_code, content={"jsonrpc": "2.0", "error": {"code": -32000, "message": error_msg}, "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None}, "id": request_id})
    else:
         print(f"Using existing LangGraph thread {thread_id} for session {session_id}, task {task_param_id}")


    # --- Call LangGraph Run Stream Endpoint ---
    try:
        async with httpx.AsyncClient(base_url=LANGGRAPH_URL) as client:
            langgraph_payload = {"input": {"messages": [{"role": "user", "type": "human", "content": message_content}]}}
            if AGENT_ID: langgraph_payload["assistant_id"] = AGENT_ID

            print(f"Calling LangGraph for task {task_param_id}: POST /threads/{thread_id}/runs/stream")
            resp = await client.post(f"/threads/{thread_id}/runs/stream", json=langgraph_payload, timeout=90.0)
            resp.raise_for_status()

            text = resp.text.strip()
            # print(f"🔥 Full LangGraph stream response task {task_param_id}:\n{text}")

            # --- Process Stream Data (same logic as before to find the final string) ---
            final_response_content = None
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("data:"):
                    try:
                        data_content = line[5:].strip();
                        if not data_content: continue
                        json_data = json.loads(data_content); current_content = None
                        if isinstance(json_data, dict) and "content" in json_data and isinstance(json_data["content"], str) and json_data["content"].strip(): current_content = json_data["content"]
                        elif isinstance(json_data, dict) and "content" in json_data and isinstance(json_data["content"], list) and len(json_data["content"]) > 0 and isinstance(json_data["content"][0], str) and json_data["content"][0].strip(): current_content = json_data["content"][0]
                        elif isinstance(json_data, dict) and "messages" in json_data:
                           for msg in reversed(json_data.get("messages", [])):
                               is_ai = msg.get("type") == "ai" or msg.get("role") == "assistant";
                               if is_ai and "content" in msg and isinstance(msg["content"], str) and msg["content"].strip(): current_content = msg["content"]; break
                        elif isinstance(json_data, dict) and json_data.get("event") == "on_chat_model_stream":
                             chunk = json_data.get("data", {}).get("chunk");
                             if chunk and isinstance(chunk, dict) and "content" in chunk and chunk["content"].strip(): current_content = chunk["content"]
                        if current_content: final_response_content = current_content
                    except Exception as parse_err: print(f"⚠️ Warning [Task {task_param_id}]: Error processing stream line: '{line}'. Error: {parse_err}")


            # --- Format and Return SUCCESS Response CONFORMING TO Task MODEL ---
            final_status_object = {
                "state": "completed",
                "timestamp": datetime.now().isoformat()
            }
            result_payload = {
                 "id": task_param_id,
                 "status": final_status_object,
                 "sessionId": session_id, # Use the correct field name
                 "artifacts": None,       # Explicitly include optional fields as None
                 "history": None,
                 "metadata": None
             }

            if final_response_content:
                print(f"✅ Successfully processed stream for task {task_param_id}. Placing answer in status.message.")
                # Package final AI response into the status message
                final_status_object["message"] = {
                     "role": "agent",
                     "parts": [{"type": "text", "text": final_response_content}]
                }
            else:
                print(f"⚠️ Warning [Task {task_param_id}]: No final AI message content captured. Sending default status message.")
                # Include a default message in status if no specific AI response found
                final_status_object["message"] = {
                     "role": "agent",
                     "parts": [{"type": "text", "text": "Agent processed the request but no text content was extracted from the final response."}]
                }

            # Construct the final JSON-RPC response payload
            response_payload_to_send = {
                "jsonrpc": "2.0",
                "result": result_payload, # The result object conforms to the Task model
                "id": request_id
            }

            # Debug print the final payload
            print(f"🔵 DEBUG: Adapter sending success payload (conforming to Task): {json.dumps(response_payload_to_send)}")
            return JSONResponse(content=response_payload_to_send)

    # --- Handle Exceptions during LangGraph RUN ---
    # Return standard JSON-RPC errors, including result object where possible
    except Exception as e:
        error_message = f"Error during LangGraph run/processing: {e}"
        error_code = -32000 # Internal error default
        status_code = 500   # Internal error default

        if isinstance(e, httpx.RequestError):
            error_message = f"LangGraph connection error during run: {e}"
            status_code = 503
        elif isinstance(e, httpx.HTTPStatusError):
            error_message = f"LangGraph run failed (HTTP {e.response.status_code})"
            try: detail = e.response.json().get("detail", e.response.text); error_message += f": {detail}"
            except Exception: error_message += f": {e.response.text}"
            # status_code = e.response.status_code # Or keep 500? Let's keep 500 for internal failure indication
        else:
             traceback.print_exc() # Log unexpected errors fully

        print(f"ERROR: {error_message} for task {task_param_id}")
        failed_status["message"] = {"role": "agent", "parts": [{"type": "text", "text": error_message}]}
        return JSONResponse(
            status_code=status_code,
            content={
                "jsonrpc": "2.0",
                "error": {"code": error_code, "message": error_message},
                "result": {"id": task_param_id, "status": failed_status, "sessionId": session_id, "artifacts": None, "history": None, "metadata": None},
                "id": request_id
            }
        )

# --- AGENT DISCOVERY AND MESSAGING TOOLS ---

async def discover_agent(agent_url: str) -> dict | None:
    """Fetch another agent's card from the /.well-known/ endpoint."""

    # Strip invisible characters (like U+200B, U+2060, etc.)
    agent_url = re.sub(r"[\u200B-\u200F\u2060-\u206F]", "", agent_url.strip())

    # First ensure the URL has a protocol
    if not (agent_url.startswith("http://") or agent_url.startswith("https://")):
        agent_url = "http://" + agent_url

    # Then ensure it ends with the correct path
    if not agent_url.endswith("/.well-known/agent.json"):
        agent_url = agent_url.rstrip("/") + "/.well-known/agent.json"

    try:
        print(f"🔍 Discovering agent at {agent_url}")
        resp = await httpx.AsyncClient().get(agent_url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to discover agent at {agent_url}: {e}")
        return None


async def send_message_to_agent(agent_card: dict, content: str, session_id: str | None = None):
    """Send a standard A2A message to another agent conforming to Google A2A Task model."""
    endpoint = agent_card.get("methods", {}).get("send")
    if not endpoint:
        base_url = agent_card.get("endpoint", "").rstrip("/")
        if not base_url:
            print("❌ Error: No endpoint found in agent card")
            return None
            
        # Fix: More robust protocol checking and addition
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            base_url = "http://" + base_url
        endpoint = base_url + "/"
    
    # Additional check to ensure endpoint has a protocol
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        endpoint = "http://" + endpoint

    payload = {
        "jsonrpc": "2.0",
        "method": "send",
        "params": {
            "id": str(uuid.uuid4()),
            "sessionId": session_id or str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": content}]
            }
        },
        "id": str(uuid.uuid4())
    }

    try:
        print(f"📨 Sending message to {endpoint}: {content}")
        resp = await httpx.AsyncClient().post(endpoint, json=payload, timeout=20.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ Failed to send message to agent: {e}")
        return None


@app.post("/send_to_peer", tags=["Agent Communication"])
async def send_to_peer(request: Request):
    """
    Sends a message to another public A2A agent using discovery + A2A Task model.
    Expects JSON with 'agent_url' and 'message'.
    """
    try:
        data = await request.json()
        agent_url = data["agent_url"]
        message = data["message"]
        session_id = data.get("session_id")  # Optional

        agent_card = await discover_agent(agent_url)
        if not agent_card:
            return JSONResponse(status_code=400, content={"error": "Agent discovery failed."})

        response = await send_message_to_agent(agent_card, message, session_id=session_id)
        return JSONResponse(content=response)
    except Exception as e:
        print(f"ERROR in /send_to_peer: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal error: {e}"})


# --- Health Check ---
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "A2A Adapter is running"}

A2A_PEER_AGENTS = os.getenv("A2A_PEER_AGENTS", "").split(",")
peer_agents = {}

@app.on_event("startup")
async def discover_peer_agents():
    global peer_agents
    print("🌐 Auto-discovering A2A peers...")
    for peer_url in A2A_PEER_AGENTS:
        if not peer_url.strip():
            continue
        discovered = await discover_agent(peer_url.strip())
        if discovered:
            peer_agents[peer_url] = discovered
            print(f"✅ Discovered: {peer_url}")
        else:
            print(f"⚠️ Failed to discover: {peer_url}")
    
    print("🌐 Peer discovery complete.")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting A2A Adapter on port {A2A_PORT}")
    print(f"Connecting to LangGraph at: {LANGGRAPH_URL}")
    # ... (rest of main) ...
    uvicorn.run(app, host="0.0.0.0", port=A2A_PORT)