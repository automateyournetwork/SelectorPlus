from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse # StreamingResponse is not used in this non-streaming approach
from fastapi.staticfiles import StaticFiles
import httpx
import json
import uuid
import os

# --- Environment Variables ---
A2A_PORT = int(os.getenv("A2A_PORT", 10000))
# Ensure LANGGRAPH_URL uses http:// if not otherwise specified
langgraph_host = os.getenv("LANGGRAPH_HOST", "localhost:8000") # Example: Use separate host/port if needed
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", f"http://{langgraph_host}")
AGENT_ID = os.getenv("AGENT_ID", "MCpyATS") # Good practice to allow overriding agent ID
AGENT_CARD_PATH = os.getenv("AGENT_CARD_PATH", "/a2a/agent.json") # Allow configuring agent card path

app = FastAPI(
    title="LangGraph A2A Adapter",
    description="Adapts LangGraph agent interactions to the A2A protocol.",
    version="1.0.0",
)

# In-memory storage for conversation_id -> thread_id mapping
# For production, consider a more persistent store (e.g., Redis, DB)
threads = {}

# 👇 Serve the .well-known directory statically
app.mount("/.well-known", StaticFiles(directory="/.well-known"), name="well-known")


@app.get("/.well-known/agent.json", tags=["A2A Discovery"])
async def agent_card():
    """Serves the agent's capability description."""
    try:
        with open(AGENT_CARD_PATH) as f:
            content = json.load(f)
            # Ensure the URL in the card reflects the actual service location if possible
            # This might require dynamic generation or configuration management
            # For now, we just serve the static file
            return JSONResponse(content=content)
    except FileNotFoundError:
        print(f"ERROR: Agent card not found at {AGENT_CARD_PATH}")
        return JSONResponse(
            status_code=404,
            content={"error": "Agent configuration file not found."}
        )
    except json.JSONDecodeError:
        print(f"ERROR: Agent card at {AGENT_CARD_PATH} is not valid JSON.")
        return JSONResponse(
            status_code=500,
            content={"error": "Agent configuration file is corrupted."}
        )
    except Exception as e:
        print(f"ERROR: Failed to serve agent card: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error serving agent card."}
        )

@app.post("/tasks/send", tags=["A2A Task Execution"])
async def send_task(request: Request):
    """Receives a task via A2A protocol and interacts with LangGraph."""
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
        )

    # --- Validate JSON-RPC Structure ---
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "method" not in payload or "params" not in payload or "id" not in payload:
         # Although A2A /tasks/send doesn't strictly enforce 'method', including it improves JSON-RPC adherence
         # We'll assume method is implicitly 'send' for this endpoint
         # Focusing on required params and id for A2A
         if "params" not in payload or "id" not in payload:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": payload.get("id")}
            )

    request_id = payload.get("id")
    params = payload.get("params", {})
    message = params.get("content")
    conversation_id = params.get("conversation_id", str(uuid.uuid4())) # Generate new conv ID if none provided

    if not message:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params: 'content' is required"}, "id": request_id}
        )

    print(f"Received task for conversation {conversation_id} (Request ID: {request_id})")

    # --- Get or Create LangGraph Thread ---
    thread_id = threads.get(conversation_id)
    if not thread_id:
        print(f"Creating new LangGraph thread for conversation {conversation_id}")
        async with httpx.AsyncClient(base_url=LANGGRAPH_URL) as client:
            try:
                # Note: Some LangGraph setups might not require assistant_id for thread creation
                # Adjust the payload as needed based on your LangGraph server's requirements
                thread_payload = {"assistant_id": AGENT_ID} if AGENT_ID else {} # Only send if AGENT_ID is set

                response = await client.post("/threads", json=thread_payload, timeout=10.0)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                thread_data = response.json()
                thread_id = thread_data.get("thread_id")

                if not thread_id:
                     print(f"ERROR: LangGraph thread creation response missing 'thread_id'. Response: {thread_data}")
                     return JSONResponse(
                        status_code=500,
                        content={"jsonrpc": "2.0", "error": {"code": -32000, "message": "LangGraph thread creation failed: Invalid response format"}, "id": request_id}
                    )

                threads[conversation_id] = thread_id
                print(f"Created LangGraph thread {thread_id} for conversation {conversation_id}")

            except httpx.RequestError as e:
                print(f"ERROR: Could not connect to LangGraph at {LANGGRAPH_URL}/threads: {e}")
                return JSONResponse(
                    status_code=503, # Service Unavailable
                    content={"jsonrpc": "2.0", "error": {"code": -32000, "message": f"LangGraph connection error: {e}"}, "id": request_id}
                )
            except httpx.HTTPStatusError as e:
                print(f"ERROR: LangGraph thread creation failed with status {e.response.status_code}. Response: {e.response.text}")
                return JSONResponse(
                    status_code=500,
                    content={"jsonrpc": "2.0", "error": {"code": -32000, "message": f"LangGraph thread creation failed (HTTP {e.response.status_code})"}, "id": request_id}
                )
            except Exception as e: # Catch other potential errors during thread creation
                 print(f"ERROR: Unexpected error during LangGraph thread creation: {e}")
                 return JSONResponse(
                    status_code=500,
                    content={"jsonrpc": "2.0", "error": {"code": -32000, "message": f"Internal server error during thread creation"}, "id": request_id}
                )
    else:
         print(f"Using existing LangGraph thread {thread_id} for conversation {conversation_id}")


    # --- Call LangGraph Run Stream Endpoint ---
    async with httpx.AsyncClient(base_url=LANGGRAPH_URL) as client:
        try:
            langgraph_payload = {
                "assistant_id": AGENT_ID, # May or may not be needed depending on LangGraph setup
                "input": {"messages": [{"role": "user", "type": "human", "content": message}]} # Use 'role':'user' too for clarity if supported
            }
            # Remove assistant_id if not set
            if not AGENT_ID:
                del langgraph_payload["assistant_id"]

            print(f"Calling LangGraph: POST /threads/{thread_id}/runs/stream")
            resp = await client.post(
                f"/threads/{thread_id}/runs/stream",
                json=langgraph_payload,
                timeout=60.0 # Increased timeout for potentially long runs
            )
            resp.raise_for_status() # Check for HTTP errors

            text = resp.text.strip()
            print(f"🔥 Full LangGraph /runs/stream response for thread {thread_id}:")
            print(text) # Log the raw stream data for debugging

            # --- Process Stream Data to Collect AI Responses ---
            ai_responses = []
            lines = text.split("\n")
            for line in lines:
                line = line.strip() # Handle potential whitespace
                if line.startswith("data:"):
                    try:
                        # Handle potential empty data chunks like "data: \n"
                        data_content = line[5:].strip()
                        if not data_content:
                            continue

                        json_data = json.loads(data_content)

                        # LangGraph stream events can have different structures.
                        # Common patterns include a top-level 'messages' list
                        # or events related to specific nodes containing message data.
                        # Adapt this logic based on your specific LangGraph output stream format.

                        # Example 1: Check for messages directly in the event data
                        if isinstance(json_data, dict) and "messages" in json_data:
                           for msg in json_data.get("messages", []):
                               # Check for role 'assistant' or type 'ai'
                               is_ai_message = msg.get("type") == "ai" or msg.get("role") == "assistant"
                               if is_ai_message and "content" in msg and isinstance(msg["content"], str):
                                    ai_responses.append(msg["content"])

                        # Example 2: Check within node events (adjust node names as needed)
                        elif isinstance(json_data, dict) and "event" in json_data and json_data["event"] == "on_chat_model_stream":
                             chunk = json_data.get("data", {}).get("chunk")
                             if chunk and isinstance(chunk, dict) and "content" in chunk:
                                 # Check if it's from AI (might need more context depending on graph)
                                 # This assumes any content in this event is AI response part
                                 ai_responses.append(chunk["content"])

                        # Add more parsing logic here if your stream structure is different

                    except json.JSONDecodeError as json_err:
                        print(f"⚠️ Warning: Could not decode JSON from stream line: '{line}'. Error: {json_err}")
                        continue # Skip malformed lines
                    except Exception as parse_err:
                        print(f"⚠️ Warning: Error processing stream line: '{line}'. Error: {parse_err}")
                        continue # Skip lines that cause other processing errors

            # --- Format and Return Response ---
            if ai_responses:
                # Decide whether to join chunks or take the last one.
                # Joining is often correct for LLM streams that output tokens sequentially.
                final_response = "".join(ai_responses)
                print(f"✅ Successfully processed stream for conversation {conversation_id}. Final Response Length: {len(final_response)}")
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": {
                        "response": final_response,
                        "conversation_id": conversation_id
                    },
                    "id": request_id
                })
            else:
                # This case means the stream finished but no AI messages were found/extracted
                print(f"⚠️ Warning: No AI messages found in LangGraph stream for conversation {conversation_id}.")
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "result": { # Still a successful call, just no AI output found
                        "response": "No response content received from agent.", # Provide a clearer message
                        "conversation_id": conversation_id
                        },
                    "id": request_id
                })

        except httpx.RequestError as e:
             print(f"ERROR: Could not connect to LangGraph at {LANGGRAPH_URL}/threads/{thread_id}/runs/stream: {e}")
             return JSONResponse(
                status_code=503, # Service Unavailable
                content={"jsonrpc": "2.0", "error": {"code": -32000, "message": f"LangGraph connection error during run: {e}"}, "id": request_id}
            )
        except httpx.HTTPStatusError as e:
            print(f"ERROR: LangGraph run failed with status {e.response.status_code}. Response: {e.response.text}")
            error_message = f"LangGraph run failed (HTTP {e.response.status_code})"
            try:
                # Try to parse error details from LangGraph response
                detail = e.response.json().get("detail", e.response.text)
                error_message += f": {detail}"
            except Exception:
                 error_message += f": {e.response.text}" # Fallback to raw text

            return JSONResponse(
                status_code=500,
                content={"jsonrpc": "2.0", "error": {"code": -32000, "message": error_message}, "id": request_id}
            )
        except Exception as e:
            print(f"ERROR: Unexpected error during LangGraph run or response processing: {e}")
            import traceback
            traceback.print_exc() # Log the full traceback for debugging
            return JSONResponse(
                status_code=500,
                content={"jsonrpc": "2.0", "error": {"code": -32000, "message": "Internal server error during agent execution"}, "id": request_id}
            )

# --- Optional: Add root endpoint for health check ---
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "A2A Adapter is running"}

# --- Optional: Add command-line execution ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting A2A Adapter on port {A2A_PORT}")
    print(f"Connecting to LangGraph at: {LANGGRAPH_URL}")
    print(f"Using Agent ID: {AGENT_ID or 'Not Specified'}")
    print(f"Serving Agent Card from: {AGENT_CARD_PATH}")
    # Ensure the agent card file exists before starting
    if not os.path.exists(AGENT_CARD_PATH):
         print(f"\n!!! WARNING: Agent card file not found at {AGENT_CARD_PATH} !!!")
         print("!!! The /.well-known/agent.json endpoint will return 404. !!!\n")

    uvicorn.run(app, host="0.0.0.0", port=A2A_PORT)