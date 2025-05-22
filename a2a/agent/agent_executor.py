import json
import os
from uuid import uuid4
import httpx

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TaskStatusUpdateEvent, TaskStatus, TaskState
from a2a.utils import new_agent_text_message

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://host.docker.internal:2024")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "SelectorPlus")

class LangGraphAgentExecutor(AgentExecutor):
    """A2A AgentExecutor wrapper for LangGraph."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task

        # Use provided IDs if present
        task_id = task.id if task and task.id else context.task_id or str(uuid4())
        context_id = task.contextId if task and task.contextId else context.context_id or str(uuid4())
        
        # Create a fallback task if needed (but re-use IDs from context)
        if not task:
            from a2a.types import Task
            task = Task(
                id=task_id,
                contextId=context_id,
                kind="task",
                status=TaskStatus(state=TaskState.submitted),
                history=[]
            )
        
        context_id = task.contextId or "unknown"
        task_id = task.id or "unknown"

        try:
            print(f"🔁 Sending request to LangGraph at {LANGGRAPH_URL}")
            async with httpx.AsyncClient(base_url=LANGGRAPH_URL, timeout=600) as client:
                # Step 1: Create thread
                print("📤 Creating thread...")
                thread_resp = await client.post("/threads", json={"assistant_id": ASSISTANT_ID})
                thread_resp.raise_for_status()
                thread_id = thread_resp.json().get("thread_id")
                print(f"✅ Thread created: {thread_id}")

                if not thread_id:
                    raise RuntimeError("❌ Failed to create LangGraph thread")

                content_chunks = []

                # Step 2: Stream execution
                print("🚀 Sending /runs/stream request...")
                async with client.stream("POST", f"/threads/{thread_id}/runs/stream", json={
                    "input": {
                        "messages": [{"role": "user", "type": "human", "content": query}],
                        "metadata": {},
                    },
                    "assistant_id": ASSISTANT_ID,
                }) as response:
                    print("🔄 Streaming response...")
                    async for line in response.aiter_lines():
                        print(f"📥 Raw line: {line}")
                        if not line.startswith("data:"):
                            print("⚠️ Skipping non-data line")
                            continue
                        try:
                            payload = json.loads(line[5:].strip())
                            print("🧾 Parsed payload:", json.dumps(payload, indent=2))

                            # ✅ Capture live stream chunks
                            if isinstance(payload.get("content"), str) and payload["content"].strip():
                                print("✅ Streaming chunk detected")
                                content_chunks.append(payload["content"])

                            # ✅ Parse final assistant message (type = ai)
                            elif isinstance(payload.get("messages"), list):
                                print("🔍 Scanning messages array...")
                                for msg in reversed(payload["messages"]):
                                    if msg.get("type") == "ai" and isinstance(msg.get("content"), str):
                                        print("✅ Found final assistant message")
                                        content_chunks.append(msg["content"])
                                        break
                        except Exception as e:
                            print(f"❌ JSON decode error: {e}")
                            continue

                final_content = "\n".join(content_chunks).strip()
                print("🧾 Final concatenated content:", repr(final_content))

                if not final_content:
                    print("🛑 No content extracted. All chunks:", content_chunks)
                    raise RuntimeError("❌ No usable content returned from LangGraph")

                print("✅ Task completed successfully. Sending TaskStatusUpdateEvent.")
                event_queue.enqueue_event(TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=new_agent_text_message(final_content, context_id, task_id),
                    ),
                    contextId=context_id,
                    taskId=task_id,
                    final=True,
                ))

        except Exception as e:
            error_msg = f"🔥 Exception: {e}"
            print(error_msg)
            event_queue.enqueue_event(TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(error_msg, context_id, task_id),
                ),
                contextId=context_id,
                taskId=task_id,
                final=True,
            ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not implemented in this agent.")
