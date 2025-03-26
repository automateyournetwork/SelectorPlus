import os
import json
import time
import select
import logging
import subprocess
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def spawn_server() -> subprocess.Popen:
    load_dotenv()  # Load environment variables from .env
    selector_url = os.getenv("SELECTOR_URL")
    selector_api_key = os.getenv("SELECTOR_AI_API_KEY")
    proc = subprocess.Popen(
        [
            "docker", "run", "-i", "--rm",
            f"-e", f"SELECTOR_URL={selector_url}",
            f"-e", f"SELECTOR_AI_API_KEY={selector_api_key}",
            "selector-mcp"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return proc

def send_request(proc: subprocess.Popen, request: dict):
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    rlist, _, _ = select.select([proc.stdout], [], [], 20.0)
    if rlist:
        try:
            return json.loads(proc.stdout.readline().strip())
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}
    return {"error": "Timed out waiting for server response."}

def call_tool(proc: subprocess.Popen, tool_name: str, payload: dict = None):
    return send_request(proc, {"method": "tools/call", "tool_name": tool_name, **(payload or {})})

def interactive_mode(proc: subprocess.Popen):
    print("\n=== Selector CLI ===")
    print("Type your question (or 'exit'):\n")
    while True:
        user_input = input("You> ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if user_input:
            response = call_tool(proc, "ask_selector", {"content": user_input})
            if response:
                if "error" in response:
                    print(f"Selector> Error: {response['error']}")
                elif "content" in response:
                    print(f"Selector> {response['content'].strip()}")
                else:
                    print(f"Selector> {response}")

def main():
    proc = spawn_server()
    if call_tool(proc, "ready") and call_tool(proc, "ready").get("status") == "ready":
        interactive_mode(proc)
    else:
        logger.error("Server did not respond with 'ready' status.")
    proc.terminate()
    proc.wait(timeout=5)

if __name__ == "__main__":
    main()