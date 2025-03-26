import asyncio
import logging
import json
import subprocess

logging.basicConfig(level=logging.INFO)

async def test_stdio_client():
    container_name = "selector-mcp"  # Replace with your container name
    command = ["docker", "exec", "-i", container_name, "python3", "mcp_server.py", "--oneshot"]

    logging.info("Testing with subprocess.run and docker exec...")
    try:
        # Construct the JSON input to send
        input_data = {
            "method": "tools/call",
            "params": {
                "name": "ask_selector",
                "arguments": {"content": "test"}
            }
        }
        input_json = json.dumps(input_data) + "\n"  # Add newline

        # Run docker exec with subprocess.run
        process = subprocess.run(
            command,
            input=input_json,
            capture_output=True,
            text=True,
        )

        # Log the output
        logging.info(f"STDOUT: {process.stdout}")
        logging.info(f"STDERR: {process.stderr}")

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
                        logging.warning(f"Ignoring invalid JSON line: {line}")
            if json_lines:
                data = json_lines[0]
                logging.info(f"Parsed JSON data: {data}")
            else:
                logging.error("No valid JSON found in output")
                raise Exception("No valid JSON found in output")

        else:
            logging.error("No output from subprocess")
            raise Exception("No output from subprocess")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_stdio_client())