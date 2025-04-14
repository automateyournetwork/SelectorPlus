import os
import sys
import json
import time
import base64
import logging
import threading
import tempfile
import re
from typing import Dict, Any, List, Union
import vl_convert as vlc
from pydantic import BaseModel, Field, ValidationError

# --- Add an Event for signalling exit ---
exit_signal = threading.Event()

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger("vegalite_server")

# --- Define output directory inside the container ---
CONTAINER_OUTPUT_DIR = "/output"
try:
    os.makedirs(CONTAINER_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Ensured output directory exists: {CONTAINER_OUTPUT_DIR}")
except OSError as e:
    logger.error(f"Could not create output directory {CONTAINER_OUTPUT_DIR}: {e}. Check permissions and volume mount.", exc_info=True)

# --- Pydantic Input Models ---
class VegaLiteSaveDataInput(BaseModel):
    name: str = Field(description="Unique name used to identify the dataset (used as filename).")
    data: List[Dict[str, Union[str, int, float]]] = Field(
        description=(
            "A list of dictionaries where each entry represents a row of data. "
            "All dictionaries should use the same keys. Values must be strings, integers, or floats."
        )
    )

class VegaLiteVisualizeDataInput(BaseModel):
    data_name: str = Field(description="Name of the previously saved dataset (must match the 'name' used in save_data).")
    vegalite_specification: str = Field(
        description=(
            "Full Vega-Lite v5 JSON specification as a string. "
            "Must NOT include the 'data' field — it will be injected automatically using the saved dataset."
        )
    )

# --- Define temp directory for JSON data files ---
TEMP_DATA_DIR = os.path.join(tempfile.gettempdir(), "vegalite_mcp_data")
os.makedirs(TEMP_DATA_DIR, exist_ok=True)
logger.info(f"Using temporary data directory for JSON: {TEMP_DATA_DIR}")

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\-]+', '_', name)
    return name[:100]

def get_data_filepath(table_name: str) -> str:
    safe_name = sanitize_filename(table_name)
    return os.path.join(TEMP_DATA_DIR, f"{safe_name}.json")

def send_response(response_data: Dict[str, Any]):
    try:
        response = json.dumps(response_data) + "\n"
        sys.stdout.write(response)
        sys.stdout.flush()
        logger.debug(f"Sent response: {response.strip()}")
    except Exception as e:
        logger.error(f"Error sending response: {e}")

def save_data_tool(validated_args: VegaLiteSaveDataInput) -> Dict[str, Any]:
    try:
        table_name = validated_args.name
        raw_data = validated_args.data

        # Validate each row has non-empty keys and values
        cleaned_data = [row for row in raw_data if isinstance(row, dict) and all(k and v is not None for k, v in row.items())]
        if not cleaned_data:
            logger.error(f"❌ No valid rows found in input data: {json.dumps(raw_data, indent=2)}")
            return {"error": f"No valid rows to save for table '{table_name}'"}

        filepath = get_data_filepath(table_name)
        
        
        logger.info(f"Original raw_data: {json.dumps(raw_data, indent=2)}")
        logger.info(f"Cleaned data: {json.dumps(cleaned_data, indent=2)}")
        
        with open(filepath, 'w') as f:
            json.dump(cleaned_data, f)

        return {"result": [{"type": "text", "text": f"✅ Data saved for table '{table_name}'"}]}
    except Exception as e:
        logger.error(f"Error in save_data_tool writing to {filepath}: {e}", exc_info=True)
        return {"error": f"Failed to save data for {table_name}: {e}"}

def visualize_data_tool(validated_args: VegaLiteVisualizeDataInput) -> Dict[str, Any]:
    """
    Generates a Vega-Lite visualization from saved JSON data and saves it
    as a PNG file to the designated output directory (/output).
    Returns a confirmation message with the path inside the container.
    Includes enhanced logging and checks around PNG generation.
    """
    # --- LOGGING LINE REMOVED FROM HERE ---
    table_name = validated_args.data_name
    spec_string = validated_args.vegalite_specification
    json_data_filepath = get_data_filepath(table_name)

    logger.info(f"Attempting to visualize data for table '{table_name}' from JSON file: {json_data_filepath}")

    try:
        if not os.path.exists(json_data_filepath):
            logger.error(f"JSON data file not found for table '{table_name}' at {json_data_filepath}")
            return {"error": f"Data file not found for table '{table_name}'. Was save_data called first?"}

        logger.info(f"Loading data from {json_data_filepath}")
        with open(json_data_filepath, 'r') as f:
            loaded_data = json.load(f)

        # --- LOGGING LINE MOVED HERE ---
        logger.info(f"Data loaded from file: {json.dumps(loaded_data, indent=2)}")

        # Parse Vega-Lite spec safely
        try:
            spec = json.loads(spec_string)
            if not isinstance(spec, dict):
                raise ValueError("Vega-Lite specification must be a dictionary.")
        except Exception as e:
            logger.error(f"Error parsing Vega-Lite spec: {e}", exc_info=True)
            return {"error": f"Error parsing Vega-Lite specification: {e}"}

        # Inject loaded data
        spec["data"] = {"values": loaded_data}

        # Log final Vega-Lite spec for debugging (ensure logging level is DEBUG)
        logger.debug("Final Vega-Lite spec (with embedded data):\n%s", json.dumps(spec, indent=2))

        # --- MODIFIED SECTION ---
        # Generate PNG with more checking
        png_binary_data = None
        try:
            logger.info(f"Calling vlc.vegalite_to_png for table '{table_name}'...")
            png_binary_data = vlc.vegalite_to_png(vl_spec=spec, scale=2)
            logger.info(f"vlc.vegalite_to_png call completed for table '{table_name}'.")
            logger.info(f"Size of generated PNG data: {len(png_binary_data) if png_binary_data else 'None'} bytes")
            if png_binary_data and len(png_binary_data) > 10: # Check header bytes
                logger.info(f"First 10 bytes of PNG data: {png_binary_data[:10]}")

            # Check if the generated data seems invalid (empty or too small)
            MIN_EXPECTED_PNG_SIZE = 100
            if not png_binary_data or len(png_binary_data) < MIN_EXPECTED_PNG_SIZE:
                logger.error(f"Generated PNG data is missing or suspiciously small ({len(png_binary_data) if png_binary_data else 0} bytes). Assuming rendering failed.")
                return {"error": f"Visualization rendering failed silently for table '{table_name}'. Output PNG data was empty or too small."}

        except Exception as render_e:
            logger.error(f"Error during vlc.vegalite_to_png: {render_e}", exc_info=True)
            return {"error": f"Failed during visualization rendering: {render_e}"}
        # --- END MODIFIED SECTION ---

        # Save file (only if png_binary_data is valid)
        png_filename = f"{sanitize_filename(table_name)}.png"
        save_filepath_in_container = os.path.join(CONTAINER_OUTPUT_DIR, png_filename)

        try:
            with open(save_filepath_in_container, 'wb') as f:
                f.write(png_binary_data)
            logger.info(f"Visualization successfully saved to file: {save_filepath_in_container}")
            return {"result": [{
                "type": "file_save_confirmation",
                "message": f"Visualization for '{table_name}' saved successfully.",
                "container_path": save_filepath_in_container
            }]}
        except Exception as write_e:
            logger.error(f"Error writing PNG file: {write_e}", exc_info=True)
            return {"error": f"Failed to save visualization file: {write_e}"}

    except FileNotFoundError:
        logger.error(f"JSON data file disappeared unexpectedly: {json_data_filepath}")
        return {"error": f"Data file not found for table '{table_name}'. Race condition or file deleted?"}
    except json.JSONDecodeError as json_e:
        logger.error(f"Invalid JSON data: {json_e}")
        return {"error": f"Failed to read saved data: Invalid JSON format."}
    except Exception as e:
        logger.error(f"Unexpected error in visualize_data_tool: {e}", exc_info=True)
        return {"error": f"Failed to visualize data: {e}"}

# --- JSON-RPC message handler ---
def handle_request(data: Dict[str, Any]):
    method = data.get("method")

    if method == "tools/call":
        tool = data.get("params", {}).get("name")
        arguments = data.get("params", {}).get("arguments")

        if not tool or arguments is None:
            send_response({"error": "Missing 'name' or 'arguments' in tool call parameters"})
            return

        log_args = list(arguments.keys()) if isinstance(arguments, dict) else type(arguments)
        logger.info(f"Received call for tool: {tool} with args (keys/type): {log_args}")

        try:
            if tool == "vegalite_save_data":
                validated_args = VegaLiteSaveDataInput(**arguments)
                send_response(save_data_tool(validated_args))
            elif tool == "vegalite_visualize_data":
                validated_args = VegaLiteVisualizeDataInput(**arguments)
                send_response(visualize_data_tool(validated_args)) # Calls the modified function
            else:
                logger.warning(f"Received call for unknown tool: {tool}")
                send_response({"error": f"Unknown tool: {tool}"})
        except ValidationError as e:
            logger.error(f"Validation Error for tool {tool}: {e}")
            send_response({"error": f"Invalid arguments for tool {tool}: {e}"})
        except Exception as e:
            logger.error(f"Unexpected error handling tool {tool}: {e}", exc_info=True)
            send_response({"error": f"Internal server error processing tool {tool}: {e}"})

    elif method == "tools/discover":
        logger.info("Processing tools/discover request")
        send_response({
            "result": [
                {
                    "name": "vegalite_save_data",
                    "description": (
                        "Save tabular data for future Vega-Lite visualizations. "
                        "The data must be a list of dictionaries with consistent keys and values that are strings, numbers, or floats. "
                        "For example: [{\"direction\": \"Input\", \"packets\": 1000}, {\"direction\": \"Output\", \"packets\": 1200}]."
                    ),
                    "parameters": VegaLiteSaveDataInput.model_json_schema()
                },
                {
                    "name": "vegalite_visualize_data",
                    "description": (
                        "Generate a Vega-Lite visualization from a previously saved dataset. "
                        "Takes a valid Vega-Lite v5 specification (as a string) and automatically injects the saved data "
                        "into the chart. Saves the output as a PNG to /output."
                    ),
                    "parameters": VegaLiteVisualizeDataInput.model_json_schema()
                }
            ]
        })
        logger.info("Discovery response sent.")
    else:
        logger.warning(f"Received unknown method: {method}")
        send_response({"error": f"Unknown method: {method}"})

# --- Stdin monitor loop (with exit signal) ---
# (monitor_stdin function remains the same)
def monitor_stdin():
    logger.info("Stdin monitor thread started.")
    while not exit_signal.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                logger.info("Stdin closed (EOF detected), signaling exit.")
                exit_signal.set()
                break

            line = line.strip()
            if not line:
                time.sleep(0.05)
                continue

            logger.debug(f"Received line from stdin: {line[:100]}...")
            try:
                data = json.loads(line)
                handle_request(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} on line: {line}")
            except Exception as handle_e:
                logger.error(f"Error handling request: {handle_e}", exc_info=True)

        except Exception as loop_e:
            logger.error(f"Exception in monitor_stdin loop: {loop_e}", exc_info=True)
            exit_signal.set()
            break
    logger.info("Stdin monitor thread finished.")


# --- Entry point (with exit signal handling) ---
# (if __name__ == "__main__": block remains the same)
if __name__ == "__main__":
    logger.info("Starting Vega-Lite Tool Server")
    if "--oneshot" in sys.argv:
        logger.info("Running in --oneshot mode.")
        try:
            line = sys.stdin.readline().strip()
            if line:
                data = json.loads(line)
                handle_request(data)
            else:
                logger.warning("Received empty input in --oneshot mode.")
        except Exception as e:
            logger.error(f"Oneshot error: {e}", exc_info=True)
            # Try sending error back even in oneshot
            try:
                send_response({"error": str(e)})
            except Exception:
                 pass # Ignore if stdout is closed
        logger.info("Oneshot execution finished.")
    else:
        logger.info("Running in persistent server mode (using file storage).")
        stdin_thread = threading.Thread(target=monitor_stdin, daemon=True)
        stdin_thread.start()
        logger.info("Main thread waiting for exit signal (stdin close or Ctrl+C)...")
        try:
            exit_signal.wait()
            logger.info("Exit signal received. Proceeding to shutdown.")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Signaling exit.")
            exit_signal.set()
        finally:
            logger.info("Vega-Lite Tool Server main process terminating.")