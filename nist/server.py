# filename: server.py (aiohttp Version)
import os
import json
import time
import logging
import asyncio
import sys
import threading
from typing import Dict, Any, Optional
# import httpx # REMOVED
import aiohttp # ADDED
from aiohttp import ClientError, ClientResponseError # ADDED specific exceptions
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel, Field, ValidationError
import logging

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger("nist_server")

NVD_API_KEY = os.getenv("NVD_API_KEY")
if not NVD_API_KEY:
    logger.error("CRITICAL: NVD_API_KEY environment variable not found by os.getenv!")
    raise ValueError("NVD_API_KEY environment variable not set")
else:
    masked_key = NVD_API_KEY[:4] + "****" + NVD_API_KEY[-4:] if len(NVD_API_KEY) > 8 else "****"
    logger.info(f"NVD_API_KEY found by os.getenv: {masked_key}")

BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
# Define headers (aiohttp often takes headers per-session or per-request)
# Let's try setting Accept like curl, leave Content-Type out for GET
HEADERS = {
    "apiKey": NVD_API_KEY,
    "Accept": "*/*", # Mimic curl's default Accept header
    "User-Agent": "mcp-nvd-client/1.0 (aiohttp)" # Identify this version
}

MCP_SERVER_NAME = "mcp-nvd"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(MCP_SERVER_NAME)

# --- Pydantic Input Models --- (Keep these as they are)
class GetCveInput(BaseModel):
    cve_id: str = Field(description="The CVE ID to retrieve (e.g., CVE-2023-1234).")
    concise: Optional[bool] = Field(default=False, description="Whether to return a concise output.")

class SearchCveInput(BaseModel):
    keyword: str = Field(description="The keyword to search for in CVE descriptions.")
    exact_match: Optional[bool] = Field(default=False, description="Whether to perform an exact match on the keyword.")
    concise: Optional[bool] = Field(default=False, description="Whether to return concise output for each CVE.")
    results: Optional[int] = Field(default=10, description="The maximum number of results to return.")

# --- MODIFIED function using 'aiohttp' (asynchronous) ---
async def make_nvd_request(url: str) -> Dict[str, Any] | None:
    """Make a request to the NVD API using 'aiohttp' with proper error handling."""
    logger.info(f"Attempting request using 'aiohttp' library to URL: {url}")
    logger.info(f"Using headers: { {k: (v[:4]+'****' if k=='apiKey' else v) for k,v in HEADERS.items()} }") # Log headers (mask key)

    # Create a ClientTimeout object for the total timeout
    timeout = aiohttp.ClientTimeout(total=30.0)

    try:
        # Create a session for the request
        # Pass headers to the session constructor
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            # Make the GET request using the session
            async with session.get(url) as response:
                logger.info(f"Response status code: {response.status}")
                # Read response text for logging/debugging before raising status
                response_text = await response.text()
                logger.info(f"Response text (partial): {response_text[:500]}")

                # Check for HTTP errors (4xx or 5xx)
                response.raise_for_status()

                # Parse JSON - response.json() is also awaitable in aiohttp
                # Need to use the stored text if using response_text earlier,
                # or call response.json() directly which reads the body again.
                # Let's use response.json() for simplicity, assuming raise_for_status passed.
                # We might need to re-read the text in the except block if needed.
                # Alternatively, parse here and handle potential JSON errors.
                try:
                    json_data = await response.json(content_type=None) # Allow any content type for json parsing
                    return json_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response: {e}")
                    logger.error(f"Response body was: {response_text}") # Use stored text
                    return None
                except Exception as e: # Catch other potential errors during json() like content type issues
                    logger.error(f"Error during response.json(): {e}")
                    logger.error(f"Response body was: {response_text}") # Use stored text
                    return None

    except ClientResponseError as e:
        # Handle HTTP status errors (4xx/5xx) caught by raise_for_status
        logger.error(f"HTTP error: {e.status} - {e.message}")
        # The response_text might not be available directly on the exception 'e'
        # If needed, we would have had to store it from the response object earlier
        logger.error(f"Response body (if available): {response_text if 'response_text' in locals() else 'N/A'}")
        return None
    except ClientError as e:
        # Handles connection errors, timeouts (ClientTimeoutError inherits from this), etc.
        logger.error(f"Client connection/request error: {e}")
        return None
    except asyncio.TimeoutError:
        # Catch timeouts explicitly if ClientTimeout is used
        logger.error("Request timed out.")
        return None
    except Exception as e:
        # Catch-all for other unexpected errors
        logger.error(f"Unexpected error in make_nvd_request: {e}", exc_info=True)
        return None


# --- Formatting function remains the same ---
def format_cve(cve: Dict[str, Any], concise: bool = False) -> str:
    # ... (keep the existing format_cve function unchanged) ...
    # (Code omitted for brevity - use the one from previous examples)
    """Helper function to format a single CVE entry."""
    try:
        cve_id = cve["id"]
        source_identifier = cve["sourceIdentifier"]
        published = cve["published"]
        lastModified = cve["lastModified"]
        vuln_status = cve["vulnStatus"]
        description = next(
            (desc["value"] for desc in cve["descriptions"] if desc["lang"] == "en"),
            "No English description available",
        )

        # Extract CVSS v3.1 metrics
        cvss_v31_metric = next(
            (metric for metric in cve.get("metrics", {}).get("cvssMetricV31", []) if metric["type"] == "Primary"),
            None,
        )
        cvss_v31_data = cvss_v31_metric["cvssData"] if cvss_v31_metric else None
        cvss_v31_score = cvss_v31_data.get("baseScore", "N/A") if cvss_v31_data else "N/A"
        cvss_v31_severity = cvss_v31_data.get("baseSeverity", "N/A") if cvss_v31_data else "N/A"
        cvss_v31_vector = cvss_v31_data.get("vectorString", "N/A") if cvss_v31_data else "N/A"
        cvss_v31_exploitability = cvss_v31_metric.get("exploitabilityScore", "N/A") if cvss_v31_metric else "N/A"
        cvss_v31_impact = cvss_v31_metric.get("impactScore", "N/A") if cvss_v31_metric else "N/A"

        # Extract CVSS v2.0 metrics
        cvss_v2 = next(
            (metric["cvssData"] for metric in cve.get("metrics", {}).get("cvssMetricV2", []) if metric["type"] == "Primary"),
            None,
        )
        cvss_v2_score = cvss_v2.get("baseScore", "N/A") if cvss_v2 else "N/A"
        cvss_v2_severity = cvss_v2.get("baseSeverity", "N/A") if cvss_v2 else "N/A"
        cvss_v2_vector = cvss_v2.get("vectorString", "N/A") if cvss_v2 else "N/A"

        # Extract weaknesses (CWE IDs)
        weaknesses = [
            desc["value"] for weak in cve.get("weaknesses", []) for desc in weak["description"] if desc["lang"] == "en"
        ]
        weaknesses_str = ", ".join(weaknesses) if weaknesses else "None listed"

        # Extract references with tags
        references = [f"{ref['url']} ({', '.join(ref.get('tags', []))})" for ref in cve.get("references", [])]
        references_str = "\n  - " + "\n  - ".join(references) if references else "None listed"

        # Extract configurations (CPEs)
        cpe_matches = []
        configurations = cve.get("configurations", [])
        if configurations:
             if isinstance(configurations, list) and len(configurations) > 0:
                  if isinstance(configurations[0], dict) and 'nodes' in configurations[0]:
                       for node in configurations[0].get("nodes", []):
                            for match in node.get("cpeMatch", []):
                                if match.get("vulnerable", False):
                                    cpe_matches.append(match["criteria"])
        configurations_str = "\n  - " + "\n  - ".join(cpe_matches) if cpe_matches else "None listed"

        # Format output
        if concise:
            return (
                f"CVE ID: {cve_id}\n"
                f"Description: {description}\n"
                f"CVSS v3.1 Score: {cvss_v31_score} ({cvss_v31_severity})"
            )
        else:
            return (
                f"CVE ID: {cve_id}\n"
                f"Source Identifier: {source_identifier}\n"
                f"Published: {published}\n"
                f"Last Modified: {lastModified}\n"
                f"Vulnerability Status: {vuln_status}\n"
                f"Description: {description}\n"
                f"CVSS v3.1 Score: {cvss_v31_score} ({cvss_v31_severity})\n"
                f"CVSS v3.1 Vector: {cvss_v31_vector}\n"
                f"CVSS v3.1 Exploitability Score: {cvss_v31_exploitability}\n"
                f"CVSS v3.1 Impact Score: {cvss_v31_impact}\n"
                f"CVSS v2.0 Score: {cvss_v2_score} ({cvss_v2_severity})\n"
                f"CVSS v2.0 Vector: {cvss_v2_vector}\n"
                f"Weaknesses (CWE): {weaknesses_str}\n"
                f"References:\n{references_str}\n"
                f"Affected Configurations (CPE):\n{configurations_str}"
            )
    except Exception as e:
        logger.error(f"Error formatting CVE {cve.get('id', 'unknown')}: {str(e)}", exc_info=True)
        return f"Error processing CVE: {str(e)}"


# --- Tool functions remain async but call the new make_nvd_request ---
async def get_cve_tool(validated_args: GetCveInput) -> str:
    """Get a CVE based on the ID and return a formatted string."""
    url = f"{BASE_URL}?cveId={validated_args.cve_id}"
    data = await make_nvd_request(url) # Still await here

    if not data or "vulnerabilities" not in data or not data["vulnerabilities"]:
        return f"No data found or error occurred for CVE ID: {validated_args.cve_id}"

    if not data["vulnerabilities"]:
         return f"Data found but no vulnerability entry for CVE ID: {validated_args.cve_id}"

    cve = data["vulnerabilities"][0]["cve"]
    logger.info(f"Processing CVE: {validated_args.cve_id}")
    return format_cve(cve, validated_args.concise)

async def search_cve_tool(validated_args: SearchCveInput) -> str:
    """Search CVEs by keyword and return formatted results."""
    params = {
        "keywordSearch": validated_args.keyword,
        "resultsPerPage": validated_args.results
    }
    if validated_args.exact_match:
        params["keywordExactMatch"] = ""
    query_string = '&'.join(f'{k}={v}' for k, v in params.items() if v is not None or k == "keywordExactMatch")
    url = f"{BASE_URL}?{query_string}"
    logger.info(f"Constructed Search URL: {url}")
    data = await make_nvd_request(url) # Still await here
    # ... (rest of search_cve_tool remains the same) ...
    if not data or "vulnerabilities" not in data:
        return f"No CVEs found or error occurred for keyword: {validated_args.keyword} (exact_match: {validated_args.exact_match})"
    if not data["vulnerabilities"]:
         total_results = data.get("totalResults", 0)
         logger.info(f"Search for '{validated_args.keyword}' returned {total_results} total results but the vulnerabilities list is empty in this page.")
         return f"No CVEs found on this page for keyword: {validated_args.keyword} (exact_match: {validated_args.exact_match}, total results: {total_results})"
    logger.info(f"Searching CVEs with keyword: {validated_args.keyword}, exact_match: {validated_args.exact_match}, results: {validated_args.results}")
    results_list = []
    for cve_data in data["vulnerabilities"]:
        formatted_cve = format_cve(cve_data["cve"], validated_args.concise)
        results_list.append(formatted_cve)
    total_results = data.get("totalResults", 0)
    result_str = f"Found {len(results_list)} of {total_results} CVEs for keyword '{validated_args.keyword}' (exact_match: {validated_args.exact_match}, results requested: {validated_args.results}):\n\n"
    result_str += "\n\n---\n\n".join(results_list)
    logger.info(f"Completed search for keyword: {validated_args.keyword}, found {len(results_list)} results")
    return result_str

# --- Request handling still uses asyncio.run ---
def send_response(response_data: Dict[str, Any]):
    """Helper function to send JSON response to stdout."""
    # ... (keep send_response unchanged) ...
    response = json.dumps(response_data) + "\n"
    sys.stdout.write(response)
    sys.stdout.flush()

def handle_request(data: Dict[str, Any]):
    """Handles incoming MCP requests."""
    # ... (keep handle_request unchanged, it still needs asyncio.run) ...
    if not isinstance(data, dict):
        send_response({"error": "Invalid request format"})
        return

    method = data.get("method")
    if method == "tools/discover":
        send_response({
            "result": [
                {
                    "name": "get_cve",
                    "description": "Retrieves detailed information for a specific CVE ID from the NIST NVD.",
                    "parameters": GetCveInput.model_json_schema()
                },
                {
                    "name": "search_cve",
                    "description": "Searches the NIST NVD for CVEs based on a keyword.",
                    "parameters": SearchCveInput.model_json_schema()
                }
            ]
        })
    elif method == "tools/call":
        tool_name = data.get("params", {}).get("name")
        arguments = data.get("params", {}).get("arguments", {})
        if tool_name == "get_cve":
            try:
                validated_args = GetCveInput(**arguments)
                result = asyncio.run(get_cve_tool(validated_args)) # Still need asyncio.run
                send_response({"result": result})
            except ValidationError as e:
                logger.error(f"Validation Error for get_cve: {e}")
                send_response({"error": f"Invalid arguments for get_cve: {e}"})
            except Exception as e:
                logger.error(f"Error calling get_cve: {e}", exc_info=True)
                send_response({"error": f"Error executing get_cve: {e}"})
        elif tool_name == "search_cve":
            try:
                validated_args = SearchCveInput(**arguments)
                result = asyncio.run(search_cve_tool(validated_args)) # Still need asyncio.run
                send_response({"result": result})
            except ValidationError as e:
                logger.error(f"Validation Error for search_cve: {e}")
                send_response({"error": f"Invalid arguments for search_cve: {e}"})
            except Exception as e:
                logger.error(f"Error calling search_cve: {e}", exc_info=True)
                send_response({"error": f"Error executing search_cve: {e}"})
        else:
            send_response({"error": f"Tool not found: {tool_name}"})
    else:
        send_response({"error": f"Unknown method: {method}"})


# --- Main execution remains similar ---
def monitor_stdin():
    """Monitors stdin for incoming MCP requests."""
    # ... (keep monitor_stdin unchanged) ...
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                time.sleep(0.1)
                continue
            try:
                data = json.loads(line)
                handle_request(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on line: '{line[:100]}...': {e}")
                send_response({"error": f"JSON decode error: {e}"})
            except Exception as e:
                 logger.error(f"Exception in handle_request for line '{line[:100]}...': {e}", exc_info=True)
                 send_response({"error": f"Internal server error: {e}"})
        except EOFError:
             logger.info("Stdin closed, exiting monitor_stdin thread.")
             break
        except Exception as e:
            logger.error(f"Exception in monitor_stdin loop: {e}", exc_info=True)
            time.sleep(0.1)

if __name__ == "__main__":
    # ... (keep __main__ block unchanged) ...
    parser = argparse.ArgumentParser(description="FastMCP Server for NIST NVD API (using aiohttp)") # Updated description
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help="Run a single request/response cycle."
    )
    args = parser.parse_args()

    logger.info("Starting FastMCP Server for NIST NVD API with Pydantic Validation (using aiohttp)") # Updated log message

    if args.oneshot:
        try:
            line = sys.stdin.readline().strip()
            if line:
                 data = json.loads(line)
                 handle_request(data)
            else:
                 logger.warning("Oneshot mode received empty input.")
                 send_response({"error": "No input received in oneshot mode"})
        except json.JSONDecodeError as e:
            logger.error(f"Oneshot JSON decode error: {e}")
            send_response({"error": f"JSON decode error: {e}"})
        except Exception as e:
            logger.error(f"Oneshot error: {e}", exc_info=True)
            send_response({"error": str(e)})
    else:
        stdin_thread = threading.Thread(target=monitor_stdin, daemon=True)
        stdin_thread.start()
        try:
             while stdin_thread.is_alive():
                 time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down FastMCP Server (KeyboardInterrupt)")
        finally:
            logger.info("Main thread exiting.")