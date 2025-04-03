import os
import json
import time
import logging
import requests
import sys
import threading
from typing import Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SERVICENOW_URL = os.getenv("SERVICENOW_URL").rstrip('/')
SERVICENOW_USER = os.getenv("SERVICENOW_USERNAME")
SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("servicenow_mcp")

# Basic Authentication for ServiceNow
auth = (SERVICENOW_USER, SERVICENOW_PASSWORD)
headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

class ServiceNowController:
    def __init__(self, servicenow_url, username, password):
        self.servicenow = servicenow_url.rstrip('/')
        self.auth = (username, password)
        self.headers = headers
    
    def get_records(self, table, query_params=None):
        """Retrieve records from a specified ServiceNow table."""
        url = f"{self.servicenow}/api/now/table/{table}"
        logging.info(f"GET Request to URL: {url}")
        try:
            response = requests.get(url, auth=self.auth, headers=self.headers, params=query_params, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"GET request failed: {e}")
            return {"error": f"Request failed: {e}"}
    
    def create_record(self, table, payload):
        url = f"{self.servicenow}/api/now/table/{table}"
        clean_headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        logging.info(f"POST Request to URL: {url} with Payload:\n{json.dumps(payload, indent=2)}")
    
        try:
            response = requests.post(
                url,
                auth=self.auth,
                headers=clean_headers,
                json=payload,
                verify=False,
                allow_redirects=False  # 🔥 THIS IS CRUCIAL 🔥
            )
    
            logging.info(f"Response Status Code: {response.status_code}")
            logging.info(f"Response Headers: {response.headers}")
            logging.info(f"Response Content: {response.text}")
    
            # Check for redirection
            if response.status_code in (301, 302, 307, 308):
                return {"error": f"Redirected to {response.headers.get('Location')}, check credentials or endpoint"}
    
            response.raise_for_status()
            return response.json()
    
        except requests.exceptions.RequestException as e:
            logging.error(f"POST request failed: {e}")
            return {"error": f"Request failed: {e}"}
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode failed: {e}")
            return {"error": f"Failed to parse JSON response: {e}"}


    def update_record(self, table, record_sys_id, payload):
        """Update a record in a specified ServiceNow table."""
        url = f"{self.servicenow}/api/now/table/{table}/{record_sys_id}"
        logging.info(f"PATCH Request to URL: {url} with Payload: {json.dumps(payload, indent=2)}")
        try:
            response = requests.patch(url, auth=self.auth, headers=self.headers, json=payload, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"PATCH request failed: {e}")
            return {"error": f"Request failed: {e}"}

# Initialize ServiceNow API Controller
servicenow_client = ServiceNowController(SERVICENOW_URL, SERVICENOW_USER, SERVICENOW_PASSWORD)

def send_response(response_data):
    """Send the response back to stdout."""
    response = json.dumps(response_data) + "\n"
    sys.stdout.write(response)
    sys.stdout.flush()

def handle_tools_discover():
    """Handle tools discovery (tools/discover)."""
    send_response({
        "result": [
            {"name": "get_servicenow_problem_sys_id", "description": "Get the sys_id for a ServiceNow problem by its number", "parameters": {"type": "object", "properties": {"problem_number": {"type": "string"}}}},
            {"name": "get_servicenow_problem_state", "description": "Get the current state of a ServiceNow problem", "parameters": {"type": "object", "properties": {"sys_id": {"type": "string"}}}},
            {"name": "get_servicenow_problem_details", "description": "Get detailed information for a ServiceNow problem", "parameters": {"type": "object", "properties": {"problem_number": {"type": "string"}}}},
            {"name": "create_servicenow_problem", "description": "Create a new ServiceNow problem", "parameters": {"type": "object", "properties": {"problem_data": {"type": "object"}}}},
            {"name": "update_servicenow_problem", "description": "Update a ServiceNow problem", "parameters": {"type": "object", "properties": {"sys_id": {"type": "string"}, "update_data": {"type": "object"}}}},
        ]
    })

def handle_tools_call(data):
    """Handle tools call (tools/call)."""
    tool_name = data.get("params", {}).get("name")
    arguments = data.get("params", {}).get("arguments", {})

    if tool_name == "get_servicenow_problem_sys_id":
        problem_number = arguments.get("problem_number", "")
        result = servicenow_client.get_records("problem", {"sysparm_query": f"number={problem_number}"})
        if result.get("result"):
            send_response({"result": result["result"][0]["sys_id"]})
        else:
            send_response({"error": "ServiceNow Problem not found"})

    elif tool_name == "get_servicenow_problem_state":
        sys_id = arguments.get("sys_id", "")
        result = servicenow_client.get_records("problem", {"sysparm_query": f"sys_id={sys_id}", "sysparm_fields": "problem_state"})
        if result.get("result"):
            send_response({"result": result["result"][0]["problem_state"]})
        else:
            send_response({"error": "ServiceNow Problem not found"})

    elif tool_name == "get_servicenow_problem_details":
        problem_number = arguments.get("problem_number", "")
        result = servicenow_client.get_records("problem", {"sysparm_query": f"number={problem_number}"})
        if result.get("result"):
            send_response({"result": json.dumps(result["result"][0], indent=2)})
        else:
            send_response({"error": "ServiceNow Problem details not found"})

    elif tool_name == "create_servicenow_problem":
        problem_data = arguments.get("problem_data", {})
        if isinstance(problem_data, str):
            try:
                problem_data = json.loads(problem_data)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing problem_data: {e}")
                send_response({"error": "Invalid problem_data format"})
                return
        logging.info(f"problem_data type: {type(problem_data)}, problem_data: {problem_data}")
        result = servicenow_client.create_record("problem", problem_data)
        send_response({"result": result})

    elif tool_name == "update_servicenow_problem":
        sys_id = arguments.get("sys_id", "")
        update_data = arguments.get("update_data", {})
        result = servicenow_client.update_record("problem", sys_id, update_data)
        send_response({"result": result})

    else:
        send_response({"error": f"Tool '{tool_name}' not implemented in ServiceNow MCP"})

def monitor_stdin():
    """Monitor stdin for input and process `tools/discover` or `tools/call`."""
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                time.sleep(0.1)
                continue

            try:
                data = json.loads(line)
                if isinstance(data, dict) and data.get("method") == "tools/call":
                    handle_tools_call(data)
                elif isinstance(data, dict) and data.get("method") == "tools/discover":
                    handle_tools_discover()

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")

        except Exception as e:
            logger.error(f"Exception in monitor_stdin: {str(e)}")
            time.sleep(0.1)

if __name__ == "__main__":
    logger.info("Starting server")

    # If --oneshot flag is passed, process one request and exit
    if "--oneshot" in sys.argv:
        try:
            line = sys.stdin.readline().strip()
            data = json.loads(line)

            if isinstance(data, dict) and data.get("method") == "tools/call":
                handle_tools_call(data)

            elif isinstance(data, dict) and data.get("method") == "tools/discover":
                handle_tools_discover()

        except Exception as e:
            logger.error(f"Oneshot error: {e}")
            send_response({"error": str(e)})

    else:
        # Default: run as a server
        monitor_stdin()  # Monitor stdin in a blocking manner for multiple requests