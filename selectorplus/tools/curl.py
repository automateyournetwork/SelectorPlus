import subprocess
import logging
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def curl_tool(input_data):
    """
    Performs an HTTP request to check if an IP is serving a website.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: {
        "agent_response": "🌍 The IP 8.8.8.8 responded with HTTP status 200. The server is nginx and the content length is 51234 bytes."
    }
    """
    try:
        # ✅ Ensure input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        url = f"http://{ip}"  # Default to HTTP
        logger.info(f"🌍 [cURL] Checking web response for IP: {ip}")

        # ✅ Run cURL Command
        cmd = f"curl -s -I -L --max-time 5 {url}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

        # ✅ Log cURL output for debugging
        curl_output = result.stdout.strip()
        logger.info(f"📜 [cURL OUTPUT]\n{curl_output}")

        # ✅ Check for errors
        if result.returncode != 0 or "could not resolve" in curl_output.lower():
            logger.warning(f"⚠️ [cURL] No web response from IP: {ip}")
            return {"agent_response": f"⚠️ No web response from IP: {ip}."}

        # ✅ Parse Headers
        headers = {}
        for line in curl_output.split("\n"):
            parts = line.split(": ", 1)
            if len(parts) == 2:
                headers[parts[0].strip().lower()] = parts[1].strip()

        # ✅ Extract Relevant Information
        status_code = headers.get("http", "Unknown").split(" ")[1] if "http" in headers else "Unknown"
        server = headers.get("server", "Unknown")
        content_length = headers.get("content-length", "Unknown")
        redirected_url = headers.get("location", "None")

        # ✅ Construct formatted response
        response_text = (
            f"🌍 The IP **{ip}** responded with **HTTP status {status_code}**. "
            f"The server is **{server}**, and the content length is **{content_length} bytes**."
            f" {'It redirects to ' + redirected_url if redirected_url != 'None' else 'No redirection detected.'}"
        )

        logger.info(f"✅ [cURL] Response: {response_text}")

        return {"agent_response": response_text}

    except subprocess.TimeoutExpired:
        logger.error(f"⏳ [cURL] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ cURL request timed out for {ip}."}

    except Exception as e:
        logger.error(f"❌ [cURL] Unexpected error: {e}")
        return {"agent_response": f"⚠️ Unexpected error while performing cURL request for {ip}."}

# ✅ Register LangChain Tool
curl_lookup_tool_obj = Tool(
    name="curl_lookup_tool",
    description="Performs a cURL request to check if an IP is serving a website and returns a human-readable HTTP response.",
    func=curl_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "8.8.8.8"}
    result = curl_tool(test_ip)
    print("cURL LOOKUP RESULT:", result)
