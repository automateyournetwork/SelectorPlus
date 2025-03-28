import subprocess
import logging
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def nslookup_tool(input_data):
    """
    Performs a reverse DNS lookup (nslookup) on an IP address and returns a formatted response.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: { "agent_response": "🔍 The reverse DNS lookup for 142.251.32.78 resolved to yyz12s07-in-f14.1e100.net." }
    """
    try:
        # ✅ Ensure input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🔍 [NSLOOKUP] Performing reverse DNS lookup for IP: {ip}")

        # ✅ Run nslookup command
        result = subprocess.run(f"nslookup {ip}", shell=True, capture_output=True, text=True, timeout=5)

        # ✅ Log full output for debugging
        nslookup_output = result.stdout.strip()
        logger.info(f"📜 [NSLOOKUP OUTPUT]\n{nslookup_output}")

        # ✅ Check for errors
        if result.returncode != 0 or "NXDOMAIN" in nslookup_output or "Non-existent" in nslookup_output:
            logger.warning(f"⚠️ [NSLOOKUP] No valid response found for IP: {ip}")
            return {"agent_response": f"⚠️ No valid reverse DNS record found for {ip}."}

        # ✅ Extract hostname from NSLOOKUP output
        hostname = "Unknown"
        for line in nslookup_output.split("\n"):
            if "name =" in line:
                hostname = line.split("=")[-1].strip()
                break

        # ✅ Construct formatted response
        response_text = f"🔍 The reverse DNS lookup for **{ip}** resolved to **{hostname}**."

        logger.info(f"✅ [NSLOOKUP] Response: {response_text}")

        return {"agent_response": response_text}

    except subprocess.TimeoutExpired:
        logger.error(f"⏳ [NSLOOKUP] Command timed out for IP: {ip}")
        return {"agent_response": f"⚠️ nslookup request timed out for {ip}."}

    except Exception as e:
        logger.error(f"❌ [NSLOOKUP] Unexpected error: {e}")
        return {"agent_response": f"⚠️ Unexpected error while performing nslookup for {ip}."}

# ✅ Register LangChain Tool
nslookup_tool_obj = Tool(
    name="nslookup_tool",
    description="Performs a reverse DNS lookup on an IP and returns a human-readable response.",
    func=nslookup_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "142.251.32.78"}
    result = nslookup_tool(test_ip)
    print("NSLOOKUP RESULT:", result)
