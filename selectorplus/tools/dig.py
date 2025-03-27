import subprocess
import logging
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def dig_tool(input_data):
    """
    Performs a DNS lookup using 'dig' to fetch DNS records for an IP.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: {
        "agent_response": "🌍 The DNS lookup for 142.251.32.78 returned the following:\n - **Host:** yyz12s07-in-f14.1e100.net.\n - **Query Time:** 20 msec\n - **Server Used:** 127.0.0.11"
    }
    """
    try:
        # ✅ Ensure correct input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🌍 [DIG] Performing DNS lookup for IP: {ip}")

        # ✅ Run Dig Command
        cmd = f"dig +short -x {ip}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

        # ✅ Log full output for debugging
        dig_output = result.stdout.strip()
        logger.info(f"📜 [DIG OUTPUT]\n{dig_output}")

        # ✅ Check if DIG failed
        if result.returncode != 0 or not dig_output:
            logger.warning(f"⚠️ [DIG] No DNS records found for {ip}")
            return {"agent_response": f"⚠️ No DNS records found for {ip}."}

        # ✅ Extract hostname from DIG response
        lines = dig_output.split("\n")
        hostname = lines[0] if lines else "Unknown"

        # ✅ Construct formatted response
        response_text = (
            f"🌍 The DNS lookup for **{ip}** returned:\n"
            f"- **Host:** {hostname}"
        )

        logger.info(f"✅ [DIG] Response: {response_text}")

        return {"agent_response": response_text}

    except subprocess.TimeoutExpired:
        logger.error(f"⏳ [DIG] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ DIG request timed out for {ip}."}

    except Exception as e:
        logger.error(f"❌ [DIG] Unexpected error: {e}")
        return {"agent_response": f"⚠️ Unexpected error while performing DIG request for {ip}."}

# ✅ Register LangChain Tool
dig_tool_obj = Tool(
    name="dig_tool",
    description="Performs a DNS reverse lookup (PTR record) for a given IP address.",
    func=dig_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "142.251.32.78"}
    result = dig_tool(test_ip)
    print("DIG LOOKUP RESULT:", result)
