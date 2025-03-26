import subprocess
import logging
from langsmith import traceable
from langchain.tools import Tool
import re

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def traceroute_tool(input_data):
    """
    Performs a traceroute using ICMP (-I) to analyze the network path to an IP.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: {
        "agent_response": "📡 Traceroute results for 8.8.8.8:\n- Hop 1: 192.168.1.1\n- Hop 2: 10.0.0.1\n- Hop 3: 8.8.8.8 (google.com)"
    }
    """
    try:
        # ✅ Validate input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🌍 [TRACEROUTE] Tracing route to IP: {ip}")

        # ✅ Run traceroute (forcing ICMP -I)
        cmd = f"traceroute -I {ip}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)

        # ✅ Log full output for debugging
        traceroute_output = result.stdout.strip()
        logger.info(f"📜 [TRACEROUTE OUTPUT]\n{traceroute_output}")

        # ✅ Check if traceroute was successful
        if result.returncode != 0 or "no reply" in traceroute_output.lower():
            logger.warning(f"⚠️ [TRACEROUTE] No response from {ip}.")
            return {"agent_response": f"⚠️ No response from {ip} during traceroute."}

        # ✅ Extract hops using regex
        hops = []
        hop_regex = re.compile(r"^\s*(\d+)\s+([\w\.\-]+)\s+\(([\d\.]+)\)\s+.*$")
        for line in traceroute_output.split("\n")[1:]:  # Skip first line (header)
            match = hop_regex.match(line)
            if match:
                hop_num = match.group(1)
                hop_host = match.group(2)
                hop_ip = match.group(3)
                hops.append(f"- **Hop {hop_num}:** {hop_ip} ({hop_host})")

        if not hops:
            logger.warning(f"⚠️ [TRACEROUTE] No valid hops extracted for IP {ip}.")
            return {"agent_response": f"⚠️ No valid hops extracted for {ip}."}

        response_text = f"📡 **Traceroute results for {ip}:**\n" + "\n".join(hops)

        logger.info(f"✅ [TRACEROUTE] Response: {response_text}")

        return {"agent_response": response_text}

    except subprocess.TimeoutExpired:
        logger.error(f"⏳ [TRACEROUTE] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ Traceroute request timed out for {ip}."}

    except Exception as e:
        logger.error(f"❌ [TRACEROUTE] Unexpected error: {e}")
        return {"agent_response": f"⚠️ Unexpected error while performing traceroute for {ip}."}

# ✅ Register LangChain Tool
traceroute_tool_obj = Tool(
    name="traceroute_tool",
    description="Performs an ICMP traceroute to analyze the network path to an IP and returns a structured response.",
    func=traceroute_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "8.8.8.8"}
    result = traceroute_tool(test_ip)
    print("TRACEROUTE RESULT:", result)
