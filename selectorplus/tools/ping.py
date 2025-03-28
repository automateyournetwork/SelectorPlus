import subprocess
import logging
import re
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def ping_tool(input_data):
    """
    Performs a ping request to check if an IP is reachable and returns a formatted response.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: {
        "agent_response": "📡 The IP 8.8.8.8 responded with an average latency of 14.4 ms. All 3 packets were received with 0% packet loss."
    }
    """
    try:
        # ✅ Validate input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🔍 [PING] Checking reachability for IP: {ip}")

        # ✅ Run Ping Command
        cmd = f"ping -c 3 {ip}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

        # ✅ Log full output for debugging
        ping_output = result.stdout.strip()
        logger.info(f"📜 [PING OUTPUT]\n{ping_output}")

        # ✅ Check if ping was successful
        if result.returncode != 0:
            logger.warning(f"⚠️ [PING] IP {ip} is unreachable.")
            return {"agent_response": f"⚠️ The IP {ip} is unreachable. Ping request failed."}

        # ✅ Extract Statistics
        packets_sent, packets_received, packet_loss = "Unknown", "Unknown", "Unknown"
        rtt_min, rtt_avg, rtt_max, rtt_mdev = "Unknown", "Unknown", "Unknown", "Unknown"

        # ✅ Extract packet loss stats
        match = re.search(r"(\d+) packets transmitted, (\d+) received, (\d+)% packet loss", ping_output)
        if match:
            packets_sent, packets_received, packet_loss = match.groups()
            packet_loss = f"{packet_loss}%"

        # ✅ Extract RTT (Round Trip Time) stats
        match = re.search(r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", ping_output)
        if match:
            rtt_min, rtt_avg, rtt_max, rtt_mdev = match.groups()

        # ✅ Construct formatted response
        response_text = (
            f"📡 The IP **{ip}** responded with an average latency of **{rtt_avg} ms**.\n"
            f"All **{packets_sent} packets** were sent, and **{packets_received}** were received, with a packet loss of **{packet_loss}**."
        )

        logger.info(f"✅ [PING] Response: {response_text}")

        return {"agent_response": response_text}

    except subprocess.TimeoutExpired:
        logger.error(f"⏳ [PING] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ Ping request timed out for {ip}."}

    except Exception as e:
        logger.error(f"❌ [PING] Unexpected error: {e}")
        return {"agent_response": f"⚠️ Unexpected error while performing ping request for {ip}."}

# ✅ Register LangChain Tool
ping_tool_obj = Tool(
    name="ping_tool",
    description="Performs a ping request to check if an IP is reachable and measures latency.",
    func=ping_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "8.8.8.8"}
    result = ping_tool(test_ip)
    print("PING LOOKUP RESULT:", result)
