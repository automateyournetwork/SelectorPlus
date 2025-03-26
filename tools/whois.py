import subprocess
import logging
import re
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def whois_tool(input_data):
    """
    Performs a WHOIS lookup on an IP address.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: {
        "agent_response": "🌍 WHOIS Lookup for 8.8.8.8:\n- Organization: Google LLC\n- Network Range: 8.8.8.0/24\n- Country: US\n- ASN: AS15169"
    }
    """
    try:
        # ✅ Ensure input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🔍 [WHOIS] Performing WHOIS lookup for IP: {ip}")

        # ✅ Run whois command
        result = subprocess.run(f"whois {ip}", shell=True, capture_output=True, text=True, timeout=5)

        # ✅ Log full command output for debugging
        whois_output = result.stdout.strip()
        logger.info(f"📜 [WHOIS OUTPUT]\n{whois_output}")

        # ✅ Check for errors
        if result.returncode != 0 or "No match" in whois_output or "Not found" in whois_output:
            logger.warning(f"⚠️ [WHOIS] No WHOIS data found for IP: {ip}")
            return {"agent_response": f"⚠️ No WHOIS data found for {ip}."}

        # ✅ Extract relevant WHOIS fields
        org_name = re.search(r"OrgName:\s*(.*)", whois_output)
        net_range = re.search(r"NetRange:\s*(.*)", whois_output)
        country = re.search(r"Country:\s*(.*)", whois_output)
        asn = re.search(r"OriginAS:\s*(.*)", whois_output)  # Autonomous System Number

        response_text = f"""🌍 **WHOIS Lookup for {ip}:**\n
        - **Organization:** {org_name.group(1) if org_name else 'Unknown'}
        - **Network Range:** {net_range.group(1) if net_range else 'Unknown'}
        - **Country:** {country.group(1) if country else 'Unknown'}
        - **ASN:** {asn.group(1) if asn else 'Unknown'}
        """

        logger.info(f"✅ [WHOIS] Processed response: {response_text.strip()}")

        return {"agent_response": response_text.strip()}

    except subprocess.TimeoutExpired:
        logger.error(f"⏳ [WHOIS] Command timed out for IP: {ip}")
        return {"agent_response": f"⚠️ WHOIS request timed out for {ip}."}

    except Exception as e:
        logger.error(f"❌ [WHOIS] Unexpected error: {e}")
        return {"agent_response": f"⚠️ Unexpected error while performing WHOIS lookup for {ip}."}

# ✅ Register LangChain Tool
whois_tool_obj = Tool(
    name="whois_tool",
    description="Performs a WHOIS lookup on an IP and extracts organization, network range, country, and ASN details in a readable format.",
    func=whois_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "8.8.8.8"}
    result = whois_tool(test_ip)
    print("WHOIS RESULT:", result)
