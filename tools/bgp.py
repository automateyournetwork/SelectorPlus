import logging
import requests
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
def bgp_lookup_tool(input_data):
    """
    Queries BGPView API for ASN and routing information.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: { "agent_response": "🌍 The IP 142.251.32.78 is part of ASN AS15169 (Google LLC) and belongs to the prefix 142.250.0.0/15 in the US." }
    """
    try:
        # ✅ Validate input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🌍 [BGP LOOKUP] Querying ASN info for IP: {ip}")

        # ✅ API Request
        url = f"https://api.bgpview.io/ip/{ip}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # ✅ Extract ASN and Prefix from prefixes list
        prefixes = data.get("data", {}).get("prefixes", [])
        
        if prefixes:
            asn_data = prefixes[0].get("asn", {})
            asn = asn_data.get("asn", "Unknown")
            prefix = prefixes[0].get("prefix", "Unknown")
            name = asn_data.get("name", "Unknown")
            country = asn_data.get("country_code", "Unknown")
        else:
            asn, prefix, name, country = "Unknown", "Unknown", "Unknown", "Unknown"

        # ✅ Construct formatted response
        response_text = (
            f"🌍 The IP **{ip}** is part of ASN **{asn} ({name})** and belongs to the prefix **{prefix}** in **{country}**."
        )

        logger.info(f"✅ [BGP LOOKUP] Response: {response_text}")

        return {"agent_response": response_text}

    except requests.exceptions.Timeout:
        logger.error(f"⏳ [BGP LOOKUP] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ BGP lookup request timed out for {ip}."}

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [BGP LOOKUP] API request failed: {e}")
        return {"agent_response": f"⚠️ Error retrieving BGP data for {ip}."}

# ✅ Register as a LangChain Tool
bgp_lookup_tool_obj = Tool(
    name="bgp_lookup_tool",
    description="Queries BGPView API for ASN and routing information of an IP address.",
    func=bgp_lookup_tool
)
