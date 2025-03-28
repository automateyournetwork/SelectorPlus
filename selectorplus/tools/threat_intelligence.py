import requests
import logging
import os
from langsmith import traceable
from langchain.tools import Tool

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load API Key
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY")

@traceable
def threat_check_tool(input_data):
    """
    Checks if an IP address is blacklisted or has a poor reputation score using AbuseIPDB.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: { "agent_response": "🚨 The IP 142.251.32.78 has an abuse score of 85/100 and is blacklisted. It has been reported 30 times, last seen on 2024-04-28." }
    """
    try:
        # ✅ Validate input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🔍 [THREAT CHECK] Checking threat intelligence for IP: {ip}")

        # ✅ Query AbuseIPDB API
        url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}"
        headers = {
            "Key": ABUSEIPDB_API_KEY,
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=5)

        # ✅ Log the full raw API response
        logger.info(f"📜 [Threat Intelligence API Response] HTTP {response.status_code}: {response.text}")

        response.raise_for_status()
        data = response.json()

        # ✅ Extract Threat Intelligence Details
        abuse_data = data.get("data", {})
        if not abuse_data:
            logger.warning(f"⚠️ No threat intelligence data found for IP: {ip}")
            return {"agent_response": f"⚠️ No threat intelligence data found for {ip}."}

        abuse_score = abuse_data.get("abuseConfidenceScore", 0)
        is_blacklisted = "🚨 **Blacklisted**" if abuse_data.get("isWhitelisted", False) is False else "✅ **Not Blacklisted**"
        confidence_level = abuse_data.get("abuseConfidenceScore", 0)
        total_reports = abuse_data.get("totalReports", 0)
        last_reported = abuse_data.get("lastReportedAt", "Unknown")

        # ✅ Construct formatted response
        response_text = (
            f"🚨 **Threat Intelligence for {ip}:**\n"
            f"- **Abuse Score:** {abuse_score}/100\n"
            f"- **Status:** {is_blacklisted}\n"
            f"- **Confidence Level:** {confidence_level}%\n"
            f"- **Total Reports:** {total_reports}\n"
            f"- **Last Reported:** {last_reported}\n"
        )

        logger.info(f"✅ [THREAT CHECK] Response: {response_text}")

        return {"agent_response": response_text}

    except requests.exceptions.Timeout:
        logger.error(f"⏳ [THREAT CHECK] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ Threat intelligence request timed out for {ip}."}

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [THREAT CHECK] API request failed: {e}")
        return {"agent_response": f"⚠️ Error retrieving threat intelligence for {ip}."}

# ✅ Register LangChain Tool
threat_check_tool_obj = Tool(
    name="threat_check_tool",
    description="Checks if an IP is blacklisted or has a poor reputation score using AbuseIPDB and returns a formatted response.",
    func=threat_check_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "142.251.32.78"}
    result = threat_check_tool(test_ip)
    print("THREAT CHECK RESULT:", result)
