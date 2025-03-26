import os
import logging
import requests
from dotenv import load_dotenv
from langsmith import traceable
from langchain.tools import Tool

# ✅ Load Environment Variables
load_dotenv()

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ WeatherAPI Base URL
BASE_API_URL = "https://api.weatherapi.com/v1"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

@traceable
def get_location_tool(input_data):
    """
    Fetches geographic information for a public IP address and returns a formatted response.

    Parameters:
    - input_data (dict): Must contain {"ip": "x.x.x.x"}

    Returns:
    - dict: { "agent_response": "📍 The IP 142.251.32.78 is located in Toronto, Ontario, Canada (Lat: 43.6532, Lon: -79.3832)." }
    """
    try:
        # ✅ Ensure correct input format
        if not isinstance(input_data, dict) or "ip" not in input_data:
            logger.warning("⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}.")
            return {"agent_response": "⚠️ Invalid input format. Expected {'ip': 'x.x.x.x'}."}

        ip = input_data["ip"]
        logger.info(f"🌍 [WeatherAPI] Fetching geolocation for IP: {ip}")

        # ✅ Make API request
        url = f"{BASE_API_URL}/ip.json"
        params = {"key": WEATHER_API_KEY, "q": ip}

        response = requests.get(url, params=params, timeout=5)
        logger.info(f"📜 [WeatherAPI Response] HTTP {response.status_code}: {response.text}")

        response.raise_for_status()
        data = response.json()

        # ✅ Extract geolocation info
        city = data.get("city", "Unknown")
        region = data.get("region", "Unknown")
        country = data.get("country_name", "Unknown")
        lat = data.get("lat", "N/A")
        lon = data.get("lon", "N/A")

        # ✅ Construct formatted response
        response_text = (
            f"📍 The IP **{ip}** is located in **{city}, {region}, {country}** "
            f"(Lat: **{lat}**, Lon: **{lon}**)."
        )

        logger.info(f"✅ [WeatherAPI] Response: {response_text}")

        return {"agent_response": response_text}

    except requests.exceptions.Timeout:
        logger.error(f"❌ [WeatherAPI] Request timed out for IP: {ip}")
        return {"agent_response": f"⚠️ Geolocation request timed out for {ip}."}

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ [WeatherAPI] HTTP error for IP {ip}: {e}")
        return {"agent_response": f"⚠️ Geolocation service returned an error for {ip}."}

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [WeatherAPI] API request failed: {e}")
        return {"agent_response": f"⚠️ Unable to retrieve location for {ip} at this time."}

# ✅ Register LangChain Tool
get_location_tool_obj = Tool(
    name="get_location_tool",
    description="Fetches geographic location details for a public IP and returns a formatted response.",
    func=get_location_tool
)

# ✅ Test the tool
if __name__ == "__main__":
    test_ip = {"ip": "142.251.32.78"}
    result = get_location_tool(test_ip)
    print("WEATHER LOOKUP RESULT:", result)
