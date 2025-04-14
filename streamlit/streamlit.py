import os
import json
import base64
import requests
import traceback
import streamlit as st

# Configure Streamlit page
st.set_page_config(page_title="Selector Plus Copilot", page_icon="🔍")

# API URL
API_BASE_URL = "http://host.docker.internal:2024"

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None  # Start with None instead of default_thread
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Keep the same branding and header
st.image("logo.jpeg")
st.markdown("---")
st.write("Welcome to Selector Plus, your MCP-powered assistant!")
st.markdown("---")
st.write(
    """
Selector Plus connects to a variety of MCP (Model Context Protocol) servers to provide you with powerful tools. To use these tools effectively, provide clear and specific instructions in your prompts. Here are some examples:

* **To talk to Selector:** Selector device health for device S3
* **To post a message to a Slack channel:** "Send 'Hello team!' to Slack channel <channelID>"
* **To search the web:** "Search for the latest news on AI"
* **To create a drawing:** "Draw a flowchart of a website"
* **To read a file:** "Read the contents of 'my_file.txt'"
* **To get GitHub information:** "Get the latest commits from the 'my_repo' repository"
* **To get google map information:** "Find the nearest coffee shop"

The more specific your prompt, the better the results.
"""
)
st.markdown("---")
st.write(
    """
### Invoking Tools Using Prompts:

Here are the MCP Servers attached to Selector Plus:

* **Selector:** Interface into Selector S2App for network device management and monitoring.
* **Brave Search:** Local and Global Web Search for information retrieval.
* **Excalidraw:** Create and export drawings to JSON format.
* **Filesystem:** Create folders, files, and read file contents on the server.
* **GitHub:** Create files on GitHub repositories.
* **Google Maps:** Geolocation and elevation data, find places, and directions.
* **Sequential Thinking:** Tools for sequential data analysis and processing.
* **Slack:** Post messages to Slack channels within your workspace.
* **BGP:** Retrieve BGP ASN information for a public IP address.
* **Curl:** Perform HTTP requests to public IP addresses or websites.
* **Dig:** Perform DNS lookups (dig) for a public IP address.
* **NSLookup:** Perform name server lookups (nslookup) for a public IP address.
* **PING:** Ping a public IP address to check connectivity.
* **Geolocation:** Obtain geographical location information for a public IP address.
* **Threat Intelligence:** Retrieve threat intelligence reports for a public IP address.
* **Traceroute:** Trace the route packets take to reach a public IP address.
* **WHOIS:** Obtain WHOIS information for a public IP address.
"""
)
st.markdown("---")
# Keep layout the same as the upload page
st.header("Chat with Selector Plus")

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(":green[Your Question:]")
        st.markdown(msg["content"])
    else:
        st.markdown(":red[Selector Plus:]")
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Enter your message"):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        try:
            # If no thread exists, create a new thread
            if not st.session_state["thread_id"]:
                # Create thread request
                thread_response = requests.post(
                    f"{API_BASE_URL}/threads",
                    json={"assistant_id": "SelectorPlus"}
                )

                # Ensure thread creation was successful
                if thread_response.status_code == 200:
                    st.session_state["thread_id"] = thread_response.json().get(
                        "thread_id", "default_thread"
                    )
                else:
                    raise Exception("Failed to create thread")

            # Prepare API request
            api_payload = {
                "assistant_id": "SelectorPlus",
                "input": {
                    "messages": [
                        {"type": "human", "content": prompt}
                    ]
                }
            }

            # Send request to API
            response = requests.post(
                f"{API_BASE_URL}/threads/{st.session_state['thread_id']}/runs/stream",
                json=api_payload,
                timeout=30,
            )

            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                ai_messages = []

                for line in lines:
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])
                            if "messages" in json_data:
                                for msg in json_data["messages"]:
                                    if msg.get("type") == "ai":
                                        if "content" in msg:
                                            ai_messages.append(msg["content"])

                                    # ✅ Catch tool responses that have "tool_call_id" (ToolMessage)
                                    elif "tool_call_id" in msg and "content" in msg:
                                        try:
                                            # 🛠️ FIXED: Support structured object directly
                                            content_raw = msg["content"]
                                            if isinstance(content_raw, str):
                                                tool_response = json.loads(content_raw)
                                            else:
                                                tool_response = content_raw

                                            if isinstance(tool_response, dict):
                                                if "result" in tool_response:
                                                    result = tool_response["result"]
                                                    if isinstance(result, dict) and "content" in result:
                                                        ai_messages.append(result["content"])
                                                    else:
                                                        ai_messages.append(str(result))
                                                elif "content" in tool_response:
                                                    ai_messages.append(tool_response["content"])
                                            else:
                                                ai_messages.append(str(tool_response))

                                        except json.JSONDecodeError:
                                            ai_messages.append(str(msg["content"]))  # Fallback if it's just raw text

                                        # 🛠️ Handle tool calls like Slack
                                        if "tool_calls" in msg:
                                            for tool_call in msg["tool_calls"]:
                                                if tool_call["name"] == "slack_post_message":
                                                    tool_output = next(
                                                        (
                                                            m["content"]
                                                            for m in json_data["messages"]
                                                            if m.get("tool_call_id") == tool_call["id"]
                                                        ),
                                                        None,
                                                    )
                                                    if tool_output:
                                                        try:
                                                            if isinstance(tool_output, str):
                                                                tool_output_json = json.loads(tool_output)
                                                            else:
                                                                tool_output_json = tool_output

                                                            if tool_output_json.get("ok"):
                                                                ai_messages.append("✅ Message sent successfully to Slack.")
                                                            else:
                                                                ai_messages.append("⚠️ Message sending to Slack failed.")
                                                        except json.JSONDecodeError:
                                                            ai_messages.append("❌ Error processing Slack response.")

                        except json.JSONDecodeError:
                            st.error(f"JSON Decode Error in line: {line}")
                            
                if ai_messages:
                    # Use the last AI message
                    ai_response = ai_messages[-1]
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": ai_response}
                    )
                    st.chat_message("assistant").write(ai_response)
                else:
                    st.error("No response from the AI")
            else:
                st.error(f"API Error: {response.status_code}")
                st.error(response.text)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    st.markdown("---")