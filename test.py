import httpx
import asyncio
import traceback # Import traceback

async def test_post():
    # url = "http://localhost:10000/tasks/send"
    url = "http://127.0.0.1:10000/tasks/send"
    # Use a different content/ID for a fresh request in logs
    payload = {"jsonrpc":"2.0","method":"send","params":{"content":"test from basic script v2"},"id":100}
    print(f"Attempting POST to {url}")
    try:
        # Disable proxy detection, increase timeout substantially
        # Use a client-level timeout instead of post-level
        async with httpx.AsyncClient(timeout=90.0) as client:
            print(f"Client timeout set to: {client.timeout}")
            response = await client.post(url, json=payload)
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
            try:
                # Let's see the JSON the adapter actually sent back
                print(f"Response JSON: {response.json()}")
            except Exception as json_err:
                print(f"Could not parse response as JSON: {json_err}")

    except httpx.TimeoutException as e: # Catch timeout specifically
         print(f"\n--- Timeout Error ---")
         print(f"Error Type: {type(e)}")
         print(f"Error Details: {e}")
         print(traceback.format_exc()) # Print traceback for timeout
    except httpx.RequestError as e: # Catch other httpx request errors (like ConnectError)
         print(f"\n--- Request Error ---")
         print(f"Error Type: {type(e)}")
         print(f"Error Details: {e}")
         print(traceback.format_exc())
    except Exception as e: # Catch any other exceptions
         print(f"\n--- Other Error ---")
         print(f"Error Type: {type(e)}")
         print(f"Error Details: {e}")
         print(traceback.format_exc()) # Print the full traceback

if __name__ == "__main__":
    asyncio.run(test_post())