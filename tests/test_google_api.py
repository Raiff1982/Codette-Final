import os
import json
import asyncio
from dotenv import load_dotenv
import aiohttp

async def test_google_api():
    load_dotenv()
    
    # Get credentials
    api_key = os.getenv("GOOGLE_API_KEY", "")
    search_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ID", "")
    
    print("\n=== Google API Test ===")
    print(f"API Key (first 8 chars): {api_key[:8]}...")
    print(f"Search Engine ID format check: {'Looks correct' if ':' in search_id else 'Incorrect format - should contain a colon (:)'}")
    
    # Test API key validity
    print("\nTesting API key...")
    validation_url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx=test&q=test"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(validation_url) as response:
                data = await response.text()
                print(f"\nAPI Response Status: {response.status}")
                print("Response Details:")
                try:
                    json_data = json.loads(data)
                    if 'error' in json_data:
                        print(f"Error Type: {json_data['error'].get('status', 'Unknown')}")
                        print(f"Error Message: {json_data['error'].get('message', 'No message')}")
                        if 'details' in json_data['error']:
                            for detail in json_data['error']['details']:
                                if '@type' in detail and 'reason' in detail:
                                    print(f"Error Detail: {detail['@type']} - {detail.get('reason', 'No reason')}")
                    else:
                        print(json.dumps(json_data, indent=2))
                except json.JSONDecodeError:
                    print("Could not parse response as JSON")
                    print(f"Raw response: {data[:200]}...")
    
    except Exception as e:
        print(f"Connection error: {str(e)}")
    
    print("\nTroubleshooting Tips:")
    print("1. API Key should start with 'AIza'")
    print("2. Search Engine ID should contain a colon (:)")
    print("3. Custom Search API must be enabled in Google Cloud Console")
    print("4. Project must have billing enabled (even for free tier)")
    print("5. API key should not have any restrictions that would block this test")

if __name__ == "__main__":
    asyncio.run(test_google_api())