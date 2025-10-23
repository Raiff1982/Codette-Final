import asyncio
import os
from dotenv import load_dotenv
from utils.search_utility import SearchUtility

async def test_search():
    # Load environment variables
    load_dotenv()
    
    # Print debug info (without exposing full keys)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    search_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ID", "")
    
    print("Debug Information:")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    print(f"API Key length: {len(api_key)} characters")
    print(f"Search Engine ID present: {'Yes' if search_id else 'No'}")
    print(f"Search Engine ID length: {len(search_id)} characters")
    
    print("\nTesting Google Search functionality...")
    try:
        search_util = SearchUtility()
        results = await search_util.search("What is artificial intelligence?", max_results=2)
        print("\nSearch Results:")
        print(results)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please verify your API credentials in the .env file")

if __name__ == "__main__":
    asyncio.run(test_search())