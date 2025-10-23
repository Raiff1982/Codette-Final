import asyncio
from components.search_engine import SearchEngine

async def test_search():
    search_engine = SearchEngine()
    
    print("Testing Search Engine (Placeholder Implementation)")
    print("\nTest Query 1: What is artificial intelligence?")
    results1 = await search_engine.get_knowledge("What is artificial intelligence?", max_results=2)
    print(results1)
    
    print("\nTest Query 2: Tell me about programming")
    results2 = await search_engine.get_knowledge("Tell me about programming", max_results=2)
    print(results2)
    
    print("\nTest Query 3: What are the latest technologies?")
    results3 = await search_engine.get_knowledge("What are the latest technologies?", max_results=2)
    print(results3)

if __name__ == "__main__":
    asyncio.run(test_search())