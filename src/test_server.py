"""
Simplified test server with basic response generation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def generate_response(query: str) -> str:
    """Generate a simple response"""
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"[{current_time}] I'm a simplified version of Codette. You asked: {query}"

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(query: Query):
    """Process a question using simple response generation"""
    try:
        response = generate_response(query.text)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)