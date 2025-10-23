"""
Simple API for Codette using centralized interface
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime

from codette_interface import interface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with CORS
app = FastAPI(title="Codette API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(query: Query):
    """Process a question through Codette interface"""
    try:
        result = await interface.process_query(query.text)
        result["timestamp"] = str(datetime.now())
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Full system health check"""
    try:
        status = await interface.get_system_status()
        status["timestamp"] = str(datetime.now())
        return status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(datetime.now())
        }

@app.on_event("startup")
async def startup_event():
    """Initialize interface on startup"""
    await interface.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of interface"""
    await interface.shutdown()

# Server initialization
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        log_level="info"
    )