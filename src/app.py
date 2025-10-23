"""
Codette Application Server
Implements a FastAPI server with advanced Codette functionality
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from codette import Codette
from components.config_manager import EnhancedAIConfig
from components.health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Codette API",
    description="Advanced AI Processing System API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    perspectives: Optional[List[str]] = None

class StateRequest(BaseModel):
    user_id: str
    quantum_data: Optional[Dict[str, Any]] = None

class ProcessingResponse(BaseModel):
    status: str
    result: Dict[str, Any]
    timestamp: str
    quantum_state: Dict[str, Any]

# Global Codette instance
codette_instance: Optional[Codette] = None

def get_codette() -> Codette:
    """Get or create global Codette instance"""
    global codette_instance
    if codette_instance is None:
        try:
            config = EnhancedAIConfig("config.json")
            codette_instance = Codette(
                user_name="APIUser",
                perspectives=["Newton", "DaVinci", "Ethical", "Quantum", "Memory"],
                spiderweb_dim=5,
                memory_path="quantum_cocoon.json",
                recursion_depth=4,
                quantum_fluctuation=0.07
            )
            logger.info("Codette instance initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Codette: {e}")
            raise
    return codette_instance

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize Codette
        get_codette()
        
        # Initialize health monitor
        health_monitor = HealthMonitor()
        await health_monitor.initialize()
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if codette_instance:
            await codette_instance.shutdown()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

@app.get("/health")
async def health_check():
    """Check system health"""
    try:
        codette = get_codette()
        health_status = await codette.health_monitor.check_status()
        return {
            "status": "healthy" if health_status.get("healthy", False) else "unhealthy",
            "details": health_status,
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=ProcessingResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process a query through Codette's advanced systems
    
    Args:
        request: QueryRequest containing query and optional parameters
        background_tasks: FastAPI background tasks handler
        
    Returns:
        ProcessingResponse with results and quantum state
    """
    try:
        codette = get_codette()
        
        # Process through multiple perspectives if specified
        perspectives = request.perspectives or codette.perspectives
        
        # Process query
        response = await codette.process_quantum_state({
            "query": request.query,
            "context": request.context or {},
            "perspectives": perspectives,
            "user_id": request.user_id or "anonymous",
            "timestamp": str(datetime.now())
        })
        
        # Get sentiment analysis
        sentiment = codette.sentiment_analyzer.analyze(request.query)
        
        # Get ethical evaluation
        ethical_status = codette.ethical_gov.enforce_policies(response)
        
        # Enhance with creativity insights
        creative_insights = codette.creativity_engine.process({
            "query": request.query,
            "response": response
        })
        
        # Schedule background learning
        background_tasks.add_task(
            codette.learning_env.process_interaction,
            request.query,
            response
        )
        
        # Compile final response
        result = {
            "response": response,
            "sentiment": sentiment,
            "ethical_status": ethical_status,
            "creative_insights": creative_insights,
            "processed_perspectives": perspectives
        }
        
        return ProcessingResponse(
            status="success",
            result=result,
            timestamp=str(datetime.now()),
            quantum_state=codette.quantum_state
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_state")
async def update_quantum_state(request: StateRequest):
    """Update quantum state for a user"""
    try:
        codette = get_codette()
        
        # Process quantum state update
        updated_state = await codette.process_quantum_state(request.quantum_data or {})
        
        return {
            "status": "success",
            "updated_state": updated_state,
            "timestamp": str(datetime.now())
        }
        
    except Exception as e:
        logger.error(f"State update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system_state")
async def get_system_state():
    """Get complete system state"""
    try:
        codette = get_codette()
        return codette.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    try:
        config = EnhancedAIConfig("config.json")
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8000)
        
        # Run server
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise