"""
Simple Codette server for testing
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from codette import Codette

app = FastAPI()
codette = Codette(user_name="TestUser")

class Query(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None

@app.post("/ask")
async def ask_codette(query: Query):
    try:
        response = codette.respond(query.text)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}