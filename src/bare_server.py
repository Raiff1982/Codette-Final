"""
Bare minimum FastAPI server
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/ask")
def ask(query: Query):
    return {"response": f"Echo: {query.text}"}