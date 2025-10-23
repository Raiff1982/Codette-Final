
from fastapi import FastAPI
from pydantic import BaseModel
from codette.codette_core import AICore

app = FastAPI()
codette = AICore()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/codette/respond")
def respond(prompt_request: PromptRequest):
    result = codette.process_input(prompt_request.prompt)
    return {"response": result}
