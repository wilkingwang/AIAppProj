from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

class ChatRequests(BaseModel):
    prompt: str
    model: str = 'deepseek-r1:1.5b'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post('/api/chat')
async def chat(request: ChatRequests):
    base_url = 'http://localhost:11434/api/generate'
    data = {
        'model': request.model,
        'prompt': request.prompt,
        'stream': False
    }

    response = requests.post(url=base_url, json=data)
    if response.status_code == 200:
        return {"response": response.json()["response"]}
    else:
        return {"error": "Failed to get response from ollama"}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)