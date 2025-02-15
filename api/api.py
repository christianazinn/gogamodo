from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import json
import uvicorn
from pyngrok import ngrok
import os

# Configure ngrok authentication and settings
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN")  # Your static domain, e.g., "your-domain.ngrok.io"

if NGROK_AUTH_TOKEN: ngrok.set_auth_token(NGROK_AUTH_TOKEN)
else: raise RuntimeError("Please set your ngrok auth token!")
if not NGROK_DOMAIN: print("Warning: ngrok domain is not set!")

# Initialize ngrok tunnel variable
tunnel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create ngrok tunnel
    global tunnel
    port = 6979
    
    try:
        if NGROK_DOMAIN:
            # Connect with custom domain
            tunnel = ngrok.connect(port, domain=NGROK_DOMAIN)
        else:
            # Connect with random URL
            tunnel = ngrok.connect(port)
            
        print(f"ngrok tunnel created: {tunnel.public_url}")
        app.state.public_url = tunnel.public_url
        yield
    finally:
        # Shutdown: close ngrok tunnel
        if tunnel:
            ngrok.disconnect(tunnel.public_url)
        ngrok.kill()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

class ResultBatch(BaseModel):
    results: List[dict]

@app.post("/batch")
async def receive_batch(batch: ResultBatch):
    try:
        # Write results to a file
        with open("results.jsonl", "a") as f:
            for result in batch.results:
                f.write(json.dumps(result) + "\n")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/fail")
async def receive_batch(batch: ResultBatch):
    try:
        # Write results to a file
        with open("failures.jsonl", "a") as f:
            for result in batch.results:
                f.write(json.dumps(result) + "\n")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6979)

# fastapi run api.py --port 6979