import asyncio
import uvloop
import nest_asyncio
nest_asyncio.apply()

import httpx
from typing import List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

from config.config import SEED
from databases.database import get_db
from auth import get_current_user, UserOut
from routes.websockets import websocket_auth_dialogue, websocket_chat
from routes.repositories import router as repositories_router
from routes.files import router as files_router
from config.config import OLLAMA_URL
# Configure logging
uvloop.install()

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Configuration - consider using environment variables
OLLAMA_ENDPOINTS = [
    f"{OLLAMA_URL}/llama3.3:70b",
    f"{OLLAMA_URL}/llama3.2-vision:90b"
]
CHECK_INTERVAL = 240  # 4 minutes
TIMEOUT = 500  # 500 seconds timeout for requests

async def send_health_checks(client: httpx.AsyncClient, endpoints: List[str]):
    """Send health check requests to Ollama endpoints."""
    tasks = [
        client.post(
            endpoint, 
            json={"prompt": "are you there?"},  # Adjust payload as per Ollama API
            timeout=TIMEOUT
        ) for endpoint in endpoints
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for idx, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            logger.error(f"Health check failed for endpoint {endpoints[idx-1]}: {result}")
        elif hasattr(result, 'status_code'):
            logger.info(f"Health check for endpoint {endpoints[idx-1]}: Status {result.status_code}")

async def continuous_health_checks():
    """Continuously perform health checks on Ollama endpoints."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await send_health_checks(client, OLLAMA_ENDPOINTS)
            except Exception as e:
                logger.error(f"Unexpected error in health checks: {e}")
            
            await asyncio.sleep(CHECK_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict, None]:
    """
    Lifespan context manager to handle startup and shutdown events.
    
    This replaces the deprecated @app.on_event("startup") decorator.
    """
    # Startup tasks
    health_check_task = asyncio.create_task(continuous_health_checks())
    
    try:
        yield {}
    finally:
        # Cleanup tasks on shutdown
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            logger.info("Health check task cancelled successfully")

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(repositories_router)
app.include_router(files_router)

@app.get("/")
def read_root():
    return {"message": "Hello, World! This is your FastAPI backend with phone-based authentication."}

@app.get("/get_user", response_model=UserOut)
async def get_user_endpoint(token: str = Query(..., description="Authentication token")):
    """
    Retrieve user details based on the provided token.
    """
    logger.debug(f"Received token: {token}")
    try:
        user = await get_current_user(token)
        logger.debug(f"User retrieved: {user}")
        return UserOut(id=user.id, phone_number=user.phone_number)
    except Exception as e:
        logger.error(f"Error in get_user_endpoint: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid authentication credentials: {str(e)}")

@app.websocket("/ws/auth-dialogue")
async def websocket_auth_endpoint(websocket: WebSocket):
    await websocket_auth_dialogue(websocket)

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if token is None:
        await websocket.accept()
        await websocket.send_json({"error": "No token provided"})
        await websocket.close(code=1008)
        return
    await websocket_chat(websocket, token)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="uvloop", log_level="debug")