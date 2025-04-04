import asyncio
import uvloop
import nest_asyncio
nest_asyncio.apply()

import httpx
from typing import List, AsyncGenerator, Dict, Any
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
from config.config import OLLAMA_URL, LLAMA_MODEL, LLAMA_VISION_MODEL
# Configure logging
uvloop.install()

# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Configuration - consider using environment variables
OLLAMA_BASE_URL = f"{OLLAMA_URL}/api"
OLLAMA_MODELS = [LLAMA_MODEL, LLAMA_VISION_MODEL]
CHECK_INTERVAL = 240  # 4 minutes
TIMEOUT = 200  # 200 seconds timeout for requests

async def check_ollama_model(client: httpx.AsyncClient, model: str) -> Dict[str, Any]:
    """
    Perform a health check on a specific Ollama model by attempting to load it.
    
    Args:
        client: Async HTTP client
        model: Name of the model to check
    
    Returns:
        Dictionary with health check results
    """
    try:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/generate", 
            json={
                "model": model,
                "prompt": "",  # Empty prompt to just load the model
                "stream": False
            },
            timeout=TIMEOUT
        )
        
        # Check if the response is successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        return {
            "model": model,
            "status": "healthy",
            "created_at": result.get("created_at"),
            "done": result.get("done", False)
        }
    
    except httpx.RequestError as e:
        logger.error(f"Error checking model {model}: {e}")
        return {
            "model": model,
            "status": "unhealthy",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error checking model {model}: {e}")
        return {
            "model": model,
            "status": "error",
            "error": str(e)
        }

async def continuous_model_health_checks():
    """
    Continuously perform health checks on configured Ollama models.
    """
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Perform health checks for all models
                health_checks = await asyncio.gather(
                    *[check_ollama_model(client, model) for model in OLLAMA_MODELS]
                )
                
                # Log health check results
                for check in health_checks:
                    if check['status'] == 'healthy':
                        logger.info(f"Model {check['model']} is healthy")
                    else:
                        logger.warning(f"Model health check failed: {check}")
            
            except Exception as e:
                logger.error(f"Unexpected error in health checks: {e}")
            
            # Wait before next round of checks
            await asyncio.sleep(CHECK_INTERVAL)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict, None]:
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup tasks
    health_check_task = asyncio.create_task(continuous_model_health_checks())
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="uvloop")