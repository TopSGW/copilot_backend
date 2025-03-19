import asyncio
import uvloop

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

from config.config import SEED
from databases.database import get_db
from auth import get_current_user, UserOut
from routes.websockets import websocket_auth_dialogue, websocket_chat
from routes.repositories import router as repositories_router
from routes.files import router as files_router
# Configure logging
uvloop.install()

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

app = FastAPI()

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