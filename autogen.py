import os
import json
import asyncio
import shutil
import nest_asyncio
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from typing import Optional, List

import requests
import openai
from fastapi import FastAPI, Depends, HTTPException, status, Body, File, UploadFile, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt  # python-jose for JWT operations
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# AutoGen Imports (replacing CrewAI)
# -------------------------------------------------------------------------
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination

from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from vector_rag import VectorRAG

# =============================================================================
# Configuration & Environment Variables
# =============================================================================

load_dotenv()

SEED = 42
nest_asyncio.apply()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_here")  # Change in production!
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "30"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
extraction_llm = OpenAI(model="gpt-4o-mini", temperature=0.0, seed=SEED)
generation_llm = OpenAI(model="gpt-4o-mini", temperature=0.3, seed=SEED)

# Load the dataset on Larry Fink
original_documents = SimpleDirectoryReader("./data/blackrock").load_data()
# --- Step 1: Chunk and store the vector embeddings in LanceDB ---
shutil.rmtree("./test_lancedb", ignore_errors=True)
openai.api_key = OPENAI_API_KEY
vector_store = LanceDBVectorStore(
    uri="./test_lancedb",
    mode="overwrite",
)
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=32),
        OpenAIEmbedding(),
    ],
    vector_store=vector_store,
)
pipeline.run(documents=original_documents)

# =============================================================================
# Database Setup (SQLAlchemy)
# =============================================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)


# Create tables (for production, use Alembic migrations)
Base.metadata.create_all(bind=engine)

# =============================================================================
# Pydantic Models (User and New Repository/File Models)
# =============================================================================

class UserCreate(BaseModel):
    phone_number: str
    password: str


class UserOut(BaseModel):
    id: int
    phone_number: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


# --- New Models for Repositories and File Uploads ---
class RepositoryCreate(BaseModel):
    name: str


class Repository(BaseModel):
    id: str
    name: str
    owner: int  # user id of the repository owner


class FileInfo(BaseModel):
    id: str
    repository_id: str
    filename: str
    file_url: str  # Local path or URL to the file


# =============================================================================
# Password Hashing
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# JWT Token Creation
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# =============================================================================
# Database Dependency
# =============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# FastAPI App Initialization & OAuth2 Setup
# =============================================================================

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Enable CORS (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Repositories and File Upload Setup (In-Memory)
# =============================================================================

# Base directory for file uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for repositories and file metadata
repositories = {}  # repo_id -> { "id": repo_id, "name": ..., "owner": user_id }
files_db = {}      # repo_id -> list of file info dictionaries

# =============================================================================
# Helper Functions for User Management
# =============================================================================

def get_user_by_phone(db: Session, phone_number: str) -> Optional[User]:
    return db.query(User).filter(User.phone_number == phone_number).first()


def authenticate_user(db: Session, phone_number: str, password: str) -> Optional[User]:
    user = get_user_by_phone(db, phone_number)
    if user and verify_password(password, user.hashed_password):
        return user
    return None


def get_current_user(token: str) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        phone_number: str = payload.get("sub")
        if phone_number is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # For WebSocket endpoints, get a DB session manually
    db = next(get_db())
    try:
        user = get_user_by_phone(db, phone_number)
        if user is None:
            raise credentials_exception
    finally:
        db.close()
    return user

# =============================================================================
# AutoGen & Chat Endpoints (Unchanged HTTP Endpoints Omitted for Brevity)
# =============================================================================

class AuthCrewInput(BaseModel):
    user_input: str

model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=OPENAI_API_KEY)
system_prompt = """
You are a Retrieval Augmented Generation (RAG) system designed to deliver comprehensive document analysis and question answering, with a particular emphasis on accounting and financial documents.
To ensure secure access, users must sign in. Please instruct users to sign in, and if they do not have an account, kindly guide them through the account registration process.
Step 1: Determine whether the user intends to sign-up (create a new account) or sign-in (access an existing account).
Step 2: Request that the user provide their phone number. Since phone numbers can be entered in various formats, please convert the input into a standardized format. For example, convert "+1 235-451-1236" to "+12354511236".
Step 3: Request that the user provide their password.
Output your instructions and the collected information as a JSON string with exactly the following keys: "instruction", "action", "phone_number", and "password".
If the necessary credential information is not provided, please offer clear and courteous guidance to assist the user.
Ensure that the final output is strictly in JSON format without any additional commentary.

Example output:
```json
{
    "instruction": "",
    "action": "",
    "phone_number": "",
    "password": ""
}
```

"""

vector_rag = VectorRAG("./test_lancedb")
async def get_answer(user_input: str) -> str:
    return {"query_result": vector_rag.run(user_input)}
authenticate_agent = AssistantAgent("auth_agent", model_client, system_message=system_prompt)
rag_agent = AssistantAgent(
    name="rag_agent",
    model_client=model_client,
    system_message="You are a professional and knowledgeable AI assistant powered by Retrieval-Augmented Generation (RAG). Once the user has successfully signed in or registered, please proceed to address their queries with clarity, accuracy, and promptness.",
    tools=[get_answer]
)
agent_team = RoundRobinGroupChat([authenticate_agent], max_turns=1)
async def run_auth_agent(user_input: str) -> dict:
    task_prompt = f"The user says: '{user_input}'.\n\n"
    response = await agent_team.run(task=task_prompt)
    print(response.messages[1].content)
    if "```json" in response.messages[1].content:
        pattern = r"```json(.*)```"
        match = re.search(pattern, response.messages[1].content, re.DOTALL)
        message = match.group(1) if match else response.messages[1].content
        return json.loads(message)
    else:
        return {"instruction": response.messages[1].content, "action": "ask", "phone_number": "", "password": ""}

# =============================================================================
# WebSocket Endpoints
# =============================================================================

# WebSocket endpoint for authentication dialogue (sign-up/sign-in)
@app.websocket("/ws/auth-dialogue")
async def websocket_auth_dialogue(websocket: WebSocket):
    await websocket.accept()
    # Expect a JSON message with the key "user_input"
    data = await websocket.receive_text()
    auth_input_data = json.loads(data)
    user_input = auth_input_data.get("user_input", "")
    auth_data = await run_auth_agent(user_input)
    action = auth_data.get("action")
    phone = auth_data.get("phone_number")
    password = auth_data.get("password")
    print(auth_data)
    if action == "" or phone == "" or password == "":
        await websocket.send_json({"message": auth_data.get("instruction"), "status": False})
        await websocket.close()
        return
    token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_HOURS)
    db = next(get_db())
    try:
        if action.lower() == "sign-up":
            if get_user_by_phone(db, phone):
                await websocket.send_json({"error": "Phone number already registered. Please sign in."})
                await websocket.close()
                return
            hashed_pw = get_password_hash(password)
            new_user = User(phone_number=phone, hashed_password=hashed_pw)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            token = create_access_token(data={"sub": new_user.phone_number}, expires_delta=token_expires)
            msg_response = await authenticate_agent.on_messages(
                [TextMessage(content="Sign up successful!", source="auth_agent")],
                cancellation_token=CancellationToken(),
            )
        elif action.lower() == "sign-in":
            user = authenticate_user(db, phone, password)
            if not user:
                await websocket.send_json({"error": "Incorrect phone number or password."})
                await websocket.close()
                return
            token = create_access_token(data={"sub": user.phone_number}, expires_delta=token_expires)
            msg_response = await authenticate_agent.on_messages(
                [TextMessage(content="Sign in successful", source="auth_agent")],
                cancellation_token=CancellationToken(),
            )
        await websocket.send_json({"message": msg_response.chat_message.content, "token": token, "status": True})
    finally:
        db.close()
    await websocket.close()

# WebSocket endpoint for chat; expects a valid token as a query parameter.
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    # Retrieve token from query parameters
    token = websocket.query_params.get("token")
    if token is None:
        await websocket.close(code=1008)
        return
    try:
        user = get_current_user(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    # Expect a JSON message with "user_input"
    data = await websocket.receive_text()
    auth_input_data = json.loads(data)
    user_input = auth_input_data.get("user_input", "")
    response = await rag_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken(),
    )
    await websocket.send_json({"message": response.chat_message.content})
    await websocket.close()

# =============================================================================
# New Repository & File Upload Endpoints
# =============================================================================

@app.post("/repositories/", response_model=Repository)
def create_repository(
    repo: RepositoryCreate,
    current_user: User = Depends(get_current_user)
):
    repo_id = str(uuid4())
    new_repo = {"id": repo_id, "name": repo.name, "owner": current_user.id}
    repositories[repo_id] = new_repo
    files_db[repo_id] = []  # initialize empty list for files
    return new_repo

@app.get("/repositories/", response_model=List[Repository])
def list_repositories(current_user: User = Depends(get_current_user)):
    # Return repositories owned by the current user
    return [repo for repo in repositories.values() if repo.get("owner") == current_user.id]

@app.post("/repositories/{repo_id}/upload/", response_model=FileInfo)
async def upload_file_to_repository(
    repo_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    # Check repository exists and belongs to current user
    if repo_id not in repositories or repositories[repo_id].get("owner") != current_user.id:
        raise HTTPException(status_code=403, detail="Repository not found or not authorized")
    # Create directory structure: uploads/<phone_number>/repositories/<repo_id>/files
    repo_upload_dir = os.path.join(
        UPLOAD_DIR,
        current_user.phone_number,
        "repositories",
        repo_id,
        "files"
    )
    os.makedirs(repo_upload_dir, exist_ok=True)
    file_location = os.path.join(repo_upload_dir, file.filename)
    content = await file.read()
    with open(file_location, "wb") as f:
        f.write(content)
    file_id = str(uuid4())
    file_info = {
        "id": file_id,
        "repository_id": repo_id,
        "filename": file.filename,
        "file_url": file_location,
    }
    files_db[repo_id].append(file_info)
    return file_info

@app.get("/repositories/{repo_id}/files/", response_model=List[FileInfo])
def list_files_in_repository(
    repo_id: str,
    current_user: User = Depends(get_current_user)
):
    if repo_id not in repositories or repositories[repo_id].get("owner") != current_user.id:
        raise HTTPException(status_code=403, detail="Repository not found or not authorized")
    return files_db.get(repo_id, [])

# =============================================================================
# Public Endpoint
# =============================================================================

@app.get("/")
def read_root():
    return {"message": "Hello, World! This is your FastAPI backend with phone-based authentication."}

# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
