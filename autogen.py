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
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from passlib.context import CryptContext
from dotenv import load_dotenv

# AutoGen Imports
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

# Configuration & Environment Variables
load_dotenv()

SEED = 42
nest_asyncio.apply()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_here")
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

# Database Setup (SQLAlchemy)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    repositories = relationship("Repository", back_populates="owner")

class Repository(Base):
    __tablename__ = "repositories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    phone_number = Column(String, ForeignKey("users.phone_number"))
    owner = relationship("User", back_populates="repositories")
    files = relationship("FileRecord", back_populates="repository")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FileRecord(Base):
    __tablename__ = "file_records"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)
    file_size = Column(Integer)
    mime_type = Column(String)
    phone_number = Column(String, ForeignKey("users.phone_number"))
    repository_id = Column(Integer, ForeignKey("repositories.id"))
    repository = relationship("Repository", back_populates="files")
    storage_path = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
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

class RepositoryCreate(BaseModel):
    name: str

class RepositoryResponse(BaseModel):
    id: int
    name: str
    phone_number: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class FileMetadata(BaseModel):
    filename: str
    original_filename: str
    file_size: int
    repository_id: int
    phone_number: str
    mime_type: str
    storage_path: str
    upload_date: datetime
    last_modified: datetime

    class Config:
        from_attributes = True

class FileResponse(BaseModel):
    message: str
    file_metadata: FileMetadata

class FileList(BaseModel):
    repository_id: int
    phone_number: str
    files: List[FileMetadata]

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# JWT Token Creation
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI App Initialization & OAuth2 Setup
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for file uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper Functions for User Management
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

    db = next(get_db())
    try:
        user = get_user_by_phone(db, phone_number)
        if user is None:
            raise credentials_exception
    finally:
        db.close()
    return user

@app.get("/get_user", response_model=UserOut)
def get_user_endpoint(token: str):
    user = get_current_user(token)
    return user

# AutoGen & Chat Endpoints
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
If user want sign in, set the json value to "sign-in". Or user want sign up, set the json value to "sign-up". 
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
    system_message="You are a professional and knowledgeable AI assistant powered by Retrieval-Augmented Generation (RAG). Once the user has successfully signed in or registered, please proceed to address their queries with clarity, accuracy, and promptness. Generally, please answer with get_answer function calling because you are rag assistant for local documents. However, if user ask general question, you can ask LLM",
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

# WebSocket Endpoints
@app.websocket("/ws/auth-dialogue")
async def websocket_auth_dialogue(websocket: WebSocket):
    await websocket.accept()
    db = next(get_db())
    print("Auth WebSocket accepted")
    websocket_open = True
    try:
        while websocket_open:
            try:
                data = await websocket.receive_text()
                print("Received auth data:", data)
            except Exception as e:
                print("Auth receive error or disconnect:", e)
                websocket_open = False
                break

            auth_input_data = json.loads(data)
            user_input = auth_input_data.get("user_input", "")
            auth_data = await run_auth_agent(user_input)
            print("Auth agent returned:", auth_data)
            action = auth_data.get("action")
            phone = auth_data.get("user_number") or auth_data.get("phone_number")
            password = auth_data.get("password")

            if action == "" or phone == "" or password == "":
                await websocket.send_json({
                    "message": auth_data.get("instruction"),
                    "status": False
                })
                continue

            token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
            if action.lower() == "sign-up":
                if get_user_by_phone(db, phone):
                    await websocket.send_json({
                        "error": "Phone number already registered. Please sign in."
                    })
                    continue
                hashed_pw = get_password_hash(password)
                new_user = User(phone_number=phone, hashed_password=hashed_pw)
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                token = create_access_token(
                    data={"sub": new_user.phone_number},
                    expires_delta=token_expires
                )
                msg_response = await authenticate_agent.on_messages(
                    [TextMessage(content="Sign up successful!", source="auth_agent")],
                    cancellation_token=CancellationToken(),
                )
            elif action.lower() == "sign-in":
                user = authenticate_user(db, phone, password)
                if not user:
                    await websocket.send_json({
                        "error": "Incorrect phone number or password."
                    })
                    continue
                token = create_access_token(
                    data={"sub": user.phone_number},
                    expires_delta=token_expires
                )
                msg_response = await authenticate_agent.on_messages(
                    [TextMessage(content="Sign in successful", source="auth_agent")],
                    cancellation_token=CancellationToken(),
                )
            else:
                await websocket.send_json({
                    "message": auth_data.get("instruction"),
                    "status": False
                })
                continue

            print("Sending auth response with token.")
            await websocket.send_json({
                "message": msg_response.chat_message.content,
                "token": token,
                "status": True
            })
    except Exception as e:
        print("Auth endpoint exception:", e)
    finally:
        db.close()
        if websocket_open:
            await websocket.close()
        print("Auth WebSocket closed.")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if token is None:
        print("No token provided. Accepting connection to send error message.")
        await websocket.accept()
        await websocket.send_json({"error": "No token provided"})
        await websocket.close(code=1008)
        return
    try:
        user = get_current_user(token)
    except HTTPException:
        print("Invalid token. Accepting connection to send error message.")
        await websocket.accept()
        await websocket.send_json({"error": "Token has expired or is invalid"})
        await websocket.close(code=1008)
        return

    await websocket.accept()
    print("Chat WebSocket accepted for user:", user.phone_number)
    websocket_open = True
    while websocket_open:
        try:
            data = await websocket.receive_text()
            print("Received data:", data)
        except Exception as e:
            print("Receive error or disconnect:", e)
            websocket_open = False
            break

        try:
            auth_input_data = json.loads(data)
            user_input = auth_input_data.get("user_input", "")
            print("Processing user input:", user_input)
            response = vector_rag.run(user_input)
            print("Response from vector_rag:", response)
            await websocket.send_json({"message": response})
        except Exception as e:
            print("Error processing message:", e)
            await websocket.send_json({"message": "An error occurred processing your request."})
    print("Closing chat WebSocket.")
    if websocket_open:
        await websocket.close()

# Repository & File Management Endpoints
@app.post("/repositories/", response_model=RepositoryResponse)
def create_repository(
    repo: RepositoryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_repo = Repository(name=repo.name, phone_number=current_user.phone_number)
    db.add(new_repo)
    db.commit()
    db.refresh(new_repo)
    return new_repo

@app.get("/repositories/", response_model=List[RepositoryResponse])
def list_repositories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Repository).filter(Repository.phone_number == current_user.phone_number).all()

@app.post("/repositories/{repository_id}/upload/", response_model=FileResponse)
async def upload_file_to_repository(
    repository_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repository = db.query(Repository).filter(
        Repository.id == repository_id,
        Repository.phone_number == current_user.phone_number
    ).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    repo_upload_dir = os.path.join(UPLOAD_DIR, current_user.phone_number, str(repository_id))
    os.makedirs(repo_upload_dir, exist_ok=True)
    file_location = os.path.join(repo_upload_dir, file.filename)
    
    content = await file.read()
    with open(file_location, "wb") as f:
        f.write(content)

    file_record = FileRecord(
        filename=file.filename,
        original_filename=file.filename,
        file_size=len(content),
        mime_type=file.content_type,
        phone_number=current_user.phone_number,
        repository_id=repository.id,
        storage_path=file_location
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)

    return FileResponse(
        message="File uploaded successfully",
        file_metadata=FileMetadata.from_orm(file_record)
    )

@app.get("/repositories/{repository_id}/files/", response_model=FileList)
def list_files_in_repository(
    repository_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repository = db.query(Repository).filter(
        Repository.id == repository_id,
        Repository.phone_number == current_user.phone_number
    ).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    files = db.query(FileRecord).filter(FileRecord.repository_id == repository_id).all()
    return FileList(
        repository_id=repository.id,
        phone_number=current_user.phone_number,
        files=[FileMetadata.from_orm(file) for file in files]
    )

@app.delete("/repositories/{repository_id}/files/{filename}", response_model=dict)
async def delete_file(
    repository_id: int,
    filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repository = db.query(Repository).filter(
        Repository.id == repository_id,
        Repository.phone_number == current_user.phone_number
    ).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    file_record = db.query(FileRecord).filter(
        FileRecord.repository_id == repository_id,
        FileRecord.filename == filename
    ).first()
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    if os.path.exists(file_record.storage_path):
        os.remove(file_record.storage_path)

    db.delete(file_record)
    db.commit()

    return {
        "message": "File deleted successfully",
        "filename": filename
    }

# Public Endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, World! This is your FastAPI backend with phone-based authentication."}

# Application Entry Point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)