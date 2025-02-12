import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import openai
from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt  # python-jose for JWT operations
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from dotenv import load_dotenv
import re

# -------------------------------------------------------------------------
# AutoGen Imports (replacing CrewAI)
# -------------------------------------------------------------------------
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination

import shutil
import nest_asyncio
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
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")


embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
extraction_llm = OpenAI(model="gpt-4o-mini", temperature=0.0, seed=SEED)
generation_llm = OpenAI(model="gpt-4o-mini", temperature=0.3, seed=SEED)

# Load the dataset on Larry Fink
original_documents = SimpleDirectoryReader("./data/blackrock").load_data()
# print(len(original_documents))

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
# Pydantic Models
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
    """
    Create a JWT access token.
    """
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


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
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

    user = get_user_by_phone(db, phone_number)
    if user is None:
        raise credentials_exception
    return user


# =============================================================================
# API Endpoints (Signup, Token, Chat)
# =============================================================================

@app.post("/signup", response_model=UserOut)
def signup(user_create: UserCreate, db: Session = Depends(get_db)):
    """
    User registration endpoint.
    """
    if get_user_by_phone(db, user_create.phone_number):
        raise HTTPException(status_code=400, detail="Phone number already registered")
    hashed_pw = get_password_hash(user_create.password)
    new_user = User(phone_number=user_create.phone_number, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    User login endpoint that returns a JWT token.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect phone number or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": user.phone_number}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# =============================================================================
# AutoGen Integration for Authentication
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
    return {
        "query_result": vector_rag.run(user_input)
    }

authenticate_agent = AssistantAgent("auth_agent", model_client, system_message=system_prompt)

rag_agent = AssistantAgent(name="rag_agent", model_client=model_client, system_message="You are a professional and knowledgeable AI assistant powered by Retrieval-Augmented Generation (RAG). Once the user has successfully signed in or registered, please proceed to address their queries with clarity, accuracy, and promptness.",
                           tools=[get_answer])
agent_team = RoundRobinGroupChat([authenticate_agent], max_turns=1)
# critic_agent = AssistantAgent(
#     "critic",
#     model_client=model_client,
#     system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
# )

rag_team = RoundRobinGroupChat([rag_agent], max_turns=1)

async def run_auth_agent(user_input: str) -> dict:
    """
    Uses AutoGen's AssistantAgent to process an authentication conversation.
    The agent is prompted to ask the user whether they want to 'signup' or 'signin'
    and then request their phone number and password.
    It returns a JSON object with keys: 'action', 'phone_number', and 'password'.
    """
    task_prompt = (
        f"The user says: '{user_input}'.\n\n"

    )
    response = await agent_team.run(task=task_prompt)
    print(response.messages[1].content)
    if "```json" in response.messages[1].content:
        pattern = r"```json(.*)```"
        match = re.search(pattern, response.messages[1].content, re.DOTALL)
        message = match.group(1) if match else response.messages[1].content
        return json.loads(message)
    else: 
        return {
            "instruction": response.messages[1].content,
            "action": "ask",
            "phone_number": "",
            "password": ""
        }

@app.post("/auth-dialogue")    
async def auth_crew_endpoint(
    auth_input: AuthCrewInput = Body(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to handle authentication using AutoGen.
    It invokes the AutoGen agent to interpret user instructions, parses the resulting JSON,
    and performs the requested action (signup or signin).
    """
    auth_data = await run_auth_agent(auth_input.user_input)

    action = auth_data.get("action")
    phone = auth_data.get("phone_number")
    password = auth_data.get("password")

    print(auth_data)
    if action == "" or phone == "" or password =="":
        return {
            "message": auth_data.get('instruction'),
            "status": False
        }
    token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_HOURS)
    if action.lower() == "sign-up":
        if get_user_by_phone(db, phone):
            raise HTTPException(status_code=400, detail="Phone number already registered. Please sign in.")
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
        text_message = TextMessage(content="Sign up successful!", source="rag_agent")

    elif action.lower() == "sign-in":
        user = authenticate_user(db, phone, password)
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect phone number or password.")
        token = create_access_token(data={"sub": user.phone_number}, expires_delta=token_expires)
        msg_response = await authenticate_agent.on_messages(        
            [TextMessage(content="Sign in successful", source="auth_agent")],
            cancellation_token=CancellationToken(),
        )
        text_message = TextMessage(content="Sign in successful!", source="rag_agent")
    return {
        "message": msg_response.chat_message.content,
        "token": token,
        "status": True
    }

@app.get("/chat")
async def chat_interface(
    current_user: User = Depends(get_current_user),     
    auth_input: AuthCrewInput = Body(...),
):
    response = await rag_agent.on_messages(
        [TextMessage(content=auth_input.user_input, source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message)

    """
    A protected endpoint to demonstrate chat integration.
    """
    return {"message": response.chat_message.content}
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
