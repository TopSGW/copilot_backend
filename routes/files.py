import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings, StorageContext, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

from databases.database import FileRecord, Repository, get_db
from auth import get_current_user, User
from config.config import UPLOAD_DIR

from pdf2image import convert_from_path
# from utils.colpali_manager import ColpaliManager
# from utils.milvus_manager import MilvusManager
import ollama
from config.config import OLLAMA_URL

from celery_worker import process_file_for_training
file_processor = ThreadPoolExecutor(max_workers=25)

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Configure LLM and embedding models
Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=500.0,
    base_url=OLLAMA_URL
)

# Define embedding model explicitly
ollama_embedding = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url=OLLAMA_URL,
)

# Set the embedding model for all indices
Settings.embed_model = ollama_embedding

router = APIRouter(prefix="/files", tags=["files"])

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

    model_config = {
        "from_attributes": True
    }

class FileResponse(BaseModel):
    message: str
    file_metadata: List[FileMetadata]

class FileList(BaseModel):
    repository_id: int
    phone_number: str
    files: List[FileMetadata]

def create_text_file(file_path: str, initial_content: str = ""):
    """
    Creates a text file at the given file_path if it does not already exist.
    Optionally writes an initial content to the file.
    """
    if os.path.exists(file_path):
        print(f"File already exists at {file_path}.")
    else:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(initial_content)
        print(f"File created successfully at: {file_path}")

def append_to_file(file_path: str, content: str):
    """
    Appends the provided content to the file at file_path.
    If the file does not exist, it will be created automatically.
    """
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content + "\n")
    print(f"Content appended to file at: {file_path}")

@router.post("/{repository_id}/upload/", response_model=FileResponse)
async def upload_files_to_repository(
    repository_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Validate repository access
    repository = db.query(Repository).filter(
        Repository.id == repository_id,
        Repository.phone_number == current_user.phone_number
    ).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Create upload directory
    repo_upload_dir = os.path.join(UPLOAD_DIR, current_user.phone_number, str(repository_id))
    os.makedirs(repo_upload_dir, exist_ok=True)
    
    uploaded_files = []

    # Process each file
    for file in files:
        # Save file to disk
        file_location = os.path.join(repo_upload_dir, file.filename)
        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)
        
        # Save file record to database
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
        uploaded_files.append(FileMetadata.model_validate(file_record))
        
        # Submit file processing task to thread pool
        # file_processor.submit(
        #     process_file_for_training, 
        #     file_location, 
        #     current_user.id, 
        #     repository_id
        # )
        result = process_file_for_training.delay(file_location, current_user.id, repository_id)
        print(result)
        result = result.ready()
        print(result)
        print(f"Submitted file {file.filename} for background processing")

    return FileResponse(
        message=f"{len(uploaded_files)} file(s) uploaded successfully. Processing in background.",
        file_metadata=uploaded_files
    )


@router.get("/{repository_id}/", response_model=FileList)
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
        files=[FileMetadata.model_validate(file) for file in files]
    )


@router.delete("/{repository_id}/{filename}", response_model=dict)
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

    real_file_name = os.path.basename(filename)
    print(real_file_name)
    file_record = db.query(FileRecord).filter(
        FileRecord.repository_id == repository_id,
        FileRecord.filename == real_file_name
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


@router.get("/{repository_id}/{filename}", response_model=FileMetadata)
def get_file_metadata(
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

    return FileMetadata.model_validate(file_record)