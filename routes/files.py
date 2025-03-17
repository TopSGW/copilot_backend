import os
import threading
import asyncio
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
from utils.milvus_manager import MilvusManager
import ollama

# Configure LLM and embedding models
Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

# Define embedding model explicitly
ollama_embedding = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
)

# Set the embedding model for all indices
Settings.embed_model = ollama_embedding

router = APIRouter(prefix="/files", tags=["files"])

# Create a thread pool executor for file processing
# Adjust max_workers based on your system's capabilities
file_processor = ThreadPoolExecutor(max_workers=20)

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


def process_file_for_training(file_location: str, user_id: int, repository_id: int):
    """
    Process uploaded files for training and indexing based on file type.
    This function handles different file types and creates appropriate indexes.
    """
    try:
        # Ensure the current thread has an event loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Extract filename and extension
        filename = os.path.basename(file_location)
        temp_dir_name, file_extension = os.path.splitext(filename)
        repo_upload_dir = os.path.dirname(file_location)
        
        print(f"Thread {threading.current_thread().name} - Processing file: {filename}, Extension: {file_extension}")
        
        # Initialize index and vector stores
        index_config = {
            "index_type": "IVF_FLAT",  # Specify the type of index
            "params": {
                "nlist": 128          # Index-specific parameter (number of clusters)
            }
        }

        property_graph_store = NebulaPropertyGraphStore(
            space=f'space_{user_id}'
        )
        graph_vec_store = MilvusVectorStore(
            uri="./milvus_graph.db", 
            collection_name=f"space_{user_id}",
            dim=8192, 
            overwrite=False,         
            similarity_metric="COSINE",
            index_config=index_config
        )

        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=property_graph_store,
            vector_store=graph_vec_store,
            llm=Settings.llm,
            embed_model=Settings.embed_model,  # Explicitly set embedding model
        )

        milvus_manager = MilvusManager(
            milvus_uri="./milvus_original.db",
            collection_name=f"original_{user_id}"    
        )
        milvus_manager.create_index()

        # Vision model prompt
        text_con_prompt = """
            Please analyze the provided image and generate a detailed, plain-language description of its contents. 
            Include key elements such as objects, people, colors, spatial relationships, background details, and any text visible in the image. 
            The goal is to create a comprehensive textual representation that fully conveys the visual information to someone who cannot see the image.
        """

        source_data = SimpleDirectoryReader(input_files=[file_location]).load_data()
        print(source_data[0].metadata)
        # Process based on file type
        match file_extension: 
            case '.txt':
                simple_doc = SimpleDirectoryReader(input_files=[file_location]).load_data()
                for doc in simple_doc:
                    graph_index.insert(doc)

            case '.jpg' | '.png' | '.jpeg':
                txt_response = ollama.chat(
                    model='llama3.2-vision:90b',
                    messages=[{
                        'role': 'user',
                        'content': text_con_prompt,
                        'images': [file_location]
                    }]
                )
                print("text response:", txt_response.message.content)
                print(f"Thread {threading.current_thread().name} - Image text response received")
                txt_file_location = os.path.join(repo_upload_dir, os.path.splitext(filename)[0] + ".txt")

                with open(txt_file_location, "w") as image_file:
                    image_file.write(str(txt_response.message.content))

                simple_doc = SimpleDirectoryReader(input_files=[txt_file_location]).load_data()
                
                for doc in simple_doc: 
                    print("meta data inserting...........")
                    doc.metadata = source_data[0].metadata
                    print("meta data printing....")
                    graph_index.insert(doc)

            case '.pdf':
                # Create a subdirectory to save images (using PDF base name)
                pdf_dir = os.path.join(repo_upload_dir, temp_dir_name)
                os.makedirs(pdf_dir, exist_ok=True)
                images = convert_from_path(file_location)
                
                print(f"Thread {threading.current_thread().name} - PDF with {len(images)} pages")

                # Save all images in the subdirectory
                for i, image in enumerate(images):
                    image_save_path = os.path.join(pdf_dir, f"page_{i}.png")
                    image.save(image_save_path, "PNG")

                # Define the combined text file with the same base name as the PDF file
                pdf_txt_file = os.path.join(repo_upload_dir, f"{temp_dir_name}.txt")
                # Clear any existing content (or create new)
                with open(pdf_txt_file, "w") as f:
                    f.write("")

                # Process each page and append its response to the combined text file
                for i, _ in enumerate(images):
                    image_save_path = os.path.join(pdf_dir, f"page_{i}.png")
                    
                    txt_response = ollama.chat(
                        model='llama3.2-vision:90b',
                        messages=[{
                            'role': 'user',
                            'content': text_con_prompt,
                            'images': [image_save_path]
                        }]
                    )
                    print(f"Thread {threading.current_thread().name} - Processing PDF page {i}")
                    
                    with open(pdf_txt_file, "a") as f:
                        f.write(f"--- Response for page {i} ---\n")
                        f.write(txt_response.message.content)
                        f.write("\n\n")
                
                # Load the combined text file for indexing
                simple_doc = SimpleDirectoryReader(input_files=[pdf_txt_file]).load_data()
                for doc in simple_doc: 
                    doc.metadata = source_data[0].metadata
                    graph_index.insert(doc)

        print(f"Thread {threading.current_thread().name} - Processing completed for: {file_location}")
        
    except Exception as e:
        print(f"Thread {threading.current_thread().name} - Error processing file {file_location}: {str(e)}")


@router.post("/{repository_id}/upload/", response_model=FileResponse)
async def upload_files_to_repository(
    repository_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
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
        
        # # Submit file processing task to thread pool
        file_processor.submit(
            process_file_for_training, 
            file_location, 
            current_user.id, 
            repository_id
        )
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