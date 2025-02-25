import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.llms.ollama import Ollama

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

from databases.database import FileRecord, Repository, get_db
from auth import get_current_user, User
from config.config import UPLOAD_DIR

from llama_index.multi_modal_llms.ollama import OllamaMultiModal

from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.indices.multi_modal.base import (
    MultiModalVectorStoreIndex,
)

Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

mm_model = OllamaMultiModal(model="llava:34b")

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

@router.post("/{repository_id}/upload/", response_model=FileResponse)
async def upload_files_to_repository(
    repository_id: int,
    files: List[UploadFile] = File(...),
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

    # vector_store = MilvusVectorStore(
    #     uri="./milvus_demo.db", 
    #     dim=1536, overwrite=False, 
    #     collection_name=f"user_{current_user.id}"
    # )

    # pipe_line = IngestionPipeline(
    #     transformations=[
    #         SentenceSplitter(chunk_size=2048, chunk_overlap=32),
    #         OpenAIEmbedding(),
    #     ],
    #     vector_store=vector_store,
    # )
    # property_graph_store = NebulaPropertyGraphStore(
    #     space=f'space_{current_user.id}'
    # )
    # graph_vec_store = MilvusVectorStore(
    #     uri="./milvus_demo.db", 
    #     collection_name=f"space_{current_user.id}",
    #     dim=1536, 
    #     overwrite=False,         
    #     metric_type="COSINE",
    #     index_type="IVF_FLAT",
    # )

    # graph_index = PropertyGraphIndex.from_existing(
    #     property_graph_store=property_graph_store,
    #     vector_store=graph_vec_store,
    #     llm=Settings._llm,
    # )
    index_config = {
        "index_type": "IVF_FLAT",  # Specify the type of index
        "params": {
            "nlist": 128          # Index-specific parameter (number of clusters)
        }
    }
    image_embed_model = ClipEmbedding(
        model="openai/clip-rn50x4",  # This model outputs 1536-dim embeddings.
        embed_batch_size=10          # DEFAULT_EMBED_BATCH_SIZE is 10.
    )
    image_vec_store = MilvusVectorStore(
        uri="./milvus_demo.db", 
        collection_name=f"image_{current_user.id}",
        dim=1536,
        overwrite=False,
        similarity_metric="COSINE",
        index_config=index_config

    )

    text_vec_store = MilvusVectorStore(
        uri="./milvus_demo.db", 
        collection_name=f"text_{current_user.id}",
        dim=1536,
        overwrite=False,
        similarity_metric="COSINE",
        index_config=index_config
    )

    index = MultiModalVectorStoreIndex.from_vector_store(
        image_embed_model=image_embed_model,
        vector_store=text_vec_store,
        image_vector_store=image_vec_store
    )

    uploaded_files = []
    for file in files:
        file_location = os.path.join(repo_upload_dir, file.filename)

        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)
        print(f"file location: {file_location}")
        converted_file_location = file_location.replace("\\", "/")
        print(f"converted_file_location: {converted_file_location}")
        documents = SimpleDirectoryReader(input_files=[converted_file_location]).load_data()

        # pipe_line.run(documents=documents)

        for doc in documents:
            index.insert(document=doc)
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

    return FileResponse(
        message=f"{len(uploaded_files)} file(s) uploaded successfully",
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
    # converted_file_location = file_record.storage_path.replace("\\", "/")
    # documents = SimpleDirectoryReader(input_files=[converted_file_location]).load_data()
    # vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1536, overwrite=False, collection_name=f"space_{current_user.id}")
    # ids = vector_store.client.query(
    #     collection_name=f"user_{current_user.id}",
    #     filter="id != ''",
    #     output_fields=["file_path", "doc_id"]    
    # )
    # delete_node_ids = []

    # for item in ids:
    #     id = item.get('id')
    #     file_path = item.get('file_path')
    #     doc_id = item.get('doc_id')
    #     print(f"{id} {file_path} {doc_id}")
    #     if file_path == converted_file_location:
    #         delete_node_ids.append(id)

    # vector_store.delete_nodes(delete_node_ids)
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