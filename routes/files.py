import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings, StorageContext, Document
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore

from databases.database import FileRecord, Repository, get_db
from auth import get_current_user, User
from config.config import UPLOAD_DIR

from llama_index.multi_modal_llms.ollama import OllamaMultiModal

from pdf2image import convert_from_path
from utils.colpali_manager import ColpaliManager
from utils.milvus_manager import MilvusManager
import ollama

Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

ollama_embedding = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
)

router = APIRouter(prefix="/files", tags=["files"])

colpali_manager = ColpaliManager()
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
    
    index_config = {
        "index_type": "IVF_FLAT",  # Specify the type of index
        "params": {
            "nlist": 128          # Index-specific parameter (number of clusters)
        }
    }

    property_graph_store = NebulaPropertyGraphStore(
        space=f'space_{current_user.id}'
    )
    graph_vec_store = MilvusVectorStore(
        uri="./milvus_graph.db", 
        collection_name=f"space_{current_user.id}",
        dim=8192, 
        overwrite=False,         
        similarity_metric="COSINE",
        index_config=index_config
    )

    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=property_graph_store,
        vector_store=graph_vec_store,
        llm=Settings.llm,
    )

    milvus_manager = MilvusManager(
        milvus_uri="./milvus_original.db",
        collection_name=f"original_{current_user.id}"    
    )
    milvus_manager.create_index()
    uploaded_files = []

    text_con_prompt= """
        Please analyze the provided image and generate a detailed, plain-language description of its contents. 
        Include key elements such as objects, people, colors, spatial relationships, background details, and any text visible in the image. 
        The goal is to create a comprehensive textual representation that fully conveys the visual information to someone who cannot see the image.
    """

    for file in files:
        file_location = os.path.join(repo_upload_dir, file.filename)
        temp_dir_name, file_extension = os.path.splitext(file.filename) # Extracting the extension name

        print(f"Uploaded file: {file.filename}, Extension: {file_extension}")

        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)
        
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
                print("text message: ", txt_response.message)
                txt_file_location = os.path.join(repo_upload_dir, os.path.splitext(file.filename)[0] + ".txt")

                with open(txt_file_location, "w") as m_file:
                    m_file.write(str(txt_response.message))

                simple_doc = SimpleDirectoryReader(input_files=[txt_file_location]).load_data()
                
                for doc in simple_doc: 
                    graph_index.insert(doc)

                colbert_vecs = colpali_manager.process_images(image_paths=[file_location])

                image_paths = []
                image_paths.append(file_location)    

                images_data = [{
                    "colbert_vecs": colbert_vecs[i],
                    "filepath": image_paths[i]
                } for i in range(len(image_paths))]

                milvus_manager.insert_images_data(images_data)

            case '.pdf':
                pdf_dir = os.path.join(repo_upload_dir, temp_dir_name)
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = file_location
                images = convert_from_path(pdf_path)
                image_paths = []
                for i, image in enumerate(images):
                    image_save_path = os.path.join(pdf_dir, f"page_{i}.png")
                    txt_save_path = os.path.join(pdf_dir, f"page_{i}.txt")
                    image.save(image_save_path, "PNG")
                    image_paths.append(image_save_path)

                    txt_response = ollama.chat(
                        model='llava:34b',
                        messages=[{
                            'role': 'user',
                            'content': text_con_prompt,
                            'images': [image_save_path]
                        }]
                    )
                    print("text message: ", txt_response.message)
                    with open(txt_save_path, "w") as m_file:
                        m_file.write(str(txt_response.message))

                    simple_doc = SimpleDirectoryReader(input_files=[txt_save_path]).load_data()
                    
                    for doc in simple_doc: 
                        graph_index.insert(doc)

                colbert_vecs = colpali_manager.process_images(image_paths=image_paths)

                images_data = [{
                    "colbert_vecs": colbert_vecs[i],
                    "filepath": image_paths[i]
                } for i in range(len(image_paths))]

                milvus_manager.insert_images_data(images_data)

        print(f"file location: {file_location}")
        converted_file_location = file_location.replace("\\", "/")
        print(f"converted_file_location: {converted_file_location}")

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