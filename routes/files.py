import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime

from databases.database import FileRecord, Repository, get_db
from auth import get_current_user, User
from config.config import UPLOAD_DIR

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

    class Config:
        from_attributes = True

class FileResponse(BaseModel):
    message: str
    file_metadata: FileMetadata

class FileList(BaseModel):
    repository_id: int
    phone_number: str
    files: List[FileMetadata]

@router.post("/{repository_id}/upload/", response_model=FileResponse)
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
        files=[FileMetadata.from_orm(file) for file in files]
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

    return FileMetadata.from_orm(file_record)