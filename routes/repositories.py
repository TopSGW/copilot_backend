from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

from databases.database import Repository, get_db
from auth import get_current_user, User

router = APIRouter(prefix="/repositories", tags=["repositories"])

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

@router.post("/", response_model=RepositoryResponse)
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

@router.get("/", response_model=List[RepositoryResponse])
async def list_repositories(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Log the request headers and body for debugging
        headers = request.headers
        body = await request.body()
        logging.info(f"Request headers: {headers}")
        logging.info(f"Request body: {body}")

        # Log the current user information
        logging.info(f"Current user: {current_user}")

        repositories = db.query(Repository).filter(Repository.phone_number == current_user.phone_number).all()
        return repositories
    except Exception as e:
        logging.error(f"An error occurred while fetching repositories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching repositories: {str(e)}")

@router.get("/{repository_id}", response_model=RepositoryResponse)
def get_repository(
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
    return repository

@router.put("/{repository_id}", response_model=RepositoryResponse)
def update_repository(
    repository_id: int,
    repo: RepositoryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    repository = db.query(Repository).filter(
        Repository.id == repository_id,
        Repository.phone_number == current_user.phone_number
    ).first()
    if not repository:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    repository.name = repo.name
    repository.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(repository)
    return repository

@router.delete("/{repository_id}", response_model=dict)
def delete_repository(
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
    
    db.delete(repository)
    db.commit()
    return {"message": f"Repository '{repository.name}' deleted successfully"}