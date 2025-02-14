from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, ValidationError
from typing import List
from datetime import datetime
import logging

from databases.database import Repository, get_db
from auth import get_current_user, User

router = APIRouter(prefix="/repositories", tags=["repositories"])

logger = logging.getLogger(__name__)

class RepositoryCreate(BaseModel):
    name: str

class RepositoryResponse(BaseModel):
    id: int
    name: str
    phone_number: str
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }

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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Listing repositories for user: {current_user.phone_number}")
        repositories = db.query(Repository).filter(Repository.phone_number == current_user.phone_number).all()
        logger.info(f"Retrieved {len(repositories)} repositories")

        response_data = [RepositoryResponse.model_validate(repo) for repo in repositories]
        logger.info("Successfully validated repository data")
        return response_data
    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Data validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"An error occurred while fetching repositories: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")

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