import os
from dotenv import load_dotenv

load_dotenv()

SEED = 42

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_here")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "30"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OLLAMA_URL = "http://157.157.221.29:20786"
os.environ['OLLAMA_HOST'] = OLLAMA_URL

props_schema = """
    `page_label` STRING,
    `file_name` STRING,
    `file_path` STRING,
    `file_type` STRING,
    `file_size` INT,
    `creation_date` STRING,
    `last_modified_date` STRING,
    `_node_content` STRING,
    `_node_type` STRING,
    `document_id` STRING,
    `doc_id` STRING,
    `ref_doc_id` STRING,
    `triplet_source_id` STRING
"""
