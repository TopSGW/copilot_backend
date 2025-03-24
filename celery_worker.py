from celery import Celery
import os
import logging
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from pdf2image import convert_from_path
import ollama
from config.config import OLLAMA_URL
import requests
import json
import base64
import nest_asyncio
nest_asyncio.apply()
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a stream handler (console output)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Initialize the LLM
Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url=OLLAMA_URL
)

# Define embedding model explicitly
ollama_embedding = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url=OLLAMA_URL,
)

# Set the embedding model for all indices
Settings.embed_model = ollama_embedding

# Configure Celery with Redis as broker
celery_app = Celery("worker", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")

# Initialize a node parser for document processing
node_parser = SimpleNodeParser.from_defaults()

def get_base64_encoded_image(image_path):
    """Convert an image to base64 encoding for API requests"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_with_ollama(image_path, prompt, ollama_url):
    """Process an image using Ollama API directly via HTTP"""
    base64_image = get_base64_encoded_image(image_path)
    
    # Prepare the request payload
    payload = {
        "model": "llama3.2-vision:90b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [ base64_image ],
            }
        ],
        "stream": False
    }
    
    # Make the API call
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{ollama_url}/api/chat",
        headers=headers,
        data=json.dumps(payload),
        timeout=300
    )
    
    # Check if the request was successful
    if response.status_code == 200:
        try:
            logger.info(f"Ollama API response: {response.text}")
            result = json.loads(response.text)
            return result["message"]["content"]
        except (json.JSONDecodeError, KeyError) as e:
            raise Exception(f"Error parsing response JSON: {e}\nRaw response: {response.text}")
    else:
        raise Exception(f"Ollama API call failed with status code {response.status_code}: {response.text}")

def clean_metadata(metadata, page_number=None):
    """
    Clean and prepare metadata for graph insertion
    """
    cleaned_metadata = {
        'file_name': metadata.get('file_name', 'Unknown'),
        'file_path': metadata.get('file_path', 'Unknown'),
        'file_type': metadata.get('file_type', 'Unknown'),
        'file_size': metadata.get('file_size', 0),
        'page_number': page_number or 0,
        'creation_date': metadata.get('creation_date', time.strftime('%Y-%m-%d')),
        'last_modified_date': metadata.get('last_modified_date', time.strftime('%Y-%m-%d')),
    }
    return cleaned_metadata

@celery_app.task
def process_file_for_training(file_location: str, user_id: int, repository_id: int):
    """
    Process uploaded files for training and indexing based on file type.
    This function handles different file types and creates appropriate indexes.
    """
    logger.info("Starting processing for file: %s", file_location)
    try:
        # Extract filename and extension
        filename = os.path.basename(file_location)
        temp_dir_name, file_extension = os.path.splitext(filename)
        repo_upload_dir = os.path.dirname(file_location)
             
        property_graph_store = NebulaPropertyGraphStore(
            space=f'space_{user_id}'
        )

        # Initialize index and vector stores
        index_config = {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 128
            }
        }

        graph_vec_store = MilvusVectorStore(
            uri="http://localhost:19530", 
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
            embed_model=Settings.embed_model,
        )
        
        # Vision model prompt
        text_con_prompt = """
            Please analyze the provided image and generate a detailed, plain-language description of its contents. 
            Include key elements such as objects, people, colors, spatial relationships, background details, and any text visible in the image. 
            The goal is to create a comprehensive textual representation that fully conveys the visual information to someone who cannot see the image.
        """

        source_data = SimpleDirectoryReader(input_files=[file_location]).load_data()
        logger.info(f"Source data metadata: {source_data[0].metadata}")
        
        # Create a thread-local event loop policy
        try:
            # Process based on file type
            match file_extension: 
                case '.txt':
                    logger.info("Starting text file processing............")
                    simple_doc = SimpleDirectoryReader(input_files=[file_location]).load_data()
                    logger.info("Simple doc has been loaded")
                    
                    for doc in simple_doc:
                        # Use the standard insert method but catch and handle event loop errors
                        try:
                            # PropertyGraphIndex doesn't have node_parser, use the regular insert method
                            # but inside try/except to handle any event loop issues
                            graph_index.insert(doc)
                        except RuntimeError as e:
                            if "Event loop is closed" in str(e):
                                # If we get an event loop error, log it and continue
                                logger.warning(f"Event loop error occurred, continuing: {str(e)}")
                                # Wait a moment before continuing
                                import time
                                time.sleep(1)
                                # Try the insert again with a fresh connection
                                graph_index = PropertyGraphIndex.from_existing(
                                    property_graph_store=property_graph_store,
                                    vector_store=graph_vec_store,
                                    llm=Settings.llm,
                                    embed_model=Settings.embed_model,
                                )
                                graph_index.insert(doc)
                            else:
                                raise
                    
                    logger.info(f"{file_location} Text file processed successfully")

                case '.jpg' | '.png' | '.jpeg':
                    txt_response = process_image_with_ollama(
                        file_location, 
                        text_con_prompt,
                        OLLAMA_URL
                    )
                    logger.info(f"Text response received from vision model")
                    logger.info(f"text response {txt_response}")
                    txt_file_location = os.path.join(repo_upload_dir, os.path.splitext(filename)[0] + ".txt")

                    with open(txt_file_location, "w") as image_file:
                        image_file.write(str(txt_response))

                    simple_doc = SimpleDirectoryReader(input_files=[txt_file_location]).load_data()
                    
                    for doc in simple_doc: 
                        logger.info("Setting metadata and inserting document")
                        doc.metadata = source_data[0].metadata
                        try:
                            graph_index.insert(doc)
                        except RuntimeError as e:
                            if "Event loop is closed" in str(e):
                                logger.warning(f"Event loop error occurred, continuing: {str(e)}")
                                import time
                                time.sleep(1)
                                graph_index = PropertyGraphIndex.from_existing(
                                    property_graph_store=property_graph_store,
                                    vector_store=graph_vec_store,
                                    llm=Settings.llm,
                                    embed_model=Settings.embed_model,
                                )
                                graph_index.insert(doc)
                            else:
                                raise

                case '.pdf':
                    # Create a subdirectory to save images (using PDF base name)
                    pdf_dir = os.path.join(repo_upload_dir, temp_dir_name)
                    os.makedirs(pdf_dir, exist_ok=True)
                    images = convert_from_path(file_location)

                    # Save all images in the subdirectory
                    for i, image in enumerate(images):
                        image_save_path = os.path.join(pdf_dir, f"page_{i}.png")
                        image.save(image_save_path, "PNG")

                    for i, image in enumerate(images):
                        image_save_path = os.path.join(pdf_dir, f"page_{i}.png")
                        txt_response = process_image_with_ollama(
                            image_save_path, 
                            text_con_prompt,
                            OLLAMA_URL
                        )
                        
                        # Create specialized metadata with page number
                        page_metadata = clean_metadata(
                            source_data[0].metadata, 
                            page_number=i+1
                        )
                        
                        # Create document with cleaned metadata
                        doc = TextNode(
                            text=txt_response,
                            metadata=page_metadata
                        )
                        
                        # Robust insertion with retry
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                graph_index.insert(doc)
                                break
                            except Exception as insert_error:
                                logger.warning(f"Insertion attempt {attempt + 1} failed: {str(insert_error)}")
                                if attempt < max_retries - 1:
                                    import time
                                    time.sleep(2 ** attempt)  # Exponential backoff
                                else:
                                    logger.error(f"Failed to insert document after {max_retries} attempts")
                                    raise

                    return {"status": "success", "file": file_location}
                               
        except Exception as e:
            logger.error(f"Error in file processing: {str(e)}")
            raise

        logger.info(f"Successfully processed file: {file_location}")
        return {"status": "success", "file": file_location}

    except Exception as e:
        logger.error(f"Error processing file {file_location}: {str(e)}")
        raise  # Re-raise the exception so Celery knows the task failed