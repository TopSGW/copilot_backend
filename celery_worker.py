from celery import Celery
import os
import logging
import asyncio
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from pdf2image import convert_from_path
import ollama
import nest_asyncio
from config.config import OLLAMA_URL

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a stream handler (console output)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

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

# Create a custom insert function that doesn't use asyncio.run()
def sync_insert_document(graph_index, document):
    """Insert a document without using asyncio.run() internally"""
    # Create nodes from the document
    nodes = graph_index.node_parser.get_nodes_from_documents([document])
    
    # Insert the nodes using a method that doesn't call asyncio.run() internally
    try:
        # Use the property graph store directly
        graph_index._insert_nodes_to_graph_store(nodes)
        
        # Add to vector store if available
        if graph_index._vector_store is not None:
            graph_index._vector_store.add([node.node_id for node in nodes], 
                                         [graph_index._embed_model.get_text_embedding(node.get_content()) for node in nodes])
    except Exception as e:
        logger.error(f"Error in sync_insert_document: {str(e)}")
        raise

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
        
        # Process based on file type
        match file_extension: 
            case '.txt':
                logger.info("Starting text file processing............")
                simple_doc = SimpleDirectoryReader(input_files=[file_location]).load_data()
                logger.info("Simple doc has been loaded")
                
                for doc in simple_doc:
                    # Use our custom insert function instead of graph_index.insert()
                    sync_insert_document(graph_index, doc)
                
                logger.info(f"{file_location} Text file processed successfully")

            case '.jpg' | '.png' | '.jpeg':
                txt_response = ollama.chat(
                    model='llama3.2-vision:90b',
                    messages=[{
                        'role': 'user',
                        'content': text_con_prompt,
                        'images': [file_location]
                    }]
                )
                logger.info(f"Text response received from vision model")
                txt_file_location = os.path.join(repo_upload_dir, os.path.splitext(filename)[0] + ".txt")

                with open(txt_file_location, "w") as image_file:
                    image_file.write(str(txt_response.message.content))

                simple_doc = SimpleDirectoryReader(input_files=[txt_file_location]).load_data()
                
                for doc in simple_doc: 
                    logger.info("Setting metadata and inserting document")
                    doc.metadata = source_data[0].metadata
                    sync_insert_document(graph_index, doc)

            case '.pdf':
                # Create a subdirectory to save images (using PDF base name)
                pdf_dir = os.path.join(repo_upload_dir, temp_dir_name)
                os.makedirs(pdf_dir, exist_ok=True)
                images = convert_from_path(file_location)
                
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
                    
                    with open(pdf_txt_file, "a") as f:
                        f.write(f"--- Response for page {i} ---\n")
                        f.write(txt_response.message.content)
                        f.write("\n\n")
                
                # Load the combined text file for indexing
                simple_doc = SimpleDirectoryReader(input_files=[pdf_txt_file]).load_data()
                for doc in simple_doc: 
                    doc.metadata = source_data[0].metadata
                    sync_insert_document(graph_index, doc)

        logger.info(f"Successfully processed file: {file_location}")
        return {"status": "success", "file": file_location}

    except Exception as e:
        logger.error(f"Error processing file {file_location}: {str(e)}")
        raise  # Re-raise the exception so Celery knows the task failed