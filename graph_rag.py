import os
import logging
import sys
from dotenv import load_dotenv

from llama_index.core.storage import StorageContext
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.memory import ChatMemoryBuffer
import nest_asyncio

nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class GraphRAG:
    def __init__(self):
        load_dotenv()
        
        # Load Nebula environment variables
        os.environ['NEBULA_USER'] = os.getenv('NEBULA_USER')
        os.environ['NEBULA_PASSWORD'] = os.getenv('NEBULA_PASSWORD')
        os.environ['NEBULA_ADDRESS'] = os.getenv('NEBULA_ADDRESS')
        
        self.setup_llm()
        self.setup_graph_index()
        self.memory = None
        self.chat_engine = None

    def setup_llm(self):
        """Configure LLM and embedding settings"""
        Settings.llm = Ollama(
            model="llama3.3:70b",
            temperature=0.3,
            request_timeout=120.0,
            base_url="http://localhost:11434"
        )

        Settings.embed_model = OllamaEmbedding(
            model_name="llama3.3:70b",
            base_url="http://localhost:11434",
        )

    def setup_graph_index(self):
        """Initialize the graph index with vector store and property graph store"""
        vec_store = SimpleVectorStore.from_persist_path("./storage_graph/nebula_vec_store.json")
        
        property_graph_store = NebulaPropertyGraphStore(
            space="llamaindex_nebula_property_graph",
        )

        self.graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=property_graph_store,
            vector_store=vec_store,
            llm=Settings.llm,
            show_progress=True
        )

    def create_query_engine(self):
        """Create and return a query engine"""
        return self.graph_index.as_query_engine(
            llm=Settings.llm,
            include_text=True
        )

    def create_chat_engine(self):
        """Create and configure the chat engine"""
        if not self.memory:
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=1500,
                llm=Settings.llm
            )

        self.chat_engine = self.graph_index.as_chat_engine(
            chat_mode="context",
            memory=self.memory,
            verbose=True,
            llm=Settings.llm
        )

    def query(self, question: str) -> str:
        """Execute a single query"""
        query_engine = self.create_query_engine()
        response = query_engine.query(question)
        return str(response)

    def chat(self, message: str) -> str:
        """Handle chat interactions"""
        if not self.chat_engine:
            self.create_chat_engine()
        response = self.chat_engine.chat(message)
        return str(response)

    def run(self, questions: list[str], mode: str = "query") -> list[str]:
        """
        Run multiple questions in either query or chat mode
        mode: "query" or "chat"
        """
        responses = []
        for question in questions:
            if mode == "chat":
                response = self.chat(question)
            else:
                response = self.query(question)
            responses.append(f"Q: {question}\nA: {response}")
        return responses


if __name__ == "__main__":
    # Initialize the GraphRAG
    graph_rag = GraphRAG()
    
    # Example queries
    questions = [
        "Who are the founders of BlackRock?",
        "How did Larry Fink and Rob Kapito meet?"
    ]
    
    # Run in query mode
    print("Query Mode Results:")
    results = graph_rag.run(questions, mode="query")
    for result in results:
        print(f"\n{result}\n---")
    
    # Run in chat mode
    print("\nChat Mode Results:")
    chat_questions = [
        "who is Larry?",
        "who is Robert?",
        "who is Susan?",
        "How did Larry Fink and Rob Kapito meet?"
    ]
    results = graph_rag.run(chat_questions, mode="chat")
    for result in results:
        print(f"\n{result}\n---")