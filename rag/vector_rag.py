import os

from dotenv import load_dotenv
import ell
from openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from pymilvus import MilvusClient, Collection, connections

import json
import prompts

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SEED = 42

# Configure Ollama embedding model
embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Configure OpenAI client for Ollama
llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real API key
)

# Register the model with Ellama
MODEL = "llama3.3:70b"
ell.config.register_model(MODEL, llm_client)


class VectorRAG:
    def __init__(self, db_path: str, collection_name: str = "llamacollection"):
        load_dotenv()
        self.embed_model = embed_model
        self.db = MilvusClient(db_path)
        self.collection_name = collection_name

    def query(self, query_vector: list, limit: int = 10) -> list:
        search_res = self.db.search(
            collection_name=self.collection_name,
            data=[
                query_vector
            ],
            limit=limit,
            output_fields=["_node_content"]
        )
        retrieved_lines_with_distances = [
           (json.loads(res["entity"]['_node_content'])['text'], res["distance"]) for res in search_res[0]
        ]
        context_list = [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]

        return context_list if context_list else None

    def embed(self, query: str) -> list:
        # Using Ollama embedding model
        embedding = self.embed_model.get_text_embedding(query)
        return embedding

    @ell.simple(model=MODEL, temperature=0.3)
    def retrieve(self, question: str, context: str) -> str:
        return [
            ell.system(prompts.RAG_SYSTEM_PROMPT),
            ell.user(prompts.RAG_USER_PROMPT.format(question=question, context=context)),
        ]

    def run(self, question: str) -> str:
        question_embedding = self.embed(question)
        context = self.query(question_embedding)
        return self.retrieve(question, context)


if __name__ == "__main__":
    vector_rag = VectorRAG("./milvus_demo.db")
    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    response = vector_rag.run(question)
    print(f"Q1: {question}\n\n{response}")

    question = "Where did Larry Fink graduate from?"
    response = vector_rag.run(question)
    print(f"---\nQ2: {question}\n\n{response}")

    question = "When was Susan Wagner born?"
    response = vector_rag.run(question)
    print(f"---\nQ3: {question}\n\n{response}")

    question = "How did Larry Fink and Rob Kapito meet?"
    response = vector_rag.run(question)
    print(f"---\nQ4: {question}\n\n{response}")