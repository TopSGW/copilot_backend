import os

import cohere
from dotenv import load_dotenv
import ell
from openai import OpenAI

import prompts
from graph_rag import GraphRAG
from vector_rag import VectorRAG

load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# Configure OpenAI client for Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real API key
)

# Register the model with Ellama
MODEL = "llama3.3:70b"
ell.config.register_model(MODEL, client)

SEED = 42

class HybridRAG:
    def __init__(self):
        self.graph_rag = GraphRAG()
        self.vector_rag = VectorRAG("./lancedb")
        self.co = cohere.ClientV2(COHERE_API_KEY)

    @ell.simple(model=MODEL, temperature=0.3)
    def hybrid_rag(self, question: str, context: str) -> str:
        return [
            ell.system(prompts.RAG_SYSTEM_PROMPT),
            ell.user(prompts.RAG_USER_PROMPT.format(question=question, context=context)),
        ]
    
    def run(self, question: str) -> str:
        question_embedding = self.vector_rag.embed(question)
        vector_docs = self.vector_rag.query(question_embedding)
        vector_docs = [doc["text"] for doc in vector_docs]

        graph_doc = self.graph_rag.run(questions=[question], mode='chat')

        docs = graph_doc + vector_docs

        docs = [str(doc) for doc in docs]

        combined_context = self.co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=docs,
            top_n=20,
            return_documents=True,
        )
        return self.hybrid_rag(question, combined_context)

if __name__ == "__main__":
    hybrid_rag = HybridRAG()

    question = "Who are the founders of BlackRock? Return the names as a numbered list."
    response = hybrid_rag.run(question)
    print(f"Q1: {question}\n\n{response}")

    question = "Where did Larry Fink graduate from?"
    response = hybrid_rag.run(question)
    print(f"---\nQ2: {question}\n\n{response}")

    question = "When was Susan Wagner born?"
    response = hybrid_rag.run(question)
    print(f"---\nQ3: {question}\n\n{response}")

    question = "How did Larry Fink and Rob Kapito meet?"
    response = hybrid_rag.run(question)
    print(f"---\nQ4: {question}\n\n{response}")

