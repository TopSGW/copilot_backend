import os
import aiohttp
import asyncio

class LlamaHandler:
    def __init__(self, system_prompt: str = "", base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.system_prompt = system_prompt

    async def agenerate_chat_completion(self, messages, model="llama3.3:70b"):
        # Prepend the system prompt if provided.
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        # The Ollama API chat endpoint.
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                # Assuming response structure: {"choices": [{"message": {"content": "text"}}]}
                return data["choices"][0]["message"]["content"]

    async def aemb_text(self, text, model="llama3.3"):
        # NOTE: Ensure your Ollama installation supports an embeddings endpoint.
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "input": text
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                # Assuming response structure: {"data": [{"embedding": [...] }]}
                return data["data"][0]["embedding"]

# Example usage:
async def main():
    handler = LlamaHandler(system_prompt="You are a helpful assistant.")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    chat_response = await handler.agenerate_chat_completion(messages)
    print("Chat Completion:", chat_response)
    
    # Example for embeddings (if supported)
    embedding = await handler.aemb_text("This is a sample sentence for embeddings.")
    print("Embedding:", embedding)

# To run the example, uncomment the following line:
# asyncio.run(main())
