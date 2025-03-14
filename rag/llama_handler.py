import os
import aiohttp
import asyncio

class LlamaHandler:
    def __init__(self, system_prompt: str = "", base_url: str = "http://localhost:11434/v1", api_key: str = None):
        # For Ollama compatibility, the API key is required but not actually used;
        # we default to "ollama" if not provided.
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
        self.base_url = base_url
        self.system_prompt = system_prompt

    async def agenerate_chat_completion(self, messages, model="llama3.3:70b"):
        # If a system prompt is provided, prepend it as a system message.
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            # The Authorization header is required by the OpenAI libraries,
            # though for Ollama it's unused.
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": messages
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Error {response.status}: {text}")
                data = await response.json()
                # The response structure is assumed to follow OpenAI's format.
                return data["choices"][0]["message"]["content"]

    async def aemb_text(self, text, model="llama2"):
        # NOTE: As of now, the embeddings API is not available in Ollama's compatibility layer.
        # Future improvements might include support for embeddings.
        raise NotImplementedError("Embeddings API is not available in Ollama's OpenAI compatibility mode.")

# Example usage:
async def main():
    handler = LlamaHandler(system_prompt="You are a helpful assistant.")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    try:
        response = await handler.agenerate_chat_completion(messages, model="llama2")
        print("Chat Completion:", response)
    except Exception as e:
        print("Error:", e)

# To run the example, uncomment the line below:
asyncio.run(main())
