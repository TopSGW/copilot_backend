import os
import aiohttp
import asyncio
import datetime


llama_system_prompt = """
You are a Retrieval Augmented Generation (RAG) system designed to deliver comprehensive document analysis and question answering, with a particular emphasis on accounting and financial documents.
To ensure secure access, users must sign in. Please instruct users to sign in, and if they do not have an account, kindly guide them through the account registration process.
Step 1: Determine whether the user intends to sign-up (create a new account) or sign-in (access an existing account). 
Step 2: Request that the user provide their phone number. Since phone numbers can be entered in various formats, please convert the input into a standardized format. For example, convert "+1 235-451-1236" to "+12354511236".
Step 3: Request that the user provide their password.
Output your instructions and the collected information as a JSON string with exactly the following keys: "instruction", "action", "phone_number", and "password".
If the necessary credential information is not provided, please offer clear and courteous guidance to assist the user.
Ensure that the final output is strictly in JSON format without any additional commentary.
If user want sign in, set the json value to "sign-in". Or user want sign up, set the json value to "sign-up". 

Example output:
{
    "instruction": "",
    "action": "",
    "phone_number": "",
    "password": ""
}
"""
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

    async def aemb_text(self, text, model="llama3.3:70b"):
        # NOTE: As of now, the embeddings API is not available in Ollama's compatibility layer.
        # Future improvements might include support for embeddings.
        raise NotImplementedError("Embeddings API is not available in Ollama's OpenAI compatibility mode.")

# Example usage:
async def main():
    print("starting..")
    start_time = datetime.datetime.now()
    print("Main function started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    handler = LlamaHandler(system_prompt=llama_system_prompt)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    try:
        response = await handler.agenerate_chat_completion(messages, model="llama3.3:70b")
        print("Chat Completion:", response)
    except Exception as e:
        print("Error:", e)

    end_time = datetime.datetime.now()
    print("Main function ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Total duration:", end_time - start_time)

# To run the example, uncomment the line below:
asyncio.run(main())
