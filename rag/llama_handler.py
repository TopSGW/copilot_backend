import os
import aiohttp
from config.config import OLLAMA_URL
import re

llama_system_prompt = """
You are a Retrieval Augmented Generation (RAG) system designed to deliver comprehensive document analysis and question answering, with a particular emphasis on accounting and financial documents.
To ensure secure access, users must sign in. Please instruct users to sign in, and if they do not have an account, kindly guide them through the account registration process.

Follow these steps exactly:

Step 1: Clearly determine if the user wants to "sign-up" (create a new account) or "sign-in" (access an existing account).

Step 2: Request the user's phone number. Since phone numbers can be entered in various formats, please standardize the input by removing spaces, dashes, parentheses, and other special characters (for example, "+1 235-451-1236" becomes "+12354511236"). Ensure that the resulting phone number contains only numbers and a leading '+' with no other special characters.

Step 3: Request the user's password.

Important JSON Formatting Instructions:
- Produce your output strictly as a JSON string without any extra commentary, whitespace, or newline characters.
- Ensure JSON validity at all times.

Use exactly the following keys for your JSON response:

{
    "instruction": "",   # Your polite instruction to the user
    "action": "",        # Either "sign-in" or "sign-up"
    "phone_number": "",  # The standardized phone number
    "password": ""       # User-provided password
}

Example output:
{"instruction": "Please sign in with your credentials.", "action": "sign-in", "phone_number": "+12354511236", "password": "userpassword123"}
"""
action_agent_prompt = """
You are an AI agent that categorizes user queries into actions: searching, saving, greeting, or any combination thereof.

Analyze the user's entire query exactly as provided and respond strictly in JSON with this structure:

{
  "actions": ["search", "save", "greet"],
  "search_content": "The entire user query if search applies, otherwise empty string",
  "save_content": "The entire user query if save applies, otherwise empty string",
  "greet_message": "Greeting response if greet applies, otherwise empty string"
}

Guidelines:
- Populate "actions" with any combination of "search", "save", and/or "greet".
- Always use the entire user query exactly as provided for "search_content" and "save_content" if the corresponding action applies.
- Provide an appropriate greeting message only if the query includes a greeting; otherwise, set "greet_message" to "".
- Always respond solely in valid JSON without extra commentary.
"""

class LlamaHandler:
    def __init__(self, system_prompt: str = "", base_url: str = f"{OLLAMA_URL}/v1", api_key: str = None):
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

    async def aemb_text(self, text, model="llava:13b"):
        # NOTE: As of now, the embeddings API is not available in Ollama's compatibility layer.
        # Future improvements might include support for embeddings.
        raise NotImplementedError("Embeddings API is not available in Ollama's OpenAI compatibility mode.")

# Example usage:
# async def main():
#     print("starting..")
#     start_time = datetime.datetime.now()
#     print("Main function started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

#     handler = LlamaHandler(system_prompt=action_agent_prompt)
#     messages = [{"role": "user", "content": "Hi there. Who is Abraham? If you know, please save this information."}]
    
#     try:
#         response = await handler.agenerate_chat_completion(messages, model="llama3.3:70b")
#         print("Chat Completion:", response)
#     except Exception as e:
#         print("Error:", e)

#     end_time = datetime.datetime.now()
#     print("Main function ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
#     print("Total duration:", end_time - start_time)

# To run the example, uncomment the line below:
# asyncio.run(main())
