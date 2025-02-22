import json
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from config.config import OPENAI_API_KEY
from .vector_rag import VectorRAG

# model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=OPENAI_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="llama3.3:70b",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
    },
)

system_prompt = """
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
```json
{
    "instruction": "",
    "action": "",
    "phone_number": "",
    "password": ""
}
```
"""

authenticate_agent = AssistantAgent("auth_agent", model_client, system_message=system_prompt)
# rag_agent = AssistantAgent(
#     name="rag_agent",
#     model_client=model_client,
#     system_message="You are a professional and knowledgeable AI assistant powered by Retrieval-Augmented Generation (RAG). Once the user has successfully signed in or registered, please proceed to address their queries with clarity, accuracy, and promptness. Generally, please answer with get_answer function calling because you are rag assistant for local documents. However, if user ask general question, you can ask LLM",
#     tools=[get_answer]
# )
agent_team = RoundRobinGroupChat([authenticate_agent], max_turns=1)

async def run_auth_agent(user_input: str) -> dict:
    task_prompt = f"The user says: '{user_input}'.\n\n"
    response = await agent_team.run(task=task_prompt)
    print(response.messages[1].content)
    if "```json" in response.messages[1].content:
        pattern = r"```json(.*)```"
        print(">>>>>>>>>>>>>>>>>>>>>>>>")
        print(response)
        match = re.search(pattern, response.messages[1].content, re.DOTALL)
        message = match.group(1) if match else response.messages[1].content
        return json.loads(message)
    else:
        return {"instruction": response.messages[1].content, "action": "ask", "phone_number": "", "password": ""}