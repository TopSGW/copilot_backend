import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

print("staring ...")
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
{
    "instruction": "",
    "action": "",
    "phone_number": "",
    "password": ""
}
"""
authenticate_agent = AssistantAgent(
    name="auth_agent",
    model_client=model_client,
    system_message=system_prompt,
)

async def run_auth_agent(user_input: str) -> dict:
    response = await authenticate_agent.on_messages(
        [TextMessage(content=user_input, source="user")],
        cancellation_token=CancellationToken(),
    )    
    print(response.chat_message.content)
    return json.loads(response.chat_message.content)
    # if "```json" in response.messages[1].content:
    #     pattern = r"```json(.*)```"
    #     match = re.search(pattern, response.messages[1].content, re.DOTALL)
    #     message = match.group(1) if match else response.messages[1].content
    #     return json.loads(message)
    # else:
    #     return {"instruction": response.messages[1].content, "action": "ask", "phone_number": "", "password": ""}

# async def main():
#     # Test input for sign-up
#     test_input_signup = "I want to create a new account"
#     result_signup = await run_auth_agent(test_input_signup)
#     print("Sign-up test result:", result_signup)

#     # Test input for sign-in
#     test_input_signin = "I want to sign in to my existing account"
#     result_signin = await run_auth_agent(test_input_signin)
#     print("Sign-in test result:", result_signin)

# if __name__ == "__main__":
#     asyncio.run(main())