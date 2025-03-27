import os
import asyncio

import json
from fastapi import WebSocket, HTTPException, Depends
from sqlalchemy.orm import Session
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from datetime import timedelta
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.prompts import ChatMessage, ChatPromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.core.llms import ChatMessage, MessageRole, TextBlock, ImageBlock
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

import prompts
from celery_worker import process_file_for_training
from databases.database import get_db, User
from auth import get_current_user, create_access_token, authenticate_user, get_password_hash, get_user_by_phone
from config.config import ACCESS_TOKEN_EXPIRE_HOURS
from rag.rag import authenticate_agent
from rag.llama_handler import LlamaHandler
from rag.llama_handler import llama_system_prompt

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from config.config import UPLOAD_DIR
from routes.files import create_text_file, append_to_file
import datetime
from config.config import OLLAMA_URL, props_schema
from pymilvus import utility

# uri = NEO4J_HOST  # Update with your Neo4j URI

# print("BOLT URI", uri)
# admin_username = NEO4J_USER        # Admin username
# admin_password = NEO4J_PASSWORD  # Admin password

# driver = GraphDatabase.driver(uri, auth=(admin_username, admin_password))

# def database_exists(db_name):
#     """
#     Check if a database exists in the Neo4j instance.
#     """
#     with driver.session(database="system") as session:
#         result = session.run("SHOW DATABASES")
#         databases = [record["name"] for record in result]
#         return db_name in databases

# def create_database_if_not_exists(db_name):
#     """
#     Create a new database if it does not already exist.
#     """
#     if database_exists(db_name):
#         print(f"Database '{db_name}' already exists.")
#     else:
#         with driver.session(database="system") as session:
#             session.run(f"CREATE DATABASE {db_name}")
#         print(f"Database '{db_name}' created successfully.")

Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=500.0,
    base_url=OLLAMA_URL
)
Settings.embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url=OLLAMA_URL,
    request_timeout=500.0,
    ollama_additional_kwargs={"mirostat": 0},
)

auth_agent = LlamaHandler(system_prompt=llama_system_prompt)

def set_graph_space(space_name: str):
    config = Config()
    config.max_connection_pool_size = 10

    connection_pool = ConnectionPool()
    if not connection_pool.init([('127.0.0.1', 9669)], config):
        print("Failed to initialize the connection pool!")
        return

    session = connection_pool.get_session('root', 'nebula')
    
    try:
        query = f'CREATE SPACE IF NOT EXISTS {space_name}(vid_type=FIXED_STRING(256));'
        result = session.execute(query)
        print("Query executed successfully!")
        print(result)
    except Exception as e:
        print("Error executing query:", e)
    finally:
        session.release()
        connection_pool.close()

async def websocket_auth_dialogue(websocket: WebSocket):
    await websocket.accept()
    db = next(get_db())
    print("Auth WebSocket accepted")
    websocket_open = True
    try:
        while websocket_open:
            try:
                data = await websocket.receive_text()
                print("Received auth data:", data)
                auth_input_data = json.loads(data)
            except json.JSONDecodeError:
                print("Error decoding JSON data")
                await websocket.send_json({
                    "message": "Invalid JSON format",
                    "status": False
                })
                continue
            except Exception as e:
                print("Auth receive error or disconnect:", e)
                websocket_open = False
                break

            messages = auth_input_data.get("user_input", [])
            
            if not isinstance(messages, list):
                await websocket.send_json({
                    "message": "Invalid input format. Expected a list of messages.",
                    "status": False
                })
                continue
            auth_data = await auth_agent.agenerate_chat_completion(messages, model="llama3.3:70b")
            print("Auth agent returned:", auth_data)
            auth_data = json.loads(auth_data)
            action = auth_data.get("action")
            phone = auth_data.get("user_number") or auth_data.get("phone_number")
            password = auth_data.get("password")

            if action == "" or phone == "" or password == "":
                await websocket.send_json({
                    "message": auth_data.get("instruction"),
                    "status": False
                })
                continue

            token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
            if action.lower() == "sign-up":
                if get_user_by_phone(db, phone):
                    response = await authenticate_agent.on_messages(
                        [TextMessage(content="Phone number already registered. Please sign in.", source="auth_agent")],
                        cancellation_token=CancellationToken(),
                    )
                    await websocket.send_json({
                        "message": response.chat_message.content
                    })
                    continue
                hashed_pw = get_password_hash(password)
                new_user = User(phone_number=phone, hashed_password=hashed_pw)
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                token = create_access_token(
                    data={"sub": new_user.phone_number},
                    expires_delta=token_expires
                )
                msg_response = await authenticate_agent.on_messages(
                    [TextMessage(content="Sign up successful! You can start query based on your own data If you did not the data, please upload the data now.", source="auth_agent")],
                    cancellation_token=CancellationToken(),
                )
                # documents = SimpleDirectoryReader("./data/blackrock").load_data()
                # if os.path('./milvus_demo.db')
                # vector_store = MilvusVectorStore(
                #     uri="http://localhost:19530", 
                #     dim=1536, 
                #     overwrite=True, 
                #     collection_name=f"user_{new_user.id}",
                #     text_key="text",
                #     metric_type="COSINE",
                #     index_type="IVF_FLAT",
                # )
                # pipeline = IngestionPipeline(
                #     transformations=[
                #         SentenceSplitter(chunk_size=2048, chunk_overlap=32),
                #         OpenAIEmbedding(),
                #     ],
                #     vector_store=vector_store,
                # )
                # pipeline.run(documents = documents)
            elif action.lower() == "sign-in":
                user = authenticate_user(db, phone, password)
                if not user:
                    response = await authenticate_agent.on_messages(
                        [TextMessage(content="Incorrect phone number or password.", source="auth_agent")],
                        cancellation_token=CancellationToken(),
                    )
                    await websocket.send_json({
                        "message": response.chat_message.content
                    })
                    continue
                token = create_access_token(
                    data={"sub": user.phone_number},
                    expires_delta=token_expires
                )
                msg_response = await authenticate_agent.on_messages(
                    [TextMessage(content="Sign in successful", source="auth_agent")],
                    cancellation_token=CancellationToken(),
                )
            else:
                await websocket.send_json({
                    "message": auth_data.get("instruction"),
                    "status": False
                })
                continue

            print("Sending auth response with token.")
            final_msg = json.loads(msg_response.chat_message.content)
            await websocket.send_json({
                "message": final_msg.get("instruction"),
                "token": token,
                "status": True
            })
    except Exception as e:
        print("Auth endpoint exception:", e)
    finally:
        db.close()
        if websocket_open:
            await websocket.close()
        print("Auth WebSocket closed.")

async def websocket_chat(websocket: WebSocket, token: str):
    try:
        user = await get_current_user(token)
    except HTTPException as e:
        print(f"Invalid token: {e.detail}. Accepting connection to send error message.")
        await websocket.accept()
        await websocket.send_json({"error": e.detail})
        await websocket.close(code=1008)
        return

    await websocket.accept()
    print(f"Chat WebSocket accepted for user: {user.phone_number}")
    websocket_open = True
    set_graph_space(space_name=f'space_{user.id}')

    # create_database_if_not_exists(f'space_{user.id}')

    # example_documents = SimpleDirectoryReader("./data/blackrock").load_data()
    # pg_neo4j_store = Neo4jPropertyGraphStore(
    #     username=NEO4J_USER,
    #     password=NEO4J_PASSWORD,
    #     url=NEO4J_HOST,
    # )
    # pg_neo4j_index = PropertyGraphIndex.from_documents(
    #     property_graph_store=pg_neo4j_store,
    #     documents=example_documents,
    #     llm=Settings.llm,
    #     embed_model=Settings.embed_model,
    # )
    # neo4j_chat_engine = pg_neo4j_index.as_chat_engine(
    #     chat_mode="context",
    #     llm=Settings.llm
    # )
    property_graph_store = NebulaPropertyGraphStore(
        space=f'space_{user.id}',
        props_schema=props_schema
    )
    index_config = {
        "index_type": "HNSW",
        "params": {
            "M": 8,
            "efConstruction": 64,
        },
        "metric_type": "COSINE"
    }
    graph_vec_store = MilvusVectorStore(
        uri="http://localhost:19530", 
        collection_name=f"space_{user.id}",
        dim=1024,
        overwrite=False,
        index_config=index_config,
        similarity_metric="COSINE",
    )

    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=property_graph_store,
        vector_store=graph_vec_store,
        llm=Settings.llm,
    )

    utility.load_collection(f"space_{user.id}")

    query_engine = graph_index.as_query_engine(llm=Settings.llm)

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="name",
        description="Query graph database based on user query",
        return_direct=False
    )
    graph_chat_engine = graph_index.as_chat_engine(
        chat_mode='context',
        llm=Settings.llm,
        system_prompt=prompts.RAG_SYSTEM_PROMPT,
    )
    graph_query_engine = graph_index.as_query_engine(
        llm=Settings.llm
    )
    # milvus_manager = MilvusManager(
    #     milvus_uri="./milvus_original.db",
    #     collection_name=f"original_{user.id}"    
    # )
    # milvus_manager.create_index()
    repo_upload_dir = os.path.join(UPLOAD_DIR, user.phone_number, "note")
    os.makedirs(repo_upload_dir, exist_ok=True)

    note_path = os.path.join(repo_upload_dir, "note.txt")
    create_text_file(note_path)

    def append_save_to_file(content: str):
        """
        Appends the provided content to the file at file_path.
        If the file does not exist, it will be created automatically.
        
        Args:
            file_path: Path to the file where content should be appended
            content: The formatted string to append to the file
        
        Returns:
            None
        """
        print("Note text training started...")
        time_content = datetime.datetime.now()
        input_content = f"[{time_content}]\n{content}"
        print(f">>>>>>>>content<<<<<<< {input_content}")
        file_path = note_path
        try:
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(input_content + "\n")
                
            # Process the file asynchronously
            process_file_for_training.delay(file_path, user.id, 1)
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"status": "error", "message": f"Failed to save: {str(e)}"}
        else:
            print(f"Content appended to file at: {file_path}")
            return {"status": "success", "message": "Content saved successfully"}

    # add_data_agent = FunctionAgent(
    #     name="add_data_agent",
    #     description="Specialized agent for explicit data saving operations.",
    #     system_prompt=(
    #         "# Data Saving Agent\n\n"
    #         "You are a specialized agent focused exclusively on explicitly requested save operations. "
    #         "Respond concisely and use tools only when the user clearly instructs to save information.\n\n"

    #         "## Core Responsibilities\n"
    #         "1. Save user-provided information upon explicit request.\n"
    #         "2. Save retrieved information explicitly requested by the user.\n\n"

    #         "## Explicit Save Request Identification\n"
    #         "- Save only if user explicitly instructs ('save this', 'remember', 'add to notes').\n"
    #         "- Do NOT save if user simply asks for file location, file names, or general queries.\n"

    #         "## Save Data Format (ALWAYS USE THIS)\n"
    #         "[topic]\n"
    #         "[detailed information]\n\n"

    #         "## Quick Decision Tree\n"
    #         "- Explicit save request → Extract info → Generate topic → Save\n"
    #         "- Retrieval + save request ('do you know X? save it') → Query → Format → Save\n"
    #         "- All other queries → Hand off immediately to query_agent\n\n"

    #         "## Response Templates\n"
    #         "- Success: 'Saved: [topic]'\n"
    #         "- Failure: 'Unable to save: [reason]'\n"
    #         "- Handoff: 'Let me get you that information.'\n\n"

    #         "## Efficiency Rules\n"
    #         "- Do not attempt implicit saves.\n"
    #         "- Respond in one concise sentence.\n"
    #         f"- Always use {note_path} as the file path.\n"
    #         "- Hand off non-save queries immediately.\n"
    #         "- Never ask clarifying questions."
    #     ),
    #     tools=[append_save_to_file, query_engine_tool],
    #     can_handoff_to=["query_agent"],
    #     llm=Settings.llm,
    # )
    # # Define a prompt for RAG operations specifically
    # query_agent = FunctionAgent(
    #     name="query_agent",
    #     description="Specialized agent for fast information retrieval and query processing.",
    #     system_prompt=prompts.RAG_SYSTEM_PROMPT,
    #     tools=[query_engine_tool],
    #     can_handoff_to=[],
    #     llm=Settings.llm,
    # )

    # # Configure the workflow with optimized settings
    # agent_workflow = AgentWorkflow(
    #     agents=[add_data_agent, query_agent],
    #     root_agent="add_data_agent",
    #     # Define a custom handoff prompt that's more concise
    #     handoff_prompt="""Useful for handing off to another agent.
    # Hand off to the appropriate specialized agent when needed:

    # {agent_info}

    # Hand off immediately without explanation or thinking steps.
    # """,
    #     # Use a minimal state prompt to reduce token usage
    #     state_prompt="""State: {state}
    # Query: {msg}""",
    #     # Configure with a timeout to prevent hanging
    # )
    unified_agent = FunctionAgent(
        name="unified_agent",
        description="Agent handling both explicit data-saving requests and general queries.",
        system_prompt=(
            "# Unified Agent\n\n"
            "You handle both explicit save operations and general queries. "
            "Respond concisely and choose actions quickly.\n\n"
            "## Save Operations (explicit requests only)\n"
            "- Trigger on clear phrases like ('save', 'remember', 'note').\n"
            "- Always save information using provided save tool.\n\n"
            "## General Queries\n"
            "- If no explicit save instruction is present, quickly query using the query engine tool.\n\n"
            "## Response Rules\n"
            "- Explicit save → Save tool.\n"
            "- General query → Query engine tool directly.\n"
            "- Respond concisely without additional explanation."
        ),
        tools=[append_save_to_file, query_engine_tool],
        llm=Settings.llm,
    )
    agent_workflow = AgentWorkflow(
        agents=[unified_agent]    
    )

    while websocket_open:
        try:
            data = await websocket.receive_text()
            print("Received data:", data)
        except Exception as e:
            print("Receive error or disconnect:", e)
            websocket_open = False
            break

        try:
            start_time = datetime.datetime.now()
            print("Main function started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
            auth_input_data = json.loads(data)
            messages = auth_input_data.get("user_input", [])
            if not isinstance(messages, list):
                await websocket.send_json({
                    "message": "Invalid input format. Expected a list of messages.",
                    "status": False
                })
                continue
            
            chat_history = []
            for message in messages:
                chat_history.append(ChatMessage(content=message.get("content"), role=message.get("role")))
            # query_vec = colpali_manager.process_text([user_input])[0]

            # search_res = milvus_manager.search(query_vec, topk=5)
            # docs = [doc for score, _ , doc in search_res]
            # print("docs", docs)
            # print("Processing user input:", user_input)

            # conversation = [
            #     {
            #         "role": "<|User|>",
            #         "content": (
            #             user_input
            #         ),
            #         "images": docs,
            #     },
            #     {"role": "<|Assistant|>", "content": ""}
            # ]

            # pil_images = DeepSeekpipeline.load_images(conversation)

            # prepared_inputs = DeepSeekpipeline.prepare_inputs(conversation, pil_images, system_prompt=prompts.RAG_SYSTEM_PROMPT)

            # vec_answer = DeepSeekpipeline.generate_response(prepared_inputs)

            # print(f"{prepared_inputs['sft_format'][0]}\n{vec_answer}")
            
            # SYSTEM_PROMPT = """
            # Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
            # """.strip()

            response = await agent_workflow.run(chat_history=chat_history)
            print(response)
            final_answer = str(response)
            

            end_time = datetime.datetime.now()
            print("Main function ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("Total duration:", end_time - start_time)


            start_time = datetime.datetime.now()
            print("Graph chat engine function started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

            if chat_history:
                last_message = messages[-1]
                print("Last message:", last_message)
            else:
                print("No messages in chat_history")
            rest_messages = chat_history[:-1]
            graph_chat_answer = await graph_chat_engine.achat(message=last_message.get("content"), chat_history=rest_messages)
            print("Graph chat answer:", str(graph_chat_answer.response))

            end_time = datetime.datetime.now()
            print("Graph chat engine function ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("Total duration:", end_time - start_time)

            start_time = datetime.datetime.now()
            print("Graph query engine function started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

            graph_query_response = await graph_query_engine.aquery(last_message.get("content"))
            print("Graph query response:", str(graph_query_response))

            end_time = datetime.datetime.now()
            print("Graph chat engine function ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
            print("Total duration:", end_time - start_time)
            # Build the user prompt by combining vector_answer and graph_response into the <context> block,
            # and including the user_input within the <question> block.
            # USER_PROMPT = f"""
            # Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
            # <context>
            # {vec_answer}
            # {str(graph_response)}
            # </context>
            # <question>
            # {user_input}
            # </question>
            # """.strip()

            # Create the chat messages using the helper method.
            # messages = [
            #     ChatMessage.from_str(SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            #     ChatMessage.from_str(USER_PROMPT, role=MessageRole.USER),
            # ]

            # final_answer = Settings.llm.chat(messages=messages)
            print(final_answer)
            if str(final_answer) == 'Empty Response':
                await websocket.send_json({"message": "There is no provided documents. Please upload documents."})
            else:    
                await websocket.send_json({"message": final_answer})
        except json.JSONDecodeError:
            print("Error decoding JSON data")
            await websocket.send_json({"message": "Invalid JSON format"})
        except Exception as e:
            print("Error processing message:", e)
            await websocket.send_json({"message": "An error occurred processing your request."})
    print("Closing chat WebSocket.")
    if websocket_open:
        await websocket.close()