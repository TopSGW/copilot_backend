import os
import nest_asyncio
nest_asyncio.apply()

import json
from fastapi import WebSocket, HTTPException, Depends
from sqlalchemy.orm import Session
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from datetime import timedelta
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.core.llms import ChatMessage, MessageRole, TextBlock, ImageBlock

import prompts

from databases.database import get_db, User
from auth import get_current_user, create_access_token, authenticate_user, get_password_hash, get_user_by_phone
from config.config import ACCESS_TOKEN_EXPIRE_HOURS
from rag.rag import run_auth_agent, authenticate_agent
from rag.llama_handler import LlamaHandler
from rag.llama_handler import llama_system_prompt

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from config.config import UPLOAD_DIR
from routes.files import create_text_file, append_to_file
from rag.llama_handler import action_agent_prompt
import ollama

Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
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
                    [TextMessage(content="Sign up successful!", source="auth_agent")],
                    cancellation_token=CancellationToken(),
                )
                # documents = SimpleDirectoryReader("./data/blackrock").load_data()
                # if os.path('./milvus_demo.db')
                # vector_store = MilvusVectorStore(
                #     uri="./milvus_demo.db", 
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
            await websocket.send_json({
                "message": msg_response.chat_message.content,
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
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    property_graph_store = NebulaPropertyGraphStore(
        space=f'space_{user.id}'
    )
    graph_vec_store = MilvusVectorStore(
        uri="./milvus_graph.db", 
        collection_name=f"space_{user.id}",
        dim=8192, 
        overwrite=False,         
        metric_type="COSINE",
        index_type="IVF_FLAT",
    )

    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=property_graph_store,
        vector_store=graph_vec_store,
        llm=Settings.llm,
    )

    graph_chat_engine = graph_index.as_chat_engine(
        chat_mode='context',
        llm=Settings.llm,
        system_prompt=prompts.RAG_SYSTEM_PROMPT,
        memory=memory
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
    while websocket_open:
        try:
            data = await websocket.receive_text()
            print("Received data:", data)
        except Exception as e:
            print("Receive error or disconnect:", e)
            websocket_open = False
            break

        try:
            auth_input_data = json.loads(data)
            user_input = auth_input_data.get("user_input", "")
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
            determine_agent = LlamaHandler(system_prompt=action_agent_prompt)
            messages = [{"role": "user", "content": user_input}]
            determine_output = await determine_agent.agenerate_chat_completion(messages=messages, model="llama:3.3:70b")
            determine_output = json.loads(determine_output)
            print(f"actions >>> {determine_output.get("actions")}")
            search_content = determine_output.get("search_content")
            print(f"search_content >>>> {search_content}")
            save_content = determine_output.get("save_content")
            print(f"save content >>>>> {save_content}")
            greet_message = determine_output.get("greet_message")
            print(f"greet message >>>> {greet_message}")

            
            if search_content != "":
                final_answer = await graph_chat_engine.achat(message=user_input)
                print("Response from graph_store:", final_answer)
            
            if save_content !="":
                append_to_file(file_path=note_path, content=user_input)
                final_answer = "The information is successfully saved!"
            
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
            final_answer = greet_message + "\n" + final_answer
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