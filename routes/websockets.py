import os
import json
from fastapi import WebSocket, HTTPException, Depends
from sqlalchemy.orm import Session
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from datetime import timedelta
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

import prompts


from databases.database import get_db, User
from auth import get_current_user, create_access_token, authenticate_user, get_password_hash, get_user_by_phone
from config.config import ACCESS_TOKEN_EXPIRE_HOURS
from rag.rag import run_auth_agent, authenticate_agent
from rag.vector_rag import VectorRAG

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

def set_graph_space(space_name: str):
    config = Config()
    config.max_connection_pool_size = 10

    connection_pool = ConnectionPool()
    if not connection_pool.init([('127.0.0.1', 9669)], config):
        print("Failed to initialize the connection pool!")
        return

    # Create a session with the Nebula Graph server
    session = connection_pool.get_session('root', 'nebula')
    
    try:
        # Define your nGQL command
        query = f'CREATE SPACE IF NOT EXISTS {space_name}(vid_type=FIXED_STRING(256));'
        # Execute the command
        result = session.execute(query)
        print("Query executed successfully!")
        print(result)
    except Exception as e:
        print("Error executing query:", e)
    finally:
        # Always release the session and close the connection pool
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
            except Exception as e:
                print("Auth receive error or disconnect:", e)
                websocket_open = False
                break

            auth_input_data = json.loads(data)
            user_input = auth_input_data.get("user_input", "")
            auth_data = await run_auth_agent(user_input)
            print("Auth agent returned:", auth_data)
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
                    await websocket.send_json({
                        "error": "Phone number already registered. Please sign in."
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
                    await websocket.send_json({
                        "error": "Incorrect phone number or password."
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
    documents = SimpleDirectoryReader("./data/blackrock").load_data()
    vector_store = MilvusVectorStore(
        uri="./milvus_demo.db", 
        collection_name=f"user_{user.id}",
        dim=1536, 
        overwrite=False,         
        metric_type="COSINE",
        index_type="IVF_FLAT",
    )
    set_graph_space(space_name=f'space_{user.id}')
    # pipeline = IngestionPipeline(
    #     transformations=[
    #         SentenceSplitter(chunk_size=2048, chunk_overlap=32),
    #         OpenAIEmbedding(),
    #     ],
    #     vector_store=vector_store,
    # )
    # pipeline.run(documents=documents)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    chat_engine = vector_index.as_chat_engine(
        chat_mode='context',
        memory=memory,
        system_prompt=prompts.RAG_SYSTEM_PROMPT,
        llm=Settings.llm
    )

    property_graph_store = NebulaPropertyGraphStore(
        space=f'space_{user.id}'
    )
    graph_vec_store = SimpleVectorStore.from_persist_path("./storage_graph/nebula_vec_store.json")

    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=property_graph_store,
        vector_store=graph_vec_store,
        llm=Settings._llm,
    )

    graph_chat_engine = graph_index.as_chat_engine(
        chat_mode='context',
        llm=Settings._llm,
        system_prompt=prompts.RAG_SYSTEM_PROMPT,
        memory=memory
    )
    # vector_rag = VectorRAG(db_path="./milvus_demo.db", collection_name=f"user_{user.id}")
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
            print("Processing user input:", user_input)
            response = chat_engine.chat(message=user_input)
            print("Response from vector_rag:", response)
            graph_response = graph_chat_engine.chat(message=user_input)
            print("Response from graph_rag:", response)
            await websocket.send_json({"message": str(response)})
        except Exception as e:
            print("Error processing message:", e)
            await websocket.send_json({"message": "An error occurred processing your request."})
    print("Closing chat WebSocket.")
    if websocket_open:
        await websocket.close()