# import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_astradb import AstraDBVectorStore
# from dotenv import load_dotenv

# from postgres_history import PostgresChatMessageHistory

# load_dotenv()

# # Env vars
# ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
# ASTRA_DB_COLLECTION = "medical_rag_chunks"

# # Retriever setup
# embedding = OpenAIEmbeddings()
# vectorstore = AstraDBVectorStore(
#     embedding=embedding,
#     collection_name=ASTRA_DB_COLLECTION,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     namespace=ASTRA_DB_KEYSPACE,
# )
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# # Prompt
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful medical assistant. Use the following context:\n{context}\n"),
#     ("human", "{question}")
# ])


# # Retriever callable to get context string
# def retrieve_docs(input_dict):
#     question = input_dict["question"]
#     docs = retriever.invoke(question)
#     context_text = "\n".join(doc.page_content for doc in docs)
#     return {"context": context_text, "question": question}

# # RAG chain
# rag_chain = (
#     retrieve_docs
#     | prompt
#     | ChatOpenAI(model="gpt-4o")
# )

# # Memory handler
# def get_memory(session_id: str):
#     return PostgresChatMessageHistory(session_id)

# # Wrap with memory
# chat_with_memory = RunnableWithMessageHistory(
#     rag_chain,
#     get_memory,
#     input_messages_key="question",
#     history_messages_key="history",
# )

# # Main interaction loop
# if __name__ == "__main__":
#     session_id = "user-1"
#     while True:
#         user_input = input("\n🧠 You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         response = chat_with_memory.invoke(
#             {"question": user_input},
#             config={"configurable": {"session_id": session_id}},
#         )
#         print(f"✅ Assistant: {response.content}")


import os
import time
import logging
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

from postgres_history import PostgresChatMessageHistory

# Load environment variables
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Helper to Load Environment Variables Safely ---
def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{name}' is not set.")
    return value


# --- Load Env Vars ---
ASTRA_DB_API_ENDPOINT = get_env_var("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = get_env_var("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = get_env_var("ASTRA_DB_KEYSPACE")
ASTRA_DB_COLLECTION = "medical_rag_chunks"


# --- Embeddings & Vectorstore Setup ---
embedding = OpenAIEmbeddings()
vectorstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Use the following context:\n{context}\n"),
    ("human", "{question}")
])


# --- Retrieval Function for Context ---
def retrieve_docs(input_dict):
    question = input_dict["question"]
    docs = retriever.invoke(question)
    context_text = "\n".join(doc.page_content for doc in docs)

    # Logging retrieved context
    logging.info(f"User Question: {question}")
    logging.info(f"Retrieved Context:\n{context_text}")

    return {"context": context_text, "question": question}


# --- RAG Chain with Model ---
def get_rag_chain():
    return retrieve_docs | prompt | ChatOpenAI(model="gpt-4o", streaming=False)


# --- PostgreSQL Chat History Handler ---
def get_memory(session_id: str):
    return PostgresChatMessageHistory(session_id)


# --- Runnable Chain with Memory ---
chat_with_memory = RunnableWithMessageHistory(
    get_rag_chain(),
    get_memory,
    input_messages_key="question",
    history_messages_key="history",
)


# --- CLI Interaction Loop ---
if __name__ == "__main__":
    session_id = "user-1"
    print("🤖 Medical Chatbot with Memory (type 'exit' or 'quit' to end)\n")

    while True:
        try:
            user_input = input("🧠 You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Session ended.")
                break

            start = time.time()

            # Invoke RAG chain
            response = chat_with_memory.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            duration = time.time() - start
            print(f"✅ Assistant: {response.content}")
            print(f"⏱️ Response time: {duration:.2f} seconds\n")

            # Optional: save session to file
            with open(f"session_{session_id}.txt", "a", encoding="utf-8") as f:
                f.write(f"\nUser: {user_input}\nAssistant: {response.content}\n")

        except Exception as e:
            logging.exception("⚠️ An error occurred during chat interaction.")
            print(f"❌ Error: {str(e)}\n")
