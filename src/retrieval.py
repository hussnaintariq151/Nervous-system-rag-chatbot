import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Get AstraDB credentials from environment variables
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
ASTRA_DB_COLLECTION = "medical_rag_chunks"  # fixed collection name as you requested

# OpenAI API key is automatically picked up by OpenAIEmbeddings from environment
embedding_model = OpenAIEmbeddings()

# Connect to AstraDB vector store
vectorstore = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_KEYSPACE,
)

# Create retriever
retriever = vectorstore.as_retriever()

# Run a test query
query = "What does the cerebellum control in the human body?"
results = retriever.invoke(query)


# Print results
print(f"🔍 Query: {query}\n")
print("📄 Retrieved Chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---\n{doc.page_content[:500]}\n")
