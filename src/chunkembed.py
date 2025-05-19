

# import os
# import pickle
# import time
# from dotenv import load_dotenv
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from tiktoken import encoding_for_model

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from convert_data import get_combined_documents

# # === Step 1: Load environment and initialize ===
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY not found in .env file!")

# # File paths
# csv_path = r"D:\Chatbot\Nervous-system-rag-chatbot\Data\ai-medical-chatbot.csv"
# pdf_path = r"D:\Chatbot\Nervous-system-rag-chatbot\Data\he Encyclopedia of the Brain and Brain Disorders, Second -- Carol Turkington and Joseph R_ Harris, Ph_D -- Facts on File library of health and -- 9780816047741 -- 366e84ee3be9a892d719d38dbd51dff6 -- Anna’s A.pdf"
# checkpoint_folder = "embeddings_checkpoints"
# os.makedirs(checkpoint_folder, exist_ok=True)

# # === Step 2: Load and chunk data ===
# def load_and_chunk(csv_path, pdf_path, chunk_size=500, chunk_overlap=100):
#     print("📚 Loading documents...")
#     documents = get_combined_documents(csv_path, pdf_path)
#     print(f"✅ Total documents loaded: {len(documents)}")

#     print("🔪 Chunking documents...")
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
#     return splitter.split_documents(documents)

# # === Step 3: Embedding initializer ===
# embedding_model = OpenAIEmbeddings(
#     openai_api_key=openai_api_key,
#     model="text-embedding-3-small"
# )

# # === Step 4: Tokenizer ===
# def num_tokens(text, model="text-embedding-3-small"):
#     enc = encoding_for_model(model)
#     return len(enc.encode(text))

# # === Step 5: Get existing batches ===
# def get_existing_batches(folder):
#     return set(
#         int(f.split("_")[2].split(".")[0])
#         for f in os.listdir(folder) if f.startswith("embeddings_batch_")
#     )

# # === Step 6: Embedding function with token limit ===
# def embed_batch(batch_id, batch_chunks, checkpoint_folder):
#     batch_texts = [doc.page_content for doc in batch_chunks]
#     token_total = sum(num_tokens(t) for t in batch_texts)

#     if token_total > 8192:
#         print(f"⚠️ Batch {batch_id} has {token_total} tokens — splitting into smaller sub-batches...")

#         # Split batch into sub-batches
#         sub_batches = []
#         current_sub = []
#         current_tokens = 0

#         for doc in batch_chunks:
#             doc_tokens = num_tokens(doc.page_content)
#             if current_tokens + doc_tokens > 8192:
#                 sub_batches.append(current_sub)
#                 current_sub = [doc]
#                 current_tokens = doc_tokens
#             else:
#                 current_sub.append(doc)
#                 current_tokens += doc_tokens
#         if current_sub:
#             sub_batches.append(current_sub)

#         all_embeddings = []
#         for idx, sub in enumerate(sub_batches):
#             print(f"🔁 Embedding sub-batch {batch_id}-{idx} with {len(sub)} docs...")
#             all_embeddings.extend(embed_batch(f"{batch_id}_{idx}", sub, checkpoint_folder))
#         return all_embeddings

#     try:
#         start_time = time.time()
#         embeddings = embedding_model.embed_documents(batch_texts)
#         with open(os.path.join(checkpoint_folder, f"embeddings_batch_{batch_id}.pkl"), "wb") as f:
#             pickle.dump(embeddings, f)
#         print(f"✅ Batch {batch_id} embedded in {time.time() - start_time:.2f}s")
#         return embeddings
#     except Exception as e:
#         print(f"❌ Error embedding batch {batch_id}: {e}")
#         return []

# # === Step 7: Process batches ===
# def process_batches_parallel(chunked_docs, batch_size, checkpoint_folder, max_workers=4):
#     existing_batches = get_existing_batches(checkpoint_folder)
#     batches_to_process = [
#         (i // batch_size, chunked_docs[i:i + batch_size])
#         for i in range(0, len(chunked_docs), batch_size)
#         if (i // batch_size) not in existing_batches
#     ]

#     print(f"🚀 Processing {len(batches_to_process)} new batches in parallel...")
#     all_embeddings = []

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(embed_batch, batch_id, batch_chunks, checkpoint_folder)
#             for batch_id, batch_chunks in batches_to_process
#         ]
#         for future in tqdm(futures, desc="Embedding Progress"):
#             all_embeddings.extend(future.result())

#     return all_embeddings

# # === Step 8: Execute pipeline ===
# def main():
#     chunked_docs = load_and_chunk(csv_path, pdf_path)
#     print(f"✅ Total chunks created: {len(chunked_docs)}")

#     # Token and cost estimation
#     total_tokens = sum(num_tokens(doc.page_content) for doc in chunked_docs)
#     estimated_cost = total_tokens / 1000 * 0.00002
#     print(f"\n🧮 Total tokens to embed: {total_tokens}")
#     print(f"💰 Estimated cost: ${estimated_cost:.4f} USD")

#     batch_size = 2000
#     all_embeddings = process_batches_parallel(chunked_docs, batch_size, checkpoint_folder)

#     print("\n✅ All batches processed.")
#     print(f"🧩 Embedded chunks this run: {len(all_embeddings)}")

#     if all_embeddings:
#         print("🔎 Sample chunk:\n", chunked_docs[0].page_content[:300])
#         print("🔢 Sample embedding (first 5 values):", all_embeddings[0][:5])
#     else:
#         print("⚠️ No new embeddings were processed in this run.")

# if __name__ == "__main__":
#     main()




import os
import pickle
import time
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tiktoken import encoding_for_model

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from convert_data import get_combined_documents

# === Step 1: Load environment and initialize ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# File paths
csv_path = r"D:\Chatbot\Nervous-system-rag-chatbot\Data\ai-medical-chatbot.csv"
pdf_path = r"D:\Chatbot\Nervous-system-rag-chatbot\Data\he Encyclopedia of the Brain and Brain Disorders, Second -- Carol Turkington and Joseph R_ Harris, Ph_D -- Facts on File library of health and -- 9780816047741 -- 366e84ee3be9a892d719d38dbd51dff6 -- Anna’s A.pdf"
checkpoint_folder = "embeddings_checkpoints"
os.makedirs(checkpoint_folder, exist_ok=True)

# === Step 2: Load and chunk data ===
def load_and_chunk(csv_path, pdf_path, chunk_size=500, chunk_overlap=100):
    print("📚 Loading documents...")
    documents = get_combined_documents(csv_path, pdf_path)
    print(f"✅ Total documents loaded: {len(documents)}")

    print("🔪 Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

# === Step 3: Embedding initializer ===
embedding_model = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    model="text-embedding-3-small"
)

# === Step 4: Tokenizer ===
def num_tokens(text, model="text-embedding-3-small"):
    enc = encoding_for_model(model)
    return len(enc.encode(text))

# === Step 5: Get existing batches ===
def get_existing_batches(folder):
    return set(
        int(f.split("_")[2].split(".")[0])
        for f in os.listdir(folder) if f.startswith("embeddings_batch_")
    )

# === Step 6: Embedding function with token limit ===
def embed_batch(batch_id, batch_chunks, vectorstore):
    from numpy.linalg import norm

    def is_valid_embedding(vec, threshold=1e-6):
        return norm(vec) > threshold

    batch_texts = [doc.page_content for doc in batch_chunks]
    metadatas = [doc.metadata for doc in batch_chunks]
    token_total = sum(num_tokens(t) for t in batch_texts)

    # If the batch is too large, split it into two and process recursively
    if token_total > 8192:
        print(f"⚠️ Batch {batch_id} has {token_total} tokens — splitting into smaller sub-batches...")
        
        if len(batch_chunks) == 1:
            print(f"⚠️ Single document too large (tokens: {token_total}) — skipping.")
            return 0  # prevent infinite recursion

        mid = len(batch_chunks) // 2
        count1 = embed_batch(f"{batch_id}-a", batch_chunks[:mid], vectorstore)
        count2 = embed_batch(f"{batch_id}-b", batch_chunks[mid:], vectorstore)
        return count1 + count2

    try:
        start_time = time.time()
        embeddings = embedding_model.embed_documents(batch_texts)

        valid_texts, valid_metas = [], []
        for text, meta, emb in zip(batch_texts, metadatas, embeddings):
            if is_valid_embedding(emb):
                valid_texts.append(text)
                valid_metas.append(meta)

        if valid_texts:
            vectorstore.add_texts(texts=valid_texts, metadatas=valid_metas)

        print(f"✅ Batch {batch_id} embedded & uploaded in {time.time() - start_time:.2f}s")
        return len(valid_texts)
    except Exception as e:
        print(f"❌ Error embedding/uploading batch {batch_id}: {e}")
        return 0


# === Step 7: Process batches ===
def process_batches_parallel(chunked_docs, batch_size, vectorstore, max_workers=4):
    batches_to_process = [
        (i // batch_size, chunked_docs[i:i + batch_size])
        for i in range(0, len(chunked_docs), batch_size)
    ]

    print(f"🚀 Uploading {len(batches_to_process)} batches directly to vector store...")

    total_uploaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(embed_batch, batch_id, batch_chunks, vectorstore)
            for batch_id, batch_chunks in batches_to_process
        ]
        for future in tqdm(futures, desc="Embedding Upload Progress"):
            total_uploaded += future.result()

    print(f"📦 Total documents uploaded: {total_uploaded}")
    return total_uploaded


# === Step 8: Execute pipeline ===
from langchain_astradb import AstraDBVectorStore

def main():
    chunked_docs = load_and_chunk(csv_path, pdf_path)
    print(f"✅ Total chunks created: {len(chunked_docs)}")

    total_tokens = sum(num_tokens(doc.page_content) for doc in chunked_docs)
    estimated_cost = total_tokens / 1000 * 0.00002
    print(f"\n🧮 Total tokens to embed: {total_tokens}")
    print(f"💰 Estimated cost: ${estimated_cost:.4f} USD")

    # Initialize AstraDBVectorStore with proper parameters
    vectorstore = AstraDBVectorStore(
        embedding=embedding_model,
        collection_name=os.getenv("ASTRA_DB_COLLECTION"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    )

    batch_size = 2000
    all_embeddings = process_batches_parallel(chunked_docs, batch_size, vectorstore)

    print("\n✅ All batches embedded and uploaded.")

if __name__ == "__main__":
    main()
