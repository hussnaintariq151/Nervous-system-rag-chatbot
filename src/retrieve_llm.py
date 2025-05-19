# import os
# import openai
# from langchain_astradb import AstraDBVectorStore
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

# # Load environment variables (ensure .env is loaded or set variables properly)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
# ASTRA_DB_COLLECTION = "medical_rag_chunks"

# # Initialize embedding and vectorstore
# embedding_model = OpenAIEmbeddings()

# vectorstore = AstraDBVectorStore(
#     embedding=embedding_model,
#     collection_name=ASTRA_DB_COLLECTION,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     namespace=ASTRA_DB_KEYSPACE,
# )

# retriever = vectorstore.as_retriever()

# def generate_answer(query: str):
#     # Retrieve relevant documents
#     results = retriever.invoke(query)


#     # Combine retrieved chunks into context for prompt
#     context = "\n\n".join([doc.page_content for doc in results])

#     # Build prompt for GPT
#     prompt = (
#         f"You are a helpful medical assistant. Use the following context to answer the question.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {query}\n"
#         f"Answer:"
#     )

#     # Call OpenAI ChatCompletion API (GPT-4 style)
#     response = openai.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt},
#     ],
#     max_tokens=300,
#     temperature=0.2,
# )


#     answer = response.choices[0].message.content.strip()
#     return answer


# if __name__ == "__main__":
#     query = "What does the cerebellum control in the human body?"
#     print(f"🔍 Query: {query}\n")
#     answer = generate_answer(query)
#     print("🧠 Generated Answer:\n")
#     print(answer)


import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
ASTRA_DB_COLLECTION = "medical_rag_chunks"

# Step 1: Define output schema
class MedicalAnswer(BaseModel):
    answer: str = Field(..., description="Clear, medically accurate, and concise response to the question")

# Step 2: Output parser
output_parser = PydanticOutputParser(pydantic_object=MedicalAnswer)

# Step 3: Prompt with format instructions
MEDICAL_PROMPT_TEMPLATE = """
You are a helpful medical assistant. Use the following medical context to answer the question accurately and concisely.

Follow this format:
{format_instructions}

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:
"""

prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT_TEMPLATE).partial(
    format_instructions=output_parser.get_format_instructions()
)

# Step 4: Generation chain
def generation():
    embeddings = OpenAIEmbeddings()

    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=ASTRA_DB_COLLECTION,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return chain

# Step 5: Run & test
if __name__ == "__main__":
    chain = generation()
    query = "What does the cerebellum control in the human body?"
    result = chain.invoke(query)
    print(f"🧠 Answer:\n{result.answer}")

