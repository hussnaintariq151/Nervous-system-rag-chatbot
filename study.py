import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to ask a question
def ask_question(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Main execution
if __name__ == "__main__":
    user_question = input("Ask your question: ")
    print("Answer:", ask_question(user_question))
