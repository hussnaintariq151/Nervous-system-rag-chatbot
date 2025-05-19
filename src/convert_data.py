import pandas as pd
from langchain_core.documents import Document
import fitz  # PyMuPDF

def convert_doctor_patient_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    docs = []
    for _, row in df.iterrows():
        content = f"Question: {row.get('Description', '')}\n\nPatient: {row.get('Patient', '')}\n\nDoctor: {row.get('Doctor', '')}"
        metadata = {"source": "doctor_patient_dataset"}
        doc = Document(page_content=content.strip(), metadata=metadata)
        docs.append(doc)
    return docs

def convert_gale_encyclopedia_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""

    for page in doc:
        all_text += page.get_text()

    doc.close()

    # Split content into logical chunks if needed — here we use double newlines
    chunks = all_text.split("\n\n")

    documents = []
    for chunk in chunks:
        clean_chunk = chunk.strip()
        if clean_chunk:  # avoid empty entries
            documents.append(
                Document(
                    page_content=clean_chunk,
                    metadata={"source": "gale_encyclopedia_pdf"}
                )
            )
    return documents

def get_combined_documents(csv_path, pdf_path):
    doctor_docs = convert_doctor_patient_csv(csv_path)
    gale_docs = convert_gale_encyclopedia_pdf(pdf_path)
    return doctor_docs + gale_docs

# Example usage:
if __name__ == "__main__":
    csv_path = r"D:\Chatbot\Nervous-system-rag-chatbot\Data\ai-medical-chatbot.csv"
    pdf_path = r"D:\Chatbot\Nervous-system-rag-chatbot\Data\he Encyclopedia of the Brain and Brain Disorders, Second -- Carol Turkington and Joseph R_ Harris, Ph_D -- Facts on File library of health and -- 9780816047741 -- 366e84ee3be9a892d719d38dbd51dff6 -- Anna’s A.pdf"

    all_docs = get_combined_documents(csv_path, pdf_path)

    print(f"Total documents loaded: {len(all_docs)}")
    print("Sample document:\n")
    print(all_docs[0])
