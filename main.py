import gradio as gr
import pandas as pd
import requests
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer
import numpy as np

# Set Groq API key
os.environ["GROQ_API_KEY"] = "hidden"

# Load your personal FAQ CSV
dataset_path = "/Users/chemb/Downloads/me_faq(Sheet1)-3.csv"

df = pd.read_csv(dataset_path, encoding="ISO-8859-1")
df.columns = df.columns.str.strip()  # Clean column names

# Create Document objects with prompt as page_content and answer in metadata
documents = [
    Document(page_content=row['prompt'], metadata={"answer": row['answer']})
    for _, row in df.iterrows()
]

# Embed using MiniLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embeddings)

# Set up retriever with top-k search
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Tokenizer (optional, currently unused)
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

# Groq fallback (for general queries)
def ask_groq(query):
    if not os.getenv("GROQ_API_KEY"):
        return "Error: Missing Groq API key."

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from Groq.")
    else:
        return f"Groq API error: {response.status_code}, {response.text}"

# Cosine similarity check
def is_relevant(query, document_content, threshold=0.6):  # Lowered threshold for flexibility
    query_embedding = embeddings.embed_query(query)
    doc_embedding = embeddings.embed_query(document_content)

    cosine_similarity = np.dot(query_embedding, doc_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    )
    # Debug print
    print(f"Similarity between query and doc: {cosine_similarity:.3f}")
    return cosine_similarity >= threshold

# Chat logic
def chatbot(query):
    retrieved_docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(retrieved_docs)} documents for query: {query}")

    if not retrieved_docs:
        print("No docs retrieved, falling back to Groq.")
        return ask_groq(query)

    for doc in retrieved_docs:
        if is_relevant(query, doc.page_content):
            print("Found relevant doc.")
            return doc.metadata["answer"]

    print("No docs passed similarity threshold, falling back to Groq.")
    return ask_groq(query)

# Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(placeholder="Ask anything u want bruvah..."),
    outputs=gr.Textbox(),
    title="Chembo's Chatbot",
    description="Ask anything",
)

iface.launch(share=True)
