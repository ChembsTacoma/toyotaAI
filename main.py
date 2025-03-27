import gradio as gr
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import AutoTokenizer

# Load the Toyota FAQ dataset
dataset_path = "FAQ Toyota(Sheet1)-2.csv"
df = pd.read_csv(dataset_path, encoding="ISO-8859-1")

documents = [Document(page_content=str(row)) for row in df['answer'].dropna().tolist()]

# Initialize FAISS Vector Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embeddings)

# Initialize the retriever with increased search depth (k=5)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
MAX_INPUT_LENGTH = 1024

# Tokenize and truncate input text
def truncate_input(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    return tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

def chatbot(query):
    truncated_query = truncate_input(query)

    # Retrieve relevant context before passing to LLM
    retrieved_docs = retriever.invoke(truncated_query)  # Using invoke() instead of deprecated method
    print("Retrieved Docs:", [doc.page_content for doc in retrieved_docs])

    if not retrieved_docs:
        return "Sorry, I couldn't find any relevant information."

    # Return the first retrieved document as the answer
    return retrieved_docs[0].page_content  # Return best match

# Create Gradio Interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(placeholder="Ask a question about Toyota..."),
    outputs=gr.Textbox(),
    title="Toyota Chatbot",
    description="Ask any question about Toyota cars, features, and services!"
)

iface.launch(share=True)
