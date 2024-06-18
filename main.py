import os
import PyPDF2
import cohere
import numpy as np
import pinecone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Initialize Pinecone
pinecone.init(api_key="39bdde41-461f-4c90-b3c4-9916a60be639")
index = pinecone.Index("third-year-cs-first-term")

# Initialize Cohere client
co = cohere.Client('TcZjPcNuntkBpDbSsH5M5X8N9vlSs6Mq11KoL3rd')

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    k: int = 5

def extract_text_from_pdfs(directory):
    text_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            with open(os.path.join(directory, filename), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                text_data.append((filename, text))
    return text_data

pdf_directory = '/content/drive/MyDrive/First term Cs 3rd/first'
pdf_texts = extract_text_from_pdfs(pdf_directory)

# Generate embeddings with embed-multilingual-v3.0
documents = [text for _, text in pdf_texts]
response = co.embed(
    texts=documents,
    model='embed-multilingual-v3.0',
    input_type='search_document'
)
embeddings = response.embeddings

# Store embeddings in Pinecone
for i, embedding in enumerate(embeddings):
    index.upsert([(str(i), embedding)])

def query_pinecone(query_text, k=5):
    # Generate embedding for the query
    query_embedding = co.embed(
        texts=[query_text],
        model='embed-multilingual-v3.0',
        input_type='search_query'
    ).embeddings[0]

    # Query Pinecone with keyword arguments
    query_response = index.query(vector=query_embedding, top_k=k)
    top_docs_indices = [match['id'] for match in query_response['matches']]

    # Retrieve top-k documents
    top_docs = [pdf_texts[int(idx)][1] for idx in top_docs_indices]

    return top_docs

def generate_answer(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=512,
        temperature=0.8
    )
    return response.generations[0].text.strip()

@app.post("/query")
async def query_chatbot(request: QueryRequest):
    query_text = request.query
    k = request.k
    
    try:
        top_docs = query_pinecone(query_text, k)
        context = " ".join(top_docs)
        answer = generate_answer(query_text, context)
        return {"question": query_text, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
