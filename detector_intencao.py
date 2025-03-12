import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx

# ✅ 1️⃣ Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

# ✅ 2️⃣ Função para carregar documentos da pasta "docs"
def load_documents(directory="docs"):
    texts = []
    intents = []

    # Verifica se a pasta existe
    if not os.path.exists(directory):
        os.makedirs(directory)  # Cria a pasta se não existir
        return texts, intents   # Retorna listas vazias se não houver arquivos

    # 🔄 Percorre todos os arquivos na pasta
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        ext = file.split(".")[-1]  # Pega a extensão do arquivo

        # 📄 Processa arquivos CSV (mensagem e intenção)
        if ext == "csv":
            df = pd.read_csv(file_path, sep=";", dtype=str)
            texts.extend(df["mensagem"].tolist())   # Salva mensagens
            intents.extend(df["intencao"].tolist()) # Salva intenções

    return texts, intents  # Retorna listas com mensagens e intenções

# ✅ 3️⃣ Carregar documentos e criar embeddings
texts, intents = load_documents()
if not texts:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado na pasta 'docs'!")

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_text("\n".join(texts))
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()

# ✅ 4️⃣ Criar a API FastAPI
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/chat")
def chat(pergunta: Pergunta):
    # 🔍 Buscar as mensagens mais próximas no CSV
    context_docs = retriever.get_relevant_documents(pergunta.pergunta)

    if not context_docs:
        return {"intencoes": []}  # Retorna lista vazia se não encontrar

    # 📄 Associar mensagens às intenções do CSV
    matched_intents = []
    for doc in context_docs:
        if doc.page_content in texts:
            index = texts.index(doc.page_content)
            matched_intents.append({"mensagem": doc.page_content, "intencao": intents[index]})

    return {"intencoes": matched_intents}
