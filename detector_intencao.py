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

# ✅ 2️⃣ Criar função para carregar documentos
def load_documents(directory="docs"):
    texts = []

    # Verifica se a pasta existe
    if not os.path.exists(directory):
        os.makedirs(directory)  # Cria a pasta se não existir
        return texts

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        ext = file.split(".")[-1]

        # 📄 Processa arquivos CSV
        if ext == "csv":
            df = pd.read_csv(file_path, sep=";", dtype=str)
            for _, row in df.iterrows():
                texts.append(f"{row['mensagem']};{row['intencao']}")

        # 📑 Processa arquivos PDF
        elif ext == "pdf":
            with open(file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    texts.append(page.extract_text())

        # 📜 Processa arquivos TXT
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        # 📘 Processa arquivos DOCX
        elif ext == "docx":
            doc = docx.Document(file_path)
            texts.extend([para.text for para in doc.paragraphs])

    return texts

# ✅ 3️⃣ Carregar os documentos e criar embeddings
texts = load_documents()
if not texts:
    raise FileNotFoundError("Nenhum documento encontrado na pasta 'docs'.")

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_text("\n".join(texts))
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retorna os 3 mais próximos

# ✅ 4️⃣ Criar a API FastAPI
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/chat")
def chat(pergunta: Pergunta):
    # 🔍 Buscar as mensagens mais próximas nos documentos
    context_docs = retriever.get_relevant_documents(pergunta.pergunta)

    if not context_docs:
        return {"intencoes": []}  # Se não encontrar nada, retorna uma lista vazia

    # 📄 Formata os pares mensagem-intenção corretamente
    intencoes = []
    for doc in context_docs:
        try:
            mensagem, intencao = doc.page_content.split(";")
            intencoes.append({"mensagem": mensagem.strip(), "intencao": intencao.strip()})
        except ValueError:
            continue  # Ignora se não tiver o formato esperado

    return {"intencoes": intencoes}
