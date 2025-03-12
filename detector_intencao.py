import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# ✅ 1️⃣ Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv()

# ✅ 2️⃣ Verifica se o CSV está presente
CSV_PATH = "knowledge_base.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Arquivo {CSV_PATH} não encontrado!")

# ✅ 3️⃣ Carregar o CSV corretamente
df = pd.read_csv(CSV_PATH, sep=";", dtype=str)
texts = df["mensagem"].tolist()

# ✅ 4️⃣ Criar os embeddings e vetor de busca
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

# ✅ 5️⃣ Configurar o modelo de linguagem
llm = ChatOpenAI()

# ✅ 6️⃣ Criar o template de resposta
rag_template = """
Você é um atendente de uma empresa.
Seu trabalho é conversar com os clientes, consultando a base de 
conhecimentos da empresa, e dar
uma resposta simples e precisa para ele, baseada na
base de dados da empresa fornecida como 
contexto.

Contexto: {context}

Pergunta do Cliente: {question}
"""

prompt = ChatPromptTemplate.from_template(rag_template)

# ✅ 7️⃣ Criar a API FastAPI
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/chat")
def chat(pergunta: Pergunta):
    context_docs = retriever.get_relevant_documents(pergunta.pergunta)
    context_text = "\n".join([doc.page_content for doc in context_docs])
    final_prompt = prompt.format(context=context_text, question=pergunta.pergunta)
    response = llm.invoke(final_prompt)
    return {"resposta": response.content}
