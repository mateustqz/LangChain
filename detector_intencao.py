import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
from dotenv import load_dotenv

# ✅ Carregar variáveis de ambiente
load_dotenv()

# ✅ Função para carregar documentos da pasta "docs"
def load_documents(directory="docs"):
    texts = []
    intents = []

    # Verifica se a pasta existe
    if not os.path.exists(directory):
        os.makedirs(directory)  # Cria a pasta se não existir
        return texts, intents

    # 🔄 Percorre todos os arquivos na pasta
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        ext = file.split(".")[-1]

        # 📄 Processa arquivos CSV (mensagem e intenção)
        if ext == "csv":
            df = pd.read_csv(file_path, sep=";", dtype=str)
            texts.extend(df["mensagem"].tolist())   # Salva mensagens
            intents.extend(df["intencao"].tolist()) # Salva intenções

    return texts, intents

# ✅ Carregar documentos e criar embeddings
texts, intents = load_documents()
if not texts:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado na pasta 'docs'!")

print(f"📂 Total de entradas no CSV: {len(texts)}")

# 🔄 Remover índice FAISS antigo e recriar
if os.path.exists("faiss_index"):
    shutil.rmtree("faiss_index")

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = text_splitter.create_documents(texts)  # ⚠️ Alteração aqui!
vectorstore = FAISS.from_documents(documents, embeddings)  # ⚠️ Agora usa documentos fragmentados!
retriever = vectorstore.as_retriever(search_kwargs={"score_threshold": 0.7})  # ⚠️ Adicionado limiar de confiança

print(f"📦 Total de documentos no FAISS: {len(vectorstore.index_to_docstore_id)}")

# 🔍 Teste FAISS manualmente
test_query = "agendar horário"
test_results = retriever.invoke(test_query)
print(f"🛠 Teste FAISS (busca por '{test_query}'): {test_results}")

# ✅ Criar a API FastAPI
app = FastAPI()

class Pergunta(BaseModel):
    pergunta: str

@app.post("/chat")
def chat(pergunta: Pergunta):
    print(f"🔎 Buscando intenção para: {pergunta.pergunta}")
    context_docs = retriever.invoke(pergunta.pergunta)[:1]  # Retorna apenas o documento mais relevante
    print(f"📝 Resultados encontrados: {context_docs}")

    if not context_docs:
        return {"intencoes": []}  # Se não encontrar nada, retorna lista vazia

    # 📌 Filtrar documentos com baixa similaridade
    matched_intents = []
    for doc in context_docs:
        if hasattr(doc, "score") and doc.score < 0.7:
            print("⚠️ Baixa similaridade! Ignorando resultado irrelevante.")
            continue  # Ignora resultados com baixa similaridade

        # 📌 Associar mensagens às intenções
        mensagem_normalizada = doc.page_content.strip().lower()
        for original, intent in zip(texts, intents):
            if mensagem_normalizada == original.strip().lower():
                matched_intents.append({"mensagem": original, "intencao": intent})
                break  # Garante que só pega uma intenção por documento

    if not matched_intents:
        return {"intencoes": []}  # Se não houver correspondência válida, retorna vazio

    print(f"✅ Intenções retornadas: {matched_intents}")
    return {"intencoes": matched_intents}
