# LangChain

.venv/Scripts/Activate

uvicorn detector_intencao:app --reload

-----------------------------------------
1️⃣ Confirme que todos os pacotes necessários estão no requirements.txt
Execute no terminal:

cat requirements.txt

O conteúdo deve incluir:

fastapi
uvicorn
pandas
python-dotenv
langchain
langchain-community
langchain-openai
faiss-cpu
PyPDF2
python-docx


Se algum estiver faltando, adicione e faça novamente. 

pip install -r requirements.txt

pip install -r requirements.txt --upgrade

-------------------------------------------------
1 - Vá até a aba "Settings" no Railway
2 - Role para baixo até a seção "Domains"
3️ - Clique em "Generate Domain" para criar uma URL pública
4️ - Copie a URL gerada e teste no navegador acessando:

https://SUA-URL-GERADA.railway.app/docs

Agora sua API deve estar acessível!

-----------------------------------------------------------------------------

No Railway (ou em um servidor com ambiente virtual):

pip install -r requirements.txt --upgrade
Se quiser atualizar todos os pacotes de uma vez, use:

pip list --outdated | awk 'NR>2 {print $1}' | xargs -n1 pip install -U
Isso atualiza todos os pacotes instalados.

Atualizar requirements.txt Fora do Virtual Env
Se você estiver fora do ambiente virtual, primeiro precisa ativá-lo:

.venv\Scripts\Activate
-------------------------------------------------------------------------

 TESTE LOCALMENTE
Antes de subir para o Railway, teste se tudo está funcionando no seu ambiente local:

Ative o ambiente virtual (caso ainda não esteja ativo)

.venv\Scripts\Activate  # Windows (PowerShell)
.venv\Scripts\activate.bat  # Windows (CMD)

Rode a API para testar

uvicorn detector_intencao:app --host 0.0.0.0 --port 8000

Abra no navegador, Vá para:

http://127.0.0.1:8000/docs
e veja se a API está rodando normalmente.

 COMMIT E DEPLOY NO RAILWAY

git add .
git commit -m "Adicionando suporte a PDF e DOCX"
git push origin main

Railway vai fazer o deploy automaticamente
Se precisar forçar o deploy, vá no Railway, clique em "Deployments" e Re-deploy

Depois que o deploy terminar, teste se está funcionando na nuvem:

SEU-LINK-DO-RAILWAY/docs
(exemplo: https://seu-projeto.up.railway.app/docs)

Faça o mesmo teste do /chat para garantir que a API está funcionando corretamente no Railway.