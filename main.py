# from llama_index.core import Settings
# from llama_index.core import PromptTemplate
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from openai import OpenAI


query = st.chat_input("Ask a question!")

# Load and chunk documents
pdfdb = ['./DocLib/7.06__The_Quantum_Harmonic_Oscillator.pdf']


chunks = []
for doc in pdfdb:
    raw_document = PyMuPDFLoader(doc).load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    doc_chunks = text_splitter.split_documents(raw_document)
    for chunk in doc_chunks:
        chunks.append(chunk) 

# Load vector embedding model
embeddings = HuggingFaceEmbeddings(model_name="snowflake/snowflake-arctic-embed-m")
db = Chroma.from_documents(chunks, embeddings)
docs = db.similarity_search(query)
search_results = ''.join([doc.page_content for doc in docs])
# query = "What is a dual space?"
# text = "This is a test document."
# query_result = embeddings.embed_query(query)
# print(query_result[:3])


client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)


# r = requests.get('http://localhost:11434/v1/chat/completions', data={
#  "model": "llama3",
#  "messages": [
#    {"role":"system", "content":"You are an AI assistant that helps people find information."},
#    {"role":"user","content":"what is FRS?"}
#  ]
# })
response = client.chat.completions.create(
  model="llama3",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": query + search_results}
  ]
)

with st.chat_message("assistant"):
    reply = st.write(response.choices[0].message.content)
# print(r.text)

#{"model": "llama3", "prompt": [query].append(docs), "stream": False}