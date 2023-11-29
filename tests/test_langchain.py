import os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embedding_model_name = '/home/searchgpt/pretrained_models/ernie-gram-zh'
docs_path = '/home/searchgpt/yq/Knowledge-ChatGLM/docs'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

docs = []

for doc in os.listdir(docs_path):
    if doc.endswith('.txt'):
        print(doc)
        loader = UnstructuredFileLoader(f'{docs_path}/{doc}', mode="elements")
        doc = loader.load()
        docs.extend(doc)

vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local('vector_store_local')
search_result = vector_store.similarity_search_with_score(query='双校互补', k=2)
print(search_result)

loader = UnstructuredFileLoader(f'{docs_path}/港科广.txt', mode="elements")
doc = loader.load()
vector_store.add_documents(doc)
print(doc)
search_result = vector_store.similarity_search_with_score(query='港科大', k=2)
print(search_result)
