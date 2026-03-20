from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

def build_vector_db_from_texts(text_list):
    docs = [Document(page_content=txt) for txt in text_list]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embedding=embeddings)

def retrieve_relevant_docs(db, query, k=3):
    return db.similarity_search(query, k=k)