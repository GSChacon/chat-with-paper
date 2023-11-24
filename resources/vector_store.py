import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class VectorStore:
    def __init__(self, db_exists, path_to_db, chunks):
        self.db_exists = db_exists
        self.path_to_db = path_to_db
        self.chunks = chunks

    def embed_chunks(self):
        embeddings = OpenAIEmbeddings()
        chunks_embeded = FAISS.from_texts(self.chunks, embedding = embeddings)
        return chunks_embeded

    def save_vector_store(self, chunks_embeded):
        chunks_embeded.save_local(self.path_to_db)
        return

    def load_vector_store(self):
        embeddings = OpenAIEmbeddings()
        faiss_index = FAISS.load_local(self.path_to_db, embeddings)
        return faiss_index

    def similarity_search(self, faiss_index, user_input, k=3):
        docs = faiss_index.similarity_search(query = user_input, k=k)
        return docs

    def handle_vector_store(self):
        if self.db_exists:
            faiss_index = self.load_vector_store()
            return faiss_index
        chunks_embeded = self.embed_chunks()
        self.save_vector_store(chunks_embeded)
        faiss_index = self.load_vector_store()
        return faiss_index
