import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def handle_user_input():
    user_input = input('What do you want to talk about? \n')
    return user_input

def define_model():
    llm = OpenAI(openai_api_key = OPENAI_API_KEY)
    return llm

def predict(model, user_input):
    res = model.predict(user_input)
    return res

def get_pdf_name(path_to_pdf):
    name_list = path_to_pdf.split('/')
    name = name_list[-1][:-4]
    return name

def load_pdf(path_to_pdf):
    loader = PyPDFLoader(path_to_pdf, extract_images = True)
    pages = loader.load_and_split()
    full_text = ''
    for page in pages:
        full_text = full_text + "\n" + page.page_content
    return full_text

def truncate_pdf_into_chunks(text, chunk_size = 1000, chunk_overlap = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_text(text=text)
    return chunks

def embed_chunks(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding = embeddings)
    return vector_store

def check_if_db_exists(name):
    path = f"{os.path.join('vector_store_db', f'{name}.pkl')}"
    return os.path.exists(path), path

def save_vector_store(vector_store, path):
    with open(path, "wb") as f:
        pickle.dump(vector_store, f)
    return None

def load_vector_store(path):
    with open(path, 'rb') as f:
        vector_store = pickle.load(f)
    return vector_store



def main():
    path_to_pdf = 'test_data/example.pdf'
    name = get_pdf_name(path_to_pdf)
    check, path_to_store = check_if_db_exists(name)
    if not check:
        print('File does not exist in database. \nCreating vector store... \n')
        text = load_pdf(path_to_pdf)
        chunks = truncate_pdf_into_chunks(text)
        vector_store = embed_chunks(chunks)
        save_vector_store(vector_store, path_to_store)
        print("Vector store created successfully!\n")
    else:
        print('File already exists in database. \nLoading vector store... \n')
        vector_store = load_vector_store(path_to_store)
    print('Loaded vector store.')
    return

if __name__ == "__main__":
    main()
