import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

class PdfHandler:
    def __init__(self, path_to_pdf):
        self.path_to_pdf = path_to_pdf

    def get_pdf_name(self):
        name_list = self.path_to_pdf.split('/')
        name = name_list[-1][:-4]
        return name

    def check_if_db_exists(self, name):
        path_to_db = f"{os.path.join('vector_store_db', f'{name}_db')}"
        return os.path.exists(path_to_db), path_to_db

    def load_pdf(self):
        loader = PyPDFLoader(self.path_to_pdf, extract_images = True)
        pages = loader.load_and_split()
        full_text = ''
        for page in pages:
            full_text = full_text + "\n" + page.page_content
        return full_text

    def truncate_pdf_into_chunks(self, text, chunk_size = 1000, chunk_overlap = 200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap)
        chunks = text_splitter.split_text(text=text)
        return chunks

    def pdf_handler(self, chunk_size = 1000, chunk_overlap = 200):
        name = self.get_pdf_name()
        db_exists, path_to_db = self.check_if_db_exists(name)
        chunks = None
        if db_exists:
            return db_exists, path_to_db, chunks
        text = self.load_pdf()
        chunks = self.truncate_pdf_into_chunks(text, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        return db_exists, path_to_db, chunks
