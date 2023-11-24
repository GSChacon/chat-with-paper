import os
import sys
from dotenv import load_dotenv
from utils.pdf_utils import PdfHandler
from resources.vector_store import VectorStore
from resources.handle_user_input import HandleUserInput
from resources.model_interface import ChatLLMOpenAi

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main(path_to_pdf):
    db_exists, path_to_db, chunks = PdfHandler(path_to_pdf).pdf_handler()

    if db_exists:
        print('File already exists in database. \nLoading vector store...')
        vectore_store = VectorStore(db_exists, path_to_db, chunks)
        faiss_index = vectore_store.handle_vector_store()
        print("Vector store loaded successfully!")
    else:
        print('File does not exist in database. \nCreating vector store...')
        vectore_store = VectorStore(db_exists, path_to_db, chunks)
        faiss_index = vectore_store.handle_vector_store()
        print("Vector store created successfully!")

    user_input = HandleUserInput().ask_user_input(first_message = True)
    if HandleUserInput().check_end(user_input):
        print('Thank you for using chat-with-paper!')
        return
    docs = vectore_store.similarity_search(faiss_index, user_input)
    llm = ChatLLMOpenAi(OPENAI_API_KEY)
    answer, chat_prompt_message = llm.initiate_chat(docs, user_input)
    print(answer)

    while True:
        new_user_input = HandleUserInput().ask_user_input(first_message = False)
        if HandleUserInput().check_end(new_user_input):
            break
        docs = vectore_store.similarity_search(faiss_index, new_user_input)
        answer, chat_prompt_message = llm.continue_chat(answer, new_user_input, docs, chat_prompt_message)
        print(answer)
    print('Thank you for using chat-with-paper!')
    return

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to/paper.pdf>")
    else:
        path_to_pdf = sys.argv[1]
        main(path_to_pdf)
