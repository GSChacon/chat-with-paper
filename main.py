import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def ask_user_input():
    user_input = input('Ask questions about your paper (type END to exit): \n')
    return user_input

def define_model(model_name='gpt-3.5-turbo'):
    llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature = 0, model_name = model_name)
    return llm

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
    path = f"{os.path.join('vector_store_db', f'{name}_db')}"
    return os.path.exists(path), path

def save_vector_store(vector_store, path):
    vector_store.save_local(path)
    return

def load_vector_store(path):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(path, embeddings)
    return vector_store

def similarity_search(vector_store, user_input, k=3):
    docs = vector_store.similarity_search(query = user_input, k=k)
    return docs

def create_prompt_template():
    template = """You are a large language model designed to help answer questions about scientific papers.
    You are constantly learning and improving, and its capabilities are constantly evolving.
    You are able to process and understand large amounts of text, including markdown, and can use this knowledge to provide accurate and informative responses to a wide range of questions.
    Additionally, you are able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
    If you can't find the answer in the relevant part of the scientific paper, it says so in their answer.
    You understand markdown.
    You use twenty senteces maximum.
    You use the following pieces of the paper to answer the question at the end. You prioratize the documentation to answer.
    Relevant part of the scientific paper in markdown: {context}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt

def run_chat(chat_prompt, model, docs, user_input):
    chat = model
    output = chat(chat_prompt.format_prompt(
        context=docs, text=user_input).to_messages())
    return output.content



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
    print('Loaded vector store')
    chat_prompt = create_prompt_template()
    model = define_model(model_name='gpt-3.5-turbo')
    user_input = ask_user_input()
    docs = similarity_search(vector_store, user_input)
    print(run_chat(chat_prompt, model, docs, user_input))
    return

if __name__ == "__main__":
    main()
