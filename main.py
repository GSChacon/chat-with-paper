import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def ask_user_input(first_message = True):
    if first_message:
        user_input = input('Ask questions about your paper (type END to exit): \n')
    else:
        user_input = input('\n')
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
    You use the pieces of the paper to answer the question at the end. You prioratize the documentation to answer.
    Other relevant parts will be provided along the way by the user/human, you use them as well.
    Relevant part of the scientific paper in markdown: {context}
    Remember that other relevant parts will be provided along the way by the user/human, you use them as well.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt

def get_first_message(chat_prompt, docs, user_input):
    chat_prompt_message = chat_prompt.format_prompt(
        context=docs, text=user_input).to_messages()
    return chat_prompt_message

def get_following_messages(chat_prompt_message, last_response, new_user_input, docs):
    chat_prompt_message.append(AIMessage(content = last_response))
    new_context = ''
    for doc in docs:
        new_context = new_context + ' ' + doc.page_content
    chat_prompt_message.append(HumanMessage(content=new_user_input + ' Here is another relevant part of the paper: ' + new_context))
    return chat_prompt_message

def run_chat(chat_prompt_message, model):
    chat = model
    output = chat(chat_prompt_message)
    return output.content



def main():
    path_to_pdf = 'test_data/example.pdf'
    name = get_pdf_name(path_to_pdf)
    check, path_to_store = check_if_db_exists(name)
    if not check:
        print('File does not exist in database. \nCreating vector store...')
        text = load_pdf(path_to_pdf)
        chunks = truncate_pdf_into_chunks(text)
        vector_store = embed_chunks(chunks)
        save_vector_store(vector_store, path_to_store)
        print("Vector store created successfully!")
    else:
        print('File already exists in database. \nLoading vector store...')
        vector_store = load_vector_store(path_to_store)
    print('Loaded vector store')
    chat_prompt = create_prompt_template()
    model = define_model(model_name='gpt-3.5-turbo')
    user_input = ask_user_input()
    if user_input == 'END':
        return
    docs = similarity_search(vector_store, user_input)
    chat_prompt_message = get_first_message(chat_prompt, docs, user_input)
    answer = run_chat(chat_prompt_message, model)
    print(answer)
    while True:
        new_user_input = ask_user_input(first_message = False)
        if new_user_input == 'END':
            break
        docs = similarity_search(vector_store, new_user_input)
        chat_prompt_message = get_following_messages(chat_prompt_message, answer, new_user_input, docs)
        answer = run_chat(chat_prompt_message, model)
        print(answer)
    return

if __name__ == "__main__":
    main()
