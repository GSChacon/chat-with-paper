from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage

class ChatLLMOpenAi:
    def __init__(self, key, model_name='gpt-3.5-turbo'):
        self.key = key
        self.model = ChatOpenAI(openai_api_key = key, temperature = 0, model_name = model_name)
        pass

    def create_prompt_template(self):
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

    def get_first_message(self, chat_prompt, docs, user_input):
        chat_prompt_message = chat_prompt.format_prompt(
            context=docs, text=user_input).to_messages()
        return chat_prompt_message

    def get_following_messages(self, chat_prompt_message, last_response, new_user_input, docs):
        chat_prompt_message.append(AIMessage(content = last_response))
        new_context = ''
        for doc in docs:
            new_context = new_context + ' ' + doc.page_content
        chat_prompt_message.append(HumanMessage(content=new_user_input + ' Here is another relevant part of the paper: ' + new_context))
        return chat_prompt_message

    def run_chat(self, chat_prompt_message, model):
        chat = model
        output = chat(chat_prompt_message)
        return output.content

    def initiate_chat(self, docs, user_input):
        chat_prompt = self.create_prompt_template()
        chat_prompt_message = self.get_first_message(chat_prompt, docs, user_input)
        answer = self.run_chat(chat_prompt_message, self.model)
        return answer, chat_prompt_message

    def continue_chat(self, answer, new_user_input, docs, history):
        chat_prompt_message = self.get_following_messages(history, answer, new_user_input, docs)
        answer = self.run_chat(chat_prompt_message, self.model)
        return answer, chat_prompt_message
