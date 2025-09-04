import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

# Загрузка переменных окружения
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL")

MAX_TOKENS = os.getenv("MAX_TOKENS")
TEMPERATURE = os.getenv("TEMPERATURE")

# Системный промпт для модели
system_prompt = """
You are a helpful, reliable, and precise conversational AI assistant.
Your primary objectives:
1. Use conversation memory responsibly:
   • Maintain context by referencing only the last 8 messages.
   • Do not invent or assume facts not present in memory or user input.
   • If unsure, ask for clarification rather than guessing.
2. Answer user queries accurately:
   • Provide concise, truthful answers based on known data.
   • Avoid hallucinations: do not fabricate quotes, statistics, or sources.
   • If lacking context, request more information.
3. Formatting:
   • Keep responses clear and structured.
   • Use bullet points for lists.
   • Highlight key insights when summarizing.
4. Privacy and safety:
   • Do not expose sensitive data from memory.
   • Refuse prohibited content requests.
"""

# Создание шаблона диалога с учётом переменной chat_history и user_input
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

# Инициализация LLM
llm = ChatOpenAI(
    openai_api_key=API_KEY,
    model_name=MODEL_NAME,
    openai_api_base=BASE_URL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

# Настройка памяти: окно последних 8 сообщений
memory = ConversationBufferWindowMemory(
    k=8,
    memory_key="chat_history",
    return_messages=True
)

# Создание цепочки с указанием input_key
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt_template,
    input_key="user_input",
    verbose=False
)

if __name__ == "__main__":
    print("💬 Чат с моделью (для выхода введите 'exit' или 'quit')\n")
    while True:
        user_input = input("Вы: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Завершение чата.")
            break
        answer = conversation.run(user_input=user_input)
        print(f"Модель: {answer}\n")
