import os
from dotenv import load_dotenv
from evolution_openai import OpenAI

load_dotenv()

MODEL_NAME = os.getenv("MODEL")
MAX_TOKENS = 512
TEMPERATURE = 0.7

BASE_URL  = os.getenv("BASE_URL")
PROJECT_ID = os.getenv("PROJECT_IDD")
KEY_ID    = os.getenv("API_TOKENN")
SECRET    = os.getenv("API_SECRETT")

print("MODEL:", MODEL_NAME)
print("PROJECT_ID:", PROJECT_ID)
print("KEY_ID:", KEY_ID)
print("SECRET:", SECRET[:5] + "..." if SECRET else None)

client = OpenAI(
    key_id=KEY_ID,
    secret=SECRET,
    base_url=BASE_URL,
    project_id=PROJECT_ID
)

class ChatMemory:
    """Хранит историю диалога между пользователем и моделью."""
    def __init__(self):
        self.history = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})

    def get_history(self):
        return self.history.copy()

    def clear(self):
        self.history.clear()

memory = ChatMemory()

def chat_with_model(user_input: str):
    """Отправка сообщения модели с историей диалога"""
    memory.add_user_message(user_input)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Ты полезный ассистент. Отвечай на русском языке."},
            *memory.get_history()
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    answer = response.choices[0].message.content
    memory.add_ai_message(answer)
    return answer

if __name__ == "__main__":
    print("💬 Чат с моделью (для выхода введите 'exit' или 'quit')\n")

    while True:
        user_input = input("Вы: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Завершение чата.")
            break

        answer = chat_with_model(user_input)
        print(f"Модель: {answer}\n")