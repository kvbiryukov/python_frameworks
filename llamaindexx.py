import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = "data"  # папка, из которой берём тексты

# Конфигурация для Cloud.ru Foundation Models
MODEL_NAME = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

MAX_TOKENS = 1000
TEMPERATURE = 0.7

# Инициализация клиента OpenAI для Cloud.ru
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def load_files_from_dir(dir_path=DATA_DIR):
    texts = []
    for file_path in Path(dir_path).glob("*.txt"):
        with open(file_path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def search_in_files(query: str, files: list):
    context_text = "\n\n".join(files)
    prompt = f"Данные:\n{context_text}\n\nВопрос: {query}\nОтветь только на основе предоставленных данных. Отвечай конкретным фрагментом из файла."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Ты помощник, который отвечает по данным из контекста."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    documents = load_files_from_dir(DATA_DIR)
    user_query = input("Введите поисковый запрос: ").strip()
    answer = search_in_files(user_query, documents)
    print("\nОтвет модели:\n", answer)
