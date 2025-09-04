import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Гарантируем вывод в UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Конфигурация
DATA_DIR = "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Конфигурация для Foundation Models API
MODEL_NAME = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

MAX_TOKENS = 2000
TEMPERATURE = 0.7

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def load_files_from_dir(dir_path=DATA_DIR):
    texts = []
    for file_path in Path(dir_path).glob("*.txt"):
        with open(file_path, encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    return texts


def create_faiss_index(documents):
    print("Создаем эмбеддинги...")
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    print("Создаем FAISS индекс...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"Индекс создан, векторов: {index.ntotal}")
    return index


def search_documents(query, index, documents, top_k=3):
    query_emb = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]


def generate_answer(query, context_docs):
    context_text = "\n\n".join(context_docs)
    prompt = (
        "Контекст:\n"
        f"{context_text}\n\n"
        f"Вопрос: {query}\n\n"
        "Отвечай только на основе контекста."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Ты помощник, отвечающий строго по контексту."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


def main():
    print("Загружаем документы...")
    documents = load_files_from_dir(DATA_DIR)
    print(f"Загружено документов: {len(documents)}")

    faiss_index = create_faiss_index(documents)

    user_query = input("\nВведите поисковый запрос: ").strip()
    context_docs = search_documents(user_query, faiss_index, documents)
    answer = generate_answer(user_query, context_docs)

    print("\nОтвет модели:")
    print(answer)


if __name__ == "__main__":
    main()
