import os
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Task, Crew, BaseLLM

load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

class CloudRuLLM(BaseLLM):
    """Адаптер для Cloud.ru Foundation Models через стандартную библиотеку OpenAI"""

    def __init__(self):
        super().__init__(model=MODEL)
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

    def call(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content

# Инициализация адаптера
llm = CloudRuLLM()

# Создание агентов
researcher = Agent(
    role="Исследователь",
    goal="Провести структурированный анализ применения технологий искусственного интеллекта в российском бизнесе за период 2023–2025 года.",
    backstory="""
    Ты эксперт-аналитик с опытом работы в сфере корпоративных инноваций и ИИ. Отличается глубоким пониманием рынков, умеет критически оценивать
    тренды и проверять источники. Всегда выделяет главное и избегает предположений без фактов.
    """,
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Технический писатель",
    goal="""
    Создать краткий, структурированный и деловой отчет на основе полученного анализа;
    акцент на ясность, практические рекомендации и бизнес-ориентированность.
    """,
    backstory="""
    Ты профессиональный технический писатель с опытом подготовки деловых обзоров и рекомендаций для руководителей.
    Отличается умением переводить сложные выводы на простой, понятный для бизнес-аудитории язык без потери точности.
    """,
    llm=llm,
    verbose=True
)

# Определение задач
research_task = Task(
    description="""
    Собери актуальный аналитический обзор по теме: «Применение ИИ в российском бизнесе в 2023–2025».
    Требуется описать:
    — главные направления использования (3-4);
    — ключевые компании и конкретные решения (3-4);
    — основные тренды и вызовы (3-4).
    Оформи анализ как структурированный список с пояснениями. Ответ до 400 слов
    """,
    agent=researcher,
    expected_output="""
    Аналитический отчет, включающий список направлений применения, перечень компаний с их решениями и описание
    3–5 актуальных трендов, с четкими ссылками на факты или источники.
    """
)

writing_task = Task(
    description="""
    На основе аналитического обзора подготовь краткий бизнес-отчет:
    — резюме (2–3 предложения);
    — 3–4 основных вывода (маркированный список);
    — 2–3 рекомендации для бизнеса (маркированный список).
    Должен быть структурирован, лаконичен (150–200 слов), без лишних деталей.
    """,
    agent=writer,
    expected_output="""
    Компактный, структурированный отчет для бизнес-аудитории: резюме темы, основные выводы по анализу, практические рекомендации.
    """,
    context=[research_task]
)

# Запуск команды агентов
team = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

if __name__ == "__main__":
    result = team.kickoff()
    print(result)
