from evolution_openai import OpenAI
from crewai import Agent, Task, Crew, BaseLLM
import os
from dotenv import load_dotenv

load_dotenv()

# Конфигурация
config = {
    "key_id": os.getenv("API_TOKEN"),
    "secret": os.getenv("API_SECRET"),
    "project_id": os.getenv("PROJECT_ID"),
    "model": os.getenv("MODEL"),
    "base_url": os.getenv("BASE_URL")
}


class EvolutionLLM(BaseLLM):
    """Адаптер evolution-openai для CrewAI"""

    def __init__(self):
        super().__init__(model=config["model"])
        self.client = OpenAI(
            key_id=config["key_id"],
            secret=config["secret"],
            project_id=config["project_id"],
            base_url=config["base_url"]
        )

    def call(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = self.client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 500)
        )
        return response.choices[0].message.content


# Инициализация адаптера
llm = EvolutionLLM()

# Создание агентов
researcher = Agent(
    role="Исследователь",
    goal="Анализировать темы и собирать ключевую информацию",
    backstory="""
    Вы опытный аналитик-исследователь. Умеете быстро разбираться в сложных темах
    и выделять самую важную информацию. Ваши анализы всегда структурированы и точны.
    """,
    llm=llm,
    verbose=True
)

# Определение задач
research_task = Task(
    description="""
    Проанализируйте тему: "Применение ИИ в российском бизнесе"
    Выясните:
    1. Основные направления использования
    2. Ключевые компании и решения
    3. Главные тренды
    """,
    agent=researcher,
    expected_output="Аналитический обзор с ключевыми выводами"
)

writer = Agent(
    role="Технический писатель",
    goal="Создавать понятные и структурированные тексты на основе анализа",
    backstory="""
    Вы профессиональный технический писатель. Превращаете сложную информацию
    в понятные и хорошо структурированные тексты. Ваш стиль — четкий и деловой.
    """,
    llm=llm,
    verbose=True
)

writing_task = Task(
    description="""
    На основе проведённого исследования создайте краткий отчёт:
    1. Краткое резюме (2–3 предложения)
    2. Основные выводы (3–4 пункта)
    3. Рекомендации (2–3 пункта)
    Стиль: деловой, конкретный
    Объём: 150–200 слов
    """,
    agent=writer,
    expected_output="Структурированный отчёт",
    context=[research_task]
)

# Запуск команды агентов
team = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

result = team.kickoff()
print(result)
