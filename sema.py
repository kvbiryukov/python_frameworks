import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from openai import OpenAI

load_dotenv()

API_KEY  = os.getenv("API_KEY")
MODEL    = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

TEMPERATURE = os.getenv("TEMPERATURE")
MAX_TOKENS = os.getenv("MAX_TOKENS")

class AgentService(ChatCompletionClientBase):

    def __init__(self, service_id: str):
        super().__init__(service_id=service_id, ai_model_id=MODEL)
        self._client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        self._model = MODEL

    async def get_chat_message_contents(self, chat_history: ChatHistory, settings: PromptExecutionSettings, **_):
        messages = [{"role": m.role.value, "content": str(m.content)} for m in chat_history.messages]
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return [ChatMessageContent(role="assistant", content=response.choices[0].message.content)]

async def main():
    kernel = Kernel()
    svc = AgentService("cloudru")
    kernel.add_service(svc)

    researcher = ChatCompletionAgent(
        service=svc,
        kernel=kernel,
        name="researcher",
        instructions="""
        Роль: Исследователь.
        Цель: Провести структурированный анализ применения ИИ в российском бизнесе (2023–2025).
        Стиль: Кратко, с фактами и ссылками, в виде списка направлений, компаний и трендов.
        """
    )

    writer = ChatCompletionAgent(
        service=svc,
        kernel=kernel,
        name="writer",
        instructions="""
        Роль: Технический писатель.
        Цель: На основе анализа подготовить деловой отчёт (150–200 слов):
         1) резюме (2–3 предложения);
         2) 3–4 вывода;
         3) 2–3 рекомендации.
        Стиль: Лаконично, понятно для руководителей.
        """
    )

    print("Исследователь начал работу...")
    research = ""
    async for step in researcher.invoke(
        """
        Собери актуальный аналитический обзор по теме: «Применение ИИ в российском бизнесе в 2023–2025».
        Требуется описать:
        — главные направления использования (3-4);
        — ключевые компании и конкретные решения (3-4);
        — основные тренды и вызовы (3-4).
        Оформи анализ как структурированный список с пояснениями. Ответ до 400 слов
        """
    ):
        research = step.message.content
        print("\nОтвет исследователя:\n", research)

    print("\nПисатель начал работу...")
    report = ""
    async for step in writer.invoke(
        f'Используя этот текст:\n{research}\n'
        """
        Подготовь краткий бизнес-отчет:
        — резюме (2–3 предложения);
        — 3–4 основных вывода (маркированный список);
        — 2–3 рекомендации для бизнеса (маркированный список).
        Должен быть структурирован, лаконичен (150–200 слов), без лишних деталей.
        """
    ):
        report = step.message.content
        print("\nОтчёт:\n", report)

if __name__ == "__main__":
    asyncio.run(main())

