import asyncio, os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from evolution_openai import OpenAI

load_dotenv()

config = {
    "key_id": os.getenv("API_TOKEN"),
    "secret": os.getenv("API_SECRET"),
    "project_id": os.getenv("PROJECT_ID"),
    "model": os.getenv("MODEL"),
    "base_url": os.getenv("BASE_URL")
}

class EvolutionChatService(ChatCompletionClientBase):
    def __init__(self, service_id: str):
        super().__init__(service_id=service_id, ai_model_id=config["model"])
        self._client = OpenAI(
            key_id=config["key_id"], secret=config["secret"],
            project_id=config["project_id"], base_url=config["base_url"]
        )
        self._model = config["model"]

    async def get_chat_message_contents(self, chat_history: ChatHistory, settings: PromptExecutionSettings, **_):
        messages = [{"role": m.role.value, "content": str(m.content)} for m in chat_history.messages]
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model=self._model, messages=messages,
            temperature=getattr(settings, "temperature", 0.7),
            max_tokens=getattr(settings, "max_tokens", 500)
        )
        return [ChatMessageContent(role="assistant", content=response.choices[0].message.content)]

async def main():
    kernel = Kernel()
    svc = EvolutionChatService("evolution")
    kernel.add_service(svc)

    researcher = ChatCompletionAgent(
        service=svc, kernel=kernel, name="researcher",
        instructions="""
        Вы опытный аналитик-исследователь. Умеете быстро разбираться в сложных темах
        и выделять самую важную информацию. Ваши анализы всегда структурированы и точны
        """
    )
    writer = ChatCompletionAgent(
        service=svc, kernel=kernel, name="writer",
        instructions="""
        Вы профессиональный технический писатель. Превращаете сложную информацию
        в понятные и хорошо структурированные тексты. Ваш стиль — четкий и деловой.
        """
    )

    print("Исследователь начал работу...")
    async for step in researcher.invoke(
        'Проанализируйте тему: "Применение ИИ в российском бизнесе" '
        '1) Направления, 2) Ключевые компании, 3) Тренды'
    ):
        research = step.message.content
    print("\nОтвет агента <researcher>:\n")
    print(research)

    print("\nТехнический писатель начал работу...")
    async for step in writer.invoke(
        f'Используя этот текст:\n{research}\n'
        'Сформируйте отчёт 150–200 слов:\n'
        '1. Краткое резюме\n2. 3–4 основных вывода\n3. 2–3 рекомендации'
    ):
        report = step.message.content
    print("\nОтвет агента <writer>:\n")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
