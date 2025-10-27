from typing import List, Dict
from ..settings import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the answer cannot be found in the context, say you don't know. "
    "Be concise and include inline citations like [1], [2] matching the sources."
)

class BaseLLM:
    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

class OpenAILLM(BaseLLM):
    def __init__(self, model: str, api_key: str | None):
        from openai import OpenAI
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)
        return resp.choices[0].message.content.strip()

class GeminiLLM(BaseLLM):
    def __init__(self, model: str, api_key: str | None):
        import google.generativeai as genai
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Flatten messages into a single prompt
        final = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            final.append(f"{role.upper()}: {content}")
        prompt = "\n".join(final)
        resp = self.model.generate_content(prompt)
        return (resp.text or "").strip()

class FakeLLM(BaseLLM):
    def generate(self, messages: List[Dict[str, str]]) -> str:
        # For tests: echo last user message with a short reply
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        return f"(fake) Based on context, I think: {last_user[:100]}"

def get_llm() -> BaseLLM:
    prov = settings.llm_provider.lower()
    if prov == "openai":
        return OpenAILLM(settings.llm_model, settings.openai_api_key)
    if prov == "gemini":
        return GeminiLLM(settings.gemini_model, settings.google_api_key)
    if prov == "fake":
        return FakeLLM()
    raise ValueError(f"Unsupported LLM_PROVIDER: {prov}")