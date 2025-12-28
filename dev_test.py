import os
from unifiedllm import LLMClient

os.environ["GEMINI_API_KEY"] = "dummy"  # just for testing

gemini_client = LLMClient(provider="gemini", model="gemini-2.0-flash")
openai_client = LLMClient(provider="openai", model="gpt-3.5-turbo")

client = LLMClient(provider="gemini", model="gemini-2.0-flash")
resp = client.chat(prompt="Explain RAG in one line.")
print(resp.text)

resp2 = client.chat(
    messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is LangGraph?"},
    ]
)
print(resp2.text)
