import os
from langchain_ollama import ChatOllama
from agent.schemas import PlannerOutput

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
)
planner_llm = llm.with_structured_output(PlannerOutput)
