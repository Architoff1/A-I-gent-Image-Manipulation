from agent.llm import llm


response = llm.invoke(
    "Explain in one sentence what an image editing AI agent does."
)

print(response.content)
