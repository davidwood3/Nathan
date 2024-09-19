from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma:2b")

print("Q & A with AI")
print("=============")

question =""
print("Question: " + question)
response= llm.invoke(question)


print("Answer: " + response.content)


