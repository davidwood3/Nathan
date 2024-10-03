from decouple import config
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = config("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

print("Q & A with AI")
print("=============")

question ="What is the currency of Michigan?"
print("Question: " + question)

response = llm.invoke(question)
print("Answer: " + response.content)
