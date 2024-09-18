from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

GOOGLE_GENAI_API_KEY = config("GOOGLE_GENAI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_GENAI_API_KEY)

print("Q & A with AI")
print("=============")

question ="What is the currency of Thailand?"
print("Question: " + question)

response = llm.invoke(question)
print("Answer: " + response.content)