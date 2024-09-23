from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from langchain.globals import set_debug
set_debug(True)

GOOGLE_GEMINI_KEY = config("GOOGLE_GENAI_API_KEY")
OPENAI_KEY = config("OPENAI_API_KEY")
#
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_GEMINI_KEY)
llm2 = ChatOpenAI(temperature=0, model_name="gpt-4o", api_key=OPENAI_KEY)

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an expert journalist.

    You need to come up with an interesting title for the following topic: {topic}

    Answer exactly with one title
    """,
)

essay_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""
    You are an expert nonfiction writer.

    You need to write a short essay of 350 words for the following title:

    {title}

    Make sure that the essay is engaging and makes the reader feel {emotion}.

    Format the output as a JSON object with three keys: 'title', 'emotion', 'essay' and fill them with respective values
    """,
)

# Use a different prompt for each chain
first_chain = title_prompt | llm | StrOutputParser()
second_chain = essay_prompt | llm2 | JsonOutputParser()


overall_chain = (
    first_chain
    | (lambda title: {"title": title, "emotion": emotion})
    | second_chain
)

st.title("Multiple LLMs - Essay Writer")

topic = st.text_input("Input Topic")
emotion = st.text_input("Input Emotion")

if topic and emotion:
    response = overall_chain.invoke({"topic": topic})
    st.write(response)

