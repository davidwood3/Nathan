from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

import streamlit as st

llm = ChatOllama(model="gemma:2b")

prompt = PromptTemplate(
    input_variables=["country", "paragraph", "language" ],
    template="What is the currency of {country}? Answer in {paragraph} short paragraphs in {language}",
)

prompt = PromptTemplate(
    input_variables=["country", "paragraph", "language"],
    template='''
    You are a currency expert. You give information about a specific currency used in a specific country.
    Avoid giving information about fictional places.
    If the country is fictional or non-existent, answer: I don't know.

    Answer the question: What is the currency of {country}?

    Answer in {paragraph} short paragraph in {language}
    ''',
)

st.title("Currency Info")

country = st.text_input("Input Country")
paragraph = st.number_input("Input Number of Paragraphs", min_value=1, max_value=5)
language = st.text_input("Input Language")

if country and paragraph and language:
    question = prompt.format(country=country, paragraph=paragraph, language=language)
    st.write(question)
    response = llm.invoke(question)
    st.write(response.content)

#
