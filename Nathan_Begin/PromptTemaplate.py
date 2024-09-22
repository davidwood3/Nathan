from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

import streamlit as st

llm = ChatOllama(model="gemma:2b")

st.title("Currency Info")

country = st.text_input("Input Country")

if country:
    question = "What is the currency of " + country + "?"
    response = llm.invoke(question)
    st.write(response.content)

