from langchain_ollama import ChatOllama
import streamlit as st

llm = ChatOllama(model="gemma:2b")

st.title("Q & A with AI")

question = st.text_input("Question: ")

if question:
    response = llm.invoke(question)
    st.write("Answer: ", response.content)
