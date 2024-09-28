from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Adding History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.messages.tool import ToolMessage

# llm = ChatOllama(model="gemma:2b")
GOOGLE_API_KEY = config("GOOGLE_GENAI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, verbose=True)
verbose = True


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI chatbot having a conversation with a human. Use the following context to understand the human question. Show your thinking.Do not include emojis in your answer. ",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
history = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.title("Q & A With AI")

for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("Your Question")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    response = chain_with_history.stream(
        {"input": question}, config={"configurable": {"session_id": "any"}}
    )
    with st.chat_message("assistant"):
        st.write_stream(response)



