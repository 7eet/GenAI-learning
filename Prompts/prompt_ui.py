from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv();


st.set_page_config(
    page_title="Simple Chat App",
)

st.title("My Chat App")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


user_input = st.text_input(
    "Enter your question",
    placeholder="Ask me anything..."
)

if st.button("Send"):

    if user_input:

        response = llm.invoke(user_input)

        st.subheader("Response")
        st.write(response.content)

    else:
        st.warning("Please enter a question.")
