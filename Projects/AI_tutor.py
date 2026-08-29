import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv



load_dotenv()

st.set_page_config(
    page_title="Explain Like I'm 10",
    layout="centered"
)

st.title("🧒 Explain Like I'm 10")

st.write(
    "Enter any topic and I'll explain it in a simple way, "
    "like I'm explaining it to a 10-year-old."
)

topic = st.text_area(
    "What do you want to learn?",
    placeholder="Example: How does the Internet work?",
    height=120
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

system_prompt = """
You are a friendly and patient teacher.

Your job is to explain difficult topics as if you are
explaining them to a curious 10-year-old child.

Follow these rules:

- Use very simple language.
- Avoid complicated technical words.
- If you must use a difficult word, explain it immediately.
- Use everyday examples and analogies.
- Break complicated ideas into small steps.
- Use short paragraphs.
- Make the explanation engaging and easy to understand.
- Use emojis when they help understanding.
- Do not assume the user has prior knowledge.
- Focus on helping the user understand WHY and HOW something works.

Structure your answer like this:

# 📚 Topic

## 🤔 What is it?

Give a simple explanation.

## 🧠 How does it work?

Explain the concept step by step.

## 🌎 Real-world example

Give an example from everyday life.

## 💡 Simple analogy

Use an easy analogy that a 10-year-old would understand.

## ⭐ Key things to remember

Give 3-5 important points.

Remember:
You are teaching, not just giving a definition.
The goal is understanding.
"""

dynamic_prompt = """
The user wants to learn about the following topic:

{topic}

Explain this topic using the teaching style and structure
defined in the system instructions.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", dynamic_prompt)
])



chain = prompt | llm


if st.button(
    "🧠 Explain This",
    use_container_width=True
):

    if not topic.strip():

        st.warning(
            "Please enter a topic you want to learn."
        )

    else:

        with st.spinner("🤔 Thinking of the simplest explanation..."):

            response = chain.invoke({
                "topic": topic
            })


        st.divider()

        st.markdown(
            "## 📖 Explanation"
        )

        st.markdown(
            response.content
        )
