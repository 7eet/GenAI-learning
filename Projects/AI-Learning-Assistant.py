import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List



load_dotenv()

st.set_page_config(
    page_title="AI Learning Assistant",
    layout="centered"
)

st.title("AI Learning Assistant")

st.write(
    "Enter any topic and I'll explain it in a way you want to."
)

topic = st.text_area(
    "What do you want to learn?",
    placeholder="Enter the Topic",
    height=120
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)




difficulty = [
    "Beginner",
    "Explain like I'm 10",
    "Intermediate",
    "Advanced"
]

selected_difficulty = st.selectbox(
    "Select a Difficulty:",
    difficulty
)



language = [
    "English",
    "Punjabi",
    "Hindi",
    "German",
    "French",
    "Italian"
]

selected_language = st.selectbox(
    "Select a Language:",
    language
)





style = [
    "Simple Explanation",
    "Use Analogies",
    "Storytelling",
    "Real-World Examples",
    "Technical Explanation"
]

selected_style = st.selectbox(
    "Select a Teaching Style:",
    style
)



class Explanation(BaseModel):
    topic: str = Field(
        description="The topic being explained"
    )

    what_is_it: str = Field(
        description="A simple explanation of what the topic is"
    )

    how_it_works: str = Field(
        description="A simple step-by-step explanation of how it works"
    )

    real_world_example: str = Field(
        description="A real-world example that makes the topic easier to understand"
    )

    analogy: str = Field(
        description="A simple analogy suitable for the selected difficulty level"
    )

    key_points: List[str] = Field(
        description="Three to five important things to remember"
    )


system_prompt = """
You are a friendly and patient AI teacher.

Your job is to explain the user's requested topic
according to the difficulty level, language, and
teaching style selected by the user.

Follow these rules:

- Adapt your vocabulary to the selected difficulty.
- Use the selected language.
- Follow the selected teaching style.
- Break complicated ideas into understandable sections.
- Explain technical terms when appropriate.
- Do not assume knowledge beyond the selected difficulty.
- Focus on helping the user understand WHY and HOW something works.
- Use examples and analogies when appropriate.

The explanation must contain:

1. What the topic is
2. How it works
3. A real-world example
4. A simple analogy
5. 3-5 key points to remember
"""


dynamic_prompt = """
The user wants to learn about:

Topic: {topic}

Language: {language}

Difficulty Level: {difficulty}

Teaching Style: {style}

Adapt your explanation to these preferences.
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", dynamic_prompt)
])



structured_llm = llm.with_structured_output(
    Explanation
)

chain = prompt | structured_llm


if st.button(
    "🧠 Explain This",
    use_container_width=True
):

    if not topic.strip():

        st.warning(
            "Please enter a topic you want to learn."
        )

    else:

        with st.spinner(
            "🤔 Thinking of the simplest explanation..."
        ):

            try:

                response = chain.invoke({
                    "topic": topic,
                    "difficulty": selected_difficulty,
                    "language": selected_language,
                    "style": selected_style
                })

                st.divider()


                st.header(
                    f"📚 {response.topic}"
                )


                st.subheader(
                    "🤔 What is it?"
                )

                st.write(
                    response.what_is_it
                )


                st.subheader(
                    "🧠 How does it work?"
                )

                st.write(
                    response.how_it_works
                )

                st.subheader(
                    "🌎 Real-world example"
                )

                st.write(
                    response.real_world_example
                )


                st.subheader(
                    "💡 Simple analogy"
                )

                st.write(
                    response.analogy
                )


                st.subheader(
                    "⭐ Key things to remember"
                )

                for point in response.key_points:

                    st.markdown(
                        f"- {point}"
                    )

            except Exception as e:

                st.error(
                    "Something went wrong while "
                    "generating the explanation."
                )

                st.exception(e)
