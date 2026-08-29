import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


st.set_page_config(
    page_title="Top 5 News",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Top 5 Latest News")
st.write("Select a topic and country to generate the top 5 news items.")


topics = [
    "Artificial Intelligence",
    "Technology",
    "Business",
    "Finance",
    "Stock Market",
    "Cryptocurrency",
    "Science",
    "Space",
    "Healthcare",
    "Sports",
    "Entertainment",
    "Movies",
    "Gaming",
    "Politics",
    "Education",
    "Travel",
    "Climate & Environment",
    "Automobiles",
    "Startups",
    "Cybersecurity"
]

selected_topic = st.selectbox(
    "Select a topic:",
    topics
)

countries = [
    "India",
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "Germany",
    "France",
    "Japan",
    "Singapore",
    "United Arab Emirates"
]

selected_country = st.selectbox(
    "Select a country:",
    countries
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# System Prompt
system_prompt = """
You are a helpful and reliable news research assistant.

Your job is to provide the top 5 newest and most important
news items based on the user's selected topic and country.

For each news item, provide:

Title
Published Date
Summary
Read more

Formatting requirements:
- Put each field on a separate line.
- Leave a blank line between each field.
- Leave a blank line between news items.
- Use Markdown formatting.
- Add a divider line after Each New
- Format each news item like this:

### [Title]

**Published Date:** [Date]

**Summary:** [4-5 sentence summary]

**Read more:** [Original article URL]

Requirements:
- Return exactly 5 news items.
- Prioritize recent and important news.
- Focus specifically on the selected country.
- Do not make up news.
- Do not make up dates.
- Do not make up URLs.
- If information cannot be verified, do not include it.
"""


dynamic_prompt = """
Find the top 5 newest and most relevant news items.

Topic: {topic}
Country: {country}

Focus specifically on news related to the selected country.

Return exactly 5 items.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", dynamic_prompt)
])



# Create LangChain Chain

chain = prompt | llm

if st.button("🔎 Generate Top 5 News", use_container_width=True):

    with st.spinner("Finding the latest news..."):

        response = chain.invoke({
            "topic": selected_topic,
            "country": selected_country
        })

        st.subheader(
            f"📰 Top 5 {selected_topic} News - {selected_country}"
        )

        st.write(response.content)