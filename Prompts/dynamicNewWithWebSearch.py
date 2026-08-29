import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List


# --------------------------------
# Load Environment Variables
# --------------------------------

load_dotenv()


# --------------------------------
# Streamlit Configuration
# --------------------------------

st.set_page_config(
    page_title="Top 5 News",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Top 5 Latest News")

st.write(
    "Select a topic and country to find the latest news."
)


# --------------------------------
# Topic Selection
# --------------------------------

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


# --------------------------------
# Country Selection
# --------------------------------

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


# --------------------------------
# Gemini LLM
# --------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# --------------------------------
# Tavily Web Search
# --------------------------------

tavily_search = TavilySearch(
    max_results=10,
    topic="news"
)


# --------------------------------
# Structured Output Models
# --------------------------------

class NewsItem(BaseModel):

    title: str = Field(
        description="The title of the news article"
    )

    published_date: str = Field(
        description="The publication date of the article"
    )

    source: str = Field(
        description="The name of the news source or publisher"
    )

    summary: str = Field(
        description="A concise 4-5 sentence summary of the article"
    )

    url: str = Field(
        description="The original URL of the news article"
    )


class NewsResponse(BaseModel):

    news: List[NewsItem] = Field(
        description="List of the top 5 news articles"
    )


# --------------------------------
# System Prompt
# --------------------------------

system_prompt = """
You are a reliable news research assistant.

You will receive search results from Tavily.

Your job is to select the top 5 newest and most relevant
news articles based on the user's selected topic and country.

For every article, provide:

- Title
- Published date
- Source name
- 4-5 sentence summary
- Original article URL

IMPORTANT RULES:

1. Return exactly 5 news articles.
2. Use ONLY information contained in the provided search results.
3. Do NOT invent news.
4. Do NOT invent publication dates.
5. Do NOT invent source names.
6. Do NOT invent URLs.
7. Prefer the newest articles.
8. Prefer reliable and relevant sources.
9. Avoid duplicate articles.
10. Make sure the URL corresponds to the selected article.
11. If a publication date is not available in the search results,
    use "Date not available".
12. If the source name is not available, use "Source not available".
"""


# --------------------------------
# Dynamic Prompt
# --------------------------------

dynamic_prompt = """
Find the top 5 newest and most relevant news articles.

Topic: {topic}

Country: {country}

Focus specifically on news related to the selected country.

Here are the web search results from Tavily:

{search_results}

Select the 5 best and newest articles from these results.
"""


# --------------------------------
# Prompt Template
# --------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", dynamic_prompt)
])


# --------------------------------
# Structured LLM
# --------------------------------

structured_llm = llm.with_structured_output(
    NewsResponse
)


# --------------------------------
# Create LangChain Chain
# --------------------------------

chain = prompt | structured_llm


# --------------------------------
# Generate News
# --------------------------------

if st.button(
    "🔎 Generate Top 5 News",
    use_container_width=True
):

    with st.spinner("Searching the web for latest news..."):

        # ----------------------------
        # Step 1: Search the Web
        # ----------------------------

        search_query = (
            f"latest {selected_topic} news "
            f"in {selected_country}"
        )

        search_results = tavily_search.invoke({
            "query": search_query
        })


        # ----------------------------
        # Step 2: Send Search Results
        # to Gemini
        # ----------------------------

        response = chain.invoke({
            "topic": selected_topic,
            "country": selected_country,
            "search_results": search_results
        })


        # ----------------------------
        # Step 3: Display Header
        # ----------------------------

        st.subheader(
            f"📰 Top 5 {selected_topic} News - "
            f"{selected_country}"
        )


        # ----------------------------
        # Step 4: Display News
        # ----------------------------

        for index, article in enumerate(
            response.news,
            start=1
        ):

            st.markdown(
                f"### {index}. {article.title}"
            )

            st.markdown(
                f"📅 **Published:** "
                f"{article.published_date}"
            )

            st.markdown(
                f"📰 **Source:** "
                f"{article.source}"
            )

            st.markdown(
                f"📝 **Summary:** "
            )

            st.write(article.summary)

            st.link_button(
                "🔗 Read More",
                article.url
            )

            st.divider()