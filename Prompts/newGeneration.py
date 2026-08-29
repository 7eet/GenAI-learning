import streamlit as st


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv();


st.set_page_config(
    page_title="AI News",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Top 5 AI News")

st.write(
    "Click the button below to find the latest and most important AI news."
)


# ---------------------------------
# Create the LLM
# ---------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

search = TavilySearch(
    max_results=10,
    topic="news",
    time_range="week"
)


system_prompt = """
You are an AI News Assistant.

Your job is to identify the top 5 most important and relevant
artificial intelligence news stories from the search results provided
to you.

Rules:

1. Return exactly 5 news stories.

2. Prioritize:
   - Recent AI news
   - Major AI company announcements
   - New AI models
   - Generative AI developments
   - AI agents
   - AI research
   - AI products
   - AI regulation
   - AI infrastructure
   - Major funding or acquisitions


4. Do not invent news, headlines, dates, sources, or URLs.

5. Only use information contained in the search results.

6. Do not include duplicate stories about the same event.

7. Give a short 4-5 sentence summary for every story.

8. Include the original article URL.

9. Rank the stories from most important to least important.

10. Make the response easy to read.
11. Show the date of new as well.
"""

if st.button("🔎 Get Latest AI News"):

    with st.spinner("Searching for the latest AI news..."):

        # Search the web
        search_results = search.invoke(
            "latest artificial intelligence AI news major announcements"
        )

        # Send search results to the LLM
        messages = [
            SystemMessage(content=system_prompt),

            HumanMessage(
                content=f"""
Here are the latest web search results:

{search_results}

Using ONLY these search results, identify and summarize
the top 5 most important AI news stories.
"""
            )
        ]

        # Invoke the model
        response = llm.invoke(messages)

    # Display result
    st.markdown(response.content)
