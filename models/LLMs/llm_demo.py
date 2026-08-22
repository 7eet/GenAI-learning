from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the captial of India?")

print(result);
