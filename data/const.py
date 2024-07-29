# import os
# from dotenv import load_dotenv
from typing import TypedDict
import streamlit as st

embedding_model = "text-embedding-3-small"
index_name = "fundasta"
gpt_model_name = "ChatGPT_3.5"
gpt_mini_model_name = "ChatGPT_4o_mini"
gemini_model_name = "Gemini_1.5_Flash"

env_tavily = st.secrets["Tavily_API_KEY_An"]
env_openai = st.secrets["OPENAI_API_KEY_FundastA"]
env_genai = st.secrets["Gemini_API_KEY_An"]
env_smith = st.secrets["LANGCHAIN_API_KEY"]
env_pinecone = st.secrets["Pinecone_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]


# GraphState is a TypedDict that defines the structure of the graph state
class GraphState(TypedDict):
    selected_model: str
    question: str
    answer: str
    context: str
    web: str
    relevance: str
    chat_history: list
    hint: str
    rewrotten_question: str
    rewrotten_question_answer: str
    reasoning: str
    source: str
