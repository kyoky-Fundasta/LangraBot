# import os
# from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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
env_google = st.secrets["Google_API_KEY_An"]
CSE_ID = st.secrets["Search_Engine_ID_An"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
client_id = st.secrets["CLIENT_ID"]


def llm_switch(selected_model):

    if selected_model == "Gemini_1.5_Flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=env_genai,
            temperature=0,
            convert_system_message_to_human=True,
            streaming=True,
        )
    elif selected_model == "ChatGPT_3.5":
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-0125",
            openai_api_key=env_openai,
            streaming=True,
        )
    elif selected_model == "ChatGPT_4o_mini":
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            openai_api_key=env_openai,
            streaming=True,
        )

    return llm


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
    response_type: str
