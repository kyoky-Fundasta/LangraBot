# import os
# from dotenv import load_dotenv
from typing import TypedDict
import streamlit as st

embedding_model = "text-embedding-3-small"
index_name = "fundasta"
gpt_model_name = "gpt-3.5-turbo-0125"
gemini_model_name = "gemini-1.5-flash"

env_tavily = st.secrets["Tavily_API_KEY_An"]
env_openai = st.secrets["OPENAI_API_KEY_FundastA"]
env_genai = st.secrets["Gemini_API_KEY_An"]
env_smith = st.secrets["LANGCHAIN_API_KEY"]
env_pinecone = st.secrets["Pinecone_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]

agent_prompt_mod = """Answer the following questions as best you can. 
You have access to the following tools:\n\n{tools}\n\n
Use the following format:\n\n
Question: the input question you must answer\n
Thought_1: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\n
Observation: the result of the action\n
... (this Thought/Action/Action Input/Observation can repeat N times)\n
Thought_2: After getting the result of the action, you must check whether you can find relevant information for the question.
- Pattern 1: I now know the final answer (For the case you know the final answer without using any tools)
- Pattern 2: I found relevant information from the action and now I know the final answer (For the case you used a tool and found relevant information)
Final Answer: the final answer to the original input question in Japanese. It is \n\n
Important:
1. If you did not use any tools, generate a general answer.
2. When you used a tool, you must check whether you can find any relevant information from the tool's output for the question. If you find any relevant information, you must use it to generate the final answer.

The final answer must always be in Japanese.

Begin!\n\n
Question: {input}\n
Thought:{agent_scratchpad}
"""


# GraphState is a TypedDict that defines the structure of the  graph state
class GraphState(TypedDict):
    question: str
    answer: str
    context: str
    web: str
    relevance: str
    chat_history: list
