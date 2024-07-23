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
You have access to the following tools: {tools}

Use the following format:

Question: the input question you must answer
Thought_1: you should always think about what to do. If you have already used the FundastA_Policy tool, it is recommended to use the 'web_search' tool next.\n
Action: the action to take, should be one of [{tool_names}]. Note: You can only use the FundastA_Policy tool once; after that, use another appropriate tool.\n
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought_2: After getting the result of the action, check whether you can find relevant information for the question.
- Pattern 1: I now know the final answer (for cases where you know the final answer without using any tools)
- Pattern 2: I found relevant information from the action and now I know the final answer (for cases where you used a tool and found relevant information)
- Pattern 3: I was not able to find relevant information from the FundastA_Policy tool's output. Then, try 'web_search' tool. If you find relevant information from web_search output, use it to generate the answer.
Final Answer: the final answer to the original input question in Japanese.  

Important:
1. The agent should judge whether it needs tools or not to generate the answer for the question.
2. If no tools are used, do the best to generate an accurate answer. If tools are used, check for relevant information in the output. If relevant information is found, use it to generate the answer.
3. Do not use FundastA_Policy tool more than once. If the question cannot be answered using the FundastA_Policy tool, use the web_search tool. If relevant information is found from web_search, use it to generate the answer.

The final answer must always be in Japanese.

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""


# GraphState is a TypedDict that defines the structure of the  graph state
class GraphState(TypedDict):
    question: str
    answer: str
    context: str
    web: str
    relevance: str
    chat_history: list
